# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
#
# End-to-end: the REAL DSparkDrafter (random weights -> drafts poorly) against a toy
# order-1 Markov base adapter. The lossless guarantee holds regardless of draft quality:
# the emitted stream must equal plain greedy decoding from the base. No reference needed.
from __future__ import annotations

import mlx.core as mx
import numpy as np

from dspark_mlx.adapter import BlockOut, StepOut
from dspark_mlx.events import SummaryEvent, TokenEvent
from dspark_mlx.generate import generate
from dspark_mlx.model.config import DSparkArgs
from dspark_mlx.model.drafter import DSparkDrafter

VOCAB, DIM, N_TARGETS = 32, 16, 2
DRAFTER_ARGS = dict(
    vocab_size=VOCAB, dim=DIM, n_heads=4, head_dim=16, rope_head_dim=8, q_lora_rank=16,
    o_lora_rank=8, o_groups=2, window_size=8, norm_eps=1e-6, rope_theta=10000.0, n_layers=2,
    n_mtp_layers=1, n_routed_experts=4, moe_inter_dim=16, n_activated_experts=2,
    n_shared_experts=1, n_hash_layers=0, score_func="sqrtsoftplus", route_scale=1.5,
    swiglu_limit=10.0, hc_mult=4, hc_sinkhorn_iters=20, dspark_block_size=4,
    dspark_noise_token_id=VOCAB - 1, dspark_target_layer_ids=[0, 1], dspark_markov_rank=8,
    temperature=0.0,
)


class MarkovAdapter:
    """Order-1 Markov base: next-token logits depend only on the last token."""

    target_layer_ids = (0, 1)

    def __init__(self, transition: np.ndarray, embed: np.ndarray):
        self.transition = transition  # [V, V] logits
        self.embed = embed            # [V, D] main hidden per token

    def prefill(self, tokens: mx.array) -> StepOut:
        toks = np.array(tokens)[0]
        logits = mx.array(self.transition[int(toks[-1])][None, :])
        return StepOut(logits=logits, main_hidden=mx.array(self.embed[toks][None, :, :]))

    def decode_step(self, token: mx.array) -> StepOut:
        tk = int(np.array(token)[0])
        return StepOut(logits=mx.array(self.transition[tk][None, :]),
                       main_hidden=mx.array(self.embed[tk][None, :]))

    def verify_forward(self, block_tokens: mx.array) -> BlockOut:
        d = np.array(block_tokens)[0]  # [K]
        return BlockOut(
            per_pos_logits=mx.array(self.transition[d][None, :, :]),
            per_pos_main_hidden=mx.array(self.embed[d][None, :, :]),
            main_hidden_last=mx.array(self.embed[int(d[-1])][None, :]),
        )

    def kv_snapshot(self):
        return None

    def kv_rollback(self, n_keep: int) -> None:
        return None


def _plain_greedy(transition: np.ndarray, start: int, steps: int) -> list[int]:
    out, last = [], start
    for _ in range(steps):
        last = int(transition[last].argmax())
        out.append(last)
    return out


def test_generate_is_lossless() -> None:
    rng = np.random.default_rng(0)
    drafter = DSparkDrafter(DSparkArgs.from_dict(DRAFTER_ARGS), max_seq_len=64)
    transition = rng.standard_normal((VOCAB, VOCAB)).astype(np.float32)
    embed = (rng.standard_normal((VOCAB, DIM * N_TARGETS)) * 0.1).astype(np.float32)
    adapter = MarkovAdapter(transition, embed)

    prompt = rng.integers(0, VOCAB, size=(1, 6)).astype(np.int32)
    events = list(generate(adapter, drafter, prompt, max_new_tokens=20))
    tokens = [e.token for e in events if isinstance(e, TokenEvent)]
    summary = events[-1]

    assert isinstance(summary, SummaryEvent)
    assert len(tokens) == 20
    assert tokens == _plain_greedy(transition, int(prompt[0, -1]), 20)


def test_advance_matches_prefill() -> None:
    # Sliding the window over the L-th token (advance) == prefilling all L tokens.
    rng = np.random.default_rng(1)
    drafter = DSparkDrafter(DSparkArgs.from_dict(DRAFTER_ARGS), max_seq_len=64)
    big = 7  # < window_size (8): no wraparound, so slot indices are stable
    main_full = (rng.standard_normal((1, big, DIM * N_TARGETS)) * 0.3).astype(np.float32)
    anchor = mx.array([3], dtype=mx.int32)

    drafter.forward_spec(anchor, mx.array(main_full), start_pos=0)
    win_full = np.array(drafter.blocks[0].attn.cache.read())

    for blk in drafter.blocks:
        blk.attn.cache.window = None
    drafter.forward_spec(anchor, mx.array(main_full[:, : big - 1]), start_pos=0)
    drafter.advance(mx.array(main_full[:, big - 1]), big - 1)
    win_adv = np.array(drafter.blocks[0].attn.cache.read())

    assert np.max(np.abs(win_full[:, : big] - win_adv[:, : big])) <= 1e-4
