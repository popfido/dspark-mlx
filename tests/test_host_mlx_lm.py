# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
#
# MlxLmHostAdapter: a real mlx-lm Qwen3 decoder (tiny random weights, no checkpoint) used as
# the DSpark base. Losslessness holds for *any* drafter quality, so a tiny random drafter
# proves the adapter + generate integration emits exactly base greedy. Also checks the
# hidden-capture shape and KV trim/rollback bookkeeping.
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from dspark_mlx.events import SummaryEvent, TokenEvent
from dspark_mlx.generate import generate
from dspark_mlx.hosts.mlx_lm import MlxLmHostAdapter
from dspark_mlx.registry import resolve_arch

HID, N_BASE_LAYERS, VOCAB, N_HEADS, N_KV, HEAD_DIM = 64, 6, 256, 4, 2, 16
TIDS = (1, 3)  # residual-stream outputs of base layers 1 and 3 (both below the final layer)
BLOCK = 4


def _tiny_qwen3_base():
    from mlx_lm.models.qwen3 import Model, ModelArgs

    args = ModelArgs(
        model_type="qwen3", hidden_size=HID, num_hidden_layers=N_BASE_LAYERS,
        intermediate_size=HID * 2, num_attention_heads=N_HEADS, rms_norm_eps=1e-6,
        vocab_size=VOCAB, num_key_value_heads=N_KV, max_position_embeddings=4096,
        rope_theta=1_000_000.0, head_dim=HEAD_DIM, tie_word_embeddings=False,
    )
    model = Model(args)
    model.eval()
    return model


def _tiny_drafter():
    cfg = dict(
        model_type="qwen3", vocab_size=VOCAB, hidden_size=HID, num_hidden_layers=2,
        num_attention_heads=N_HEADS, num_key_value_heads=N_KV, head_dim=HEAD_DIM,
        intermediate_size=HID * 2, rms_norm_eps=1e-6, rope_theta=1_000_000.0,
        target_layer_ids=list(TIDS), num_target_layers=N_BASE_LAYERS, block_size=BLOCK,
        mask_token_id=VOCAB - 1, markov_rank=8, temperature=0.0,
    )
    return resolve_arch(cfg).build(cfg, max_seq_len=256)


def _base_greedy(adapter, prompt, steps):
    """Pure greedy decode through the adapter's single-token path — the lossless reference."""
    adapter.reset()
    step = adapter.prefill(mx.array(prompt))
    tok = int(mx.argmax(step.logits[0]).item())
    out = [tok]
    while len(out) < steps:
        step = adapter.decode_step(mx.array([tok], dtype=mx.int32))
        tok = int(mx.argmax(step.logits[0]).item())
        out.append(tok)
    return out


def test_main_hidden_shape() -> None:
    mx.random.seed(0)
    adapter = MlxLmHostAdapter(_tiny_qwen3_base(), target_layer_ids=TIDS)
    out = adapter.prefill(mx.array([[3, 7, 11, 5]], dtype=mx.int32))
    assert out.main_hidden.shape == (1, 4, HID * len(TIDS))   # concat over target layers
    assert out.logits.shape == (1, VOCAB)


def test_target_layer_ids_range_validated() -> None:
    mx.random.seed(0)
    with pytest.raises(ValueError, match="out of range"):
        MlxLmHostAdapter(_tiny_qwen3_base(), target_layer_ids=(1, N_BASE_LAYERS))


def test_kv_rollback_trims_to_n_keep() -> None:
    mx.random.seed(0)
    adapter = MlxLmHostAdapter(_tiny_qwen3_base(), target_layer_ids=TIDS)
    adapter.prefill(mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32))      # offset -> 5
    assert adapter.offset == 5
    adapter.verify_forward(mx.array([[9, 8, 7]], dtype=mx.int32))     # speculatively +3 -> 8
    assert adapter.offset == 8
    adapter.kv_rollback(6)                                            # keep 5 + 1 accepted
    assert adapter.offset == 6


def test_generate_is_lossless_against_real_qwen3() -> None:
    mx.random.seed(1234)
    base = _tiny_qwen3_base()
    drafter = _tiny_drafter()
    adapter = MlxLmHostAdapter(base, target_layer_ids=TIDS)

    rng = np.random.default_rng(0)
    prompt = rng.integers(0, VOCAB, size=(1, 7)).astype(np.int32)

    drafter.reset()
    events = list(generate(adapter, drafter, prompt, max_new_tokens=24))
    tokens = [e.token for e in events if isinstance(e, TokenEvent)]
    summary = events[-1]

    assert isinstance(summary, SummaryEvent)
    assert len(tokens) == 24
    assert tokens == _base_greedy(adapter, prompt, 24)
    # a real (if random) draft stack ran end-to-end: drafts were proposed and verified
    assert summary.n_drafted > 0
