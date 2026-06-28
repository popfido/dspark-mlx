# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
#
# Qwen3 DSpark end-to-end: (1) the real drafter (random weights) is lossless against a toy
# base via the shared generate loop; (2) the real published checkpoint loads through the
# registry's layers.* map and runs forward_spec to valid tokens.
from __future__ import annotations

import json
import os

import mlx.core as mx
import numpy as np
import pytest

from dspark_mlx.adapter import BlockOut, StepOut
from dspark_mlx.events import SummaryEvent, TokenEvent
from dspark_mlx.generate import generate
from dspark_mlx.loading import load_drafter
from dspark_mlx.registry import resolve_arch

VOCAB, HID, N_TARGETS = 24, 16, 2
FC_IN = HID * N_TARGETS
QCFG = dict(
    model_type="qwen3", vocab_size=VOCAB, hidden_size=HID, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, intermediate_size=24, num_hidden_layers=2,
    target_layer_ids=[0, 1], rms_norm_eps=1e-6, rope_theta=1_000_000.0, block_size=4,
    mask_token_id=VOCAB - 1, markov_rank=8, temperature=0.0,
)

_CKPT = "/Users/Fido/workspace/omlx/_research/dspark_multi/ckpt/qwen3_4b.safetensors"
_RCFG = "/Users/Fido/workspace/omlx/_research/dspark_multi/qwen3_4b_config.json"


class MarkovAdapter:
    target_layer_ids = (0, 1)

    def __init__(self, transition, embed):
        self.transition, self.embed = transition, embed

    def prefill(self, tokens):
        t = np.array(tokens)[0]
        return StepOut(mx.array(self.transition[int(t[-1])][None, :]),
                       mx.array(self.embed[t][None, :, :]))

    def decode_step(self, token):
        tk = int(np.array(token)[0])
        return StepOut(mx.array(self.transition[tk][None, :]), mx.array(self.embed[tk][None, :]))

    def verify_forward(self, block):
        d = np.array(block)[0]
        return BlockOut(mx.array(self.transition[d][None, :, :]),
                        mx.array(self.embed[d][None, :, :]),
                        mx.array(self.embed[int(d[-1])][None, :]))

    def kv_snapshot(self):
        return None

    def kv_rollback(self, n_keep):
        return None


def _plain_greedy(transition, start, steps):
    out, last = [], start
    for _ in range(steps):
        last = int(transition[last].argmax())
        out.append(last)
    return out


def test_qwen3_generate_is_lossless() -> None:
    rng = np.random.default_rng(0)
    drafter = resolve_arch(QCFG).build(QCFG, max_seq_len=128)
    transition = rng.standard_normal((VOCAB, VOCAB)).astype(np.float32)
    embed = (rng.standard_normal((VOCAB, FC_IN)) * 0.1).astype(np.float32)
    adapter = MarkovAdapter(transition, embed)

    prompt = rng.integers(0, VOCAB, size=(1, 6)).astype(np.int32)
    events = list(generate(adapter, drafter, prompt, max_new_tokens=20))
    tokens = [e.token for e in events if isinstance(e, TokenEvent)]
    assert isinstance(events[-1], SummaryEvent)
    assert tokens == _plain_greedy(transition, int(prompt[0, -1]), 20)


@pytest.mark.skipif(not os.path.exists(_CKPT), reason="real qwen3 draft checkpoint not present")
def test_qwen3_real_weights_load_and_run() -> None:
    weights = mx.load(_CKPT)
    config = json.load(open(_RCFG))
    arch = resolve_arch(config)
    assert arch.name == "qwen3"

    drafter = arch.build(config, max_seq_len=64)
    skipped = load_drafter(drafter, weights, key_map=arch.key_map)
    assert skipped == [], f"unmapped checkpoint keys: {skipped}"

    b, prompt_len = 1, 8
    fc_in = drafter.args.fc_in
    main_full = (mx.random.normal((b, prompt_len, fc_in)) * 0.1).astype(mx.bfloat16)
    drafter.forward_spec(mx.array([5], dtype=mx.int32), main_full, start_pos=0)
    main_dec = (mx.random.normal((b, 1, fc_in)) * 0.1).astype(mx.bfloat16)
    out = drafter.forward_spec(mx.array([5], dtype=mx.int32), main_dec, start_pos=prompt_len - 1)
    ids, logits, conf = out
    mx.eval(ids, logits, conf)

    a = np.array(ids)
    assert a.shape == (b, drafter.block_size + 1)
    assert (a >= 0).all() and (a < drafter.args.vocab_size).all()
    assert np.isfinite(np.array(logits)).all()
    assert np.isfinite(np.array(conf)).all()
