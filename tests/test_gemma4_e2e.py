# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
#
# Gemma4 DSpark: backbone parity vs real transformers Gemma4 primitives, the partial-RoPE
# table vs the real Gemma4TextRotaryEmbedding, lossless e2e against a toy base, and the real
# published checkpoint load/run.
from __future__ import annotations

import json
import os

import mlx.core as mx
import numpy as np
import torch
from mlx.utils import tree_flatten, tree_unflatten

from dspark_mlx.adapter import BlockOut, StepOut
from dspark_mlx.arch.gemma4 import Gemma4Backbone, Gemma4DSparkArgs, rope_tables
from dspark_mlx.events import SummaryEvent, TokenEvent
from dspark_mlx.generate import generate
from dspark_mlx.loading import load_drafter
from dspark_mlx.registry import resolve_arch
from gemma4_ref import RefGemma4Backbone
import pytest

ARGS = Gemma4DSparkArgs(
    hidden_size=32, num_attention_heads=4, num_key_value_heads=1, head_dim=16,
    intermediate_size=24, num_hidden_layers=2, target_layer_ids=(0, 1), rms_norm_eps=1e-6,
    rope_theta=1_000_000.0, partial_rotary_factor=0.25, attention_k_eq_v=True, block_size=3,
)
_CKPT = "/Users/Fido/workspace/omlx/_research/dspark_multi/ckpt/gemma4_12b.safetensors"
_RCFG = "/Users/Fido/workspace/omlx/_research/dspark_multi/gemma4_12b_config.json"


def test_gemma4_rope_matches_transformers() -> None:
    from transformers.models.gemma4 import modeling_gemma4 as m
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    cfg_d = json.load(open(_RCFG))
    cfg = Gemma4TextConfig(**{k: v for k, v in cfg_d.items() if k != "architectures"})
    rot = m.Gemma4TextRotaryEmbedding(cfg)
    pos = torch.arange(8)[None, :]
    cos_t, sin_t = rot(torch.zeros(1, 8, cfg.global_head_dim), pos, layer_type="full_attention")
    rope = cfg_d["rope_parameters"]["full_attention"]
    mc, ms = rope_tables(mx.arange(8), cfg.global_head_dim, rope["rope_theta"], rope["partial_rotary_factor"])
    mx.eval(mc, ms)
    assert np.max(np.abs(np.array(mc) - cos_t[0].numpy())) < 1e-4
    assert np.max(np.abs(np.array(ms) - sin_t[0].numpy())) < 1e-4


def test_gemma4_backbone_parity() -> None:
    rng = np.random.default_rng(0)
    mlx_bb = Gemma4Backbone(ARGS)
    ref_bb = RefGemma4Backbone(ARGS)
    flat = dict(tree_flatten(mlx_bb.parameters()))
    weights = {p: (rng.standard_normal(tuple(v.shape)) * 0.05).astype(np.float32) for p, v in flat.items()}
    mlx_bb.update(tree_unflatten([(p, mx.array(w)) for p, w in weights.items()]))
    ref_bb.load_state_dict({p: torch.tensor(w) for p, w in weights.items()}, strict=False)

    b, ctx, block = 1, 5, ARGS.block_size
    noise = (rng.standard_normal((b, block, ARGS.hidden_size)) * 0.3).astype(np.float32)
    target_hidden = (rng.standard_normal((b, ctx, ARGS.fc_in)) * 0.3).astype(np.float32)
    cos, sin = rope_tables(mx.arange(ctx + block), ARGS.head_dim, ARGS.rope_theta, ARGS.partial_rotary_factor)
    cos_np, sin_np = np.array(cos), np.array(sin)

    tctx = mlx_bb.project_context(mx.array(target_hidden))
    o_mlx = mlx_bb(mx.array(noise), tctx, cos, sin)
    mx.eval(o_mlx)
    with torch.no_grad():
        tctx_r = ref_bb.project_context(torch.tensor(target_hidden))
        o_ref = ref_bb(torch.tensor(noise), tctx_r, torch.tensor(cos_np), torch.tensor(sin_np)).numpy()
    assert np.max(np.abs(np.array(o_mlx) - o_ref)) <= 3e-3


# ---- lossless e2e against a toy Markov base ----
VOCAB, HID, N_TARGETS = 24, 16, 2
FC_IN = HID * N_TARGETS
GCFG = dict(
    model_type="gemma4_text", vocab_size=VOCAB, hidden_size=HID, num_attention_heads=4,
    num_key_value_heads=1, head_dim=16, intermediate_size=24, num_hidden_layers=2,
    target_layer_ids=[0, 1], rms_norm_eps=1e-6, rope_theta=1_000_000.0,
    partial_rotary_factor=0.25, attention_k_eq_v=True, final_logit_softcapping=30.0,
    block_size=4, mask_token_id=VOCAB - 1, markov_rank=8, temperature=0.0,
)


class MarkovAdapter:
    target_layer_ids = (0, 1)

    def __init__(self, transition, embed):
        self.transition, self.embed = transition, embed

    def prefill(self, tokens):
        t = np.array(tokens)[0]
        return StepOut(mx.array(self.transition[int(t[-1])][None, :]), mx.array(self.embed[t][None, :, :]))

    def decode_step(self, token):
        tk = int(np.array(token)[0])
        return StepOut(mx.array(self.transition[tk][None, :]), mx.array(self.embed[tk][None, :]))

    def verify_forward(self, block):
        d = np.array(block)[0]
        return BlockOut(mx.array(self.transition[d][None, :, :]), mx.array(self.embed[d][None, :, :]),
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


def test_gemma4_generate_is_lossless() -> None:
    rng = np.random.default_rng(0)
    drafter = resolve_arch(GCFG).build(GCFG, max_seq_len=128)
    transition = rng.standard_normal((VOCAB, VOCAB)).astype(np.float32)
    embed = (rng.standard_normal((VOCAB, FC_IN)) * 0.1).astype(np.float32)
    adapter = MarkovAdapter(transition, embed)
    prompt = rng.integers(0, VOCAB, size=(1, 6)).astype(np.int32)
    events = list(generate(adapter, drafter, prompt, max_new_tokens=20))
    tokens = [e.token for e in events if isinstance(e, TokenEvent)]
    assert isinstance(events[-1], SummaryEvent)
    assert tokens == _plain_greedy(transition, int(prompt[0, -1]), 20)


@pytest.mark.skipif(not os.path.exists(_CKPT), reason="real gemma4 draft checkpoint not present")
def test_gemma4_real_weights_load_and_run() -> None:
    try:
        weights = mx.load(_CKPT)
    except RuntimeError as exc:  # partial/corrupt download
        pytest.skip(f"gemma4 checkpoint not fully downloaded: {exc}")
    config = json.load(open(_RCFG))
    arch = resolve_arch(config)
    assert arch.name == "gemma4"
    drafter = arch.build(config, max_seq_len=64)
    skipped = load_drafter(drafter, weights, key_map=arch.key_map)
    assert skipped == [], f"unmapped checkpoint keys: {skipped}"

    b, prompt_len = 1, 8
    fc_in = drafter.args.fc_in
    main_full = (mx.random.normal((b, prompt_len, fc_in)) * 0.1).astype(mx.bfloat16)
    drafter.forward_spec(mx.array([5], dtype=mx.int32), main_full, start_pos=0)
    main_dec = (mx.random.normal((b, 1, fc_in)) * 0.1).astype(mx.bfloat16)
    ids, logits, conf = drafter.forward_spec(mx.array([5], dtype=mx.int32), main_dec, start_pos=prompt_len - 1)
    mx.eval(ids, logits, conf)
    a = np.array(ids)
    assert a.shape == (b, drafter.block_size + 1)
    assert (a >= 0).all() and (a < drafter.args.vocab_size).all()
    assert np.isfinite(np.array(logits)).all()
