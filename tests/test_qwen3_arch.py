# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
#
# Parity of the MLX Qwen3 DSpark backbone (context/noise-split attention + layers + fc
# projection) against a torch reference built from real transformers Qwen3 primitives.
from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch
from mlx.utils import tree_flatten, tree_unflatten

from dspark_mlx.arch.qwen3 import Qwen3Backbone, Qwen3DSparkArgs, rope_tables
from qwen3_ref import RefQwen3Backbone

ARGS = Qwen3DSparkArgs(
    hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    intermediate_size=24, num_hidden_layers=2, target_layer_ids=(0, 1),
    rms_norm_eps=1e-6, rope_theta=1_000_000.0, block_size=3,
)


def _torch_cos_sin(positions, head_dim, theta):
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2) / head_dim))
    fr = positions[:, None] * inv[None, :]
    emb = np.concatenate([fr, fr], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def test_qwen3_backbone_parity() -> None:
    rng = np.random.default_rng(0)
    mlx_bb = Qwen3Backbone(ARGS)
    ref_bb = RefQwen3Backbone(ARGS)

    # one weight dict -> both (paths match by construction)
    flat = dict(tree_flatten(mlx_bb.parameters()))
    weights = {p: (rng.standard_normal(tuple(v.shape)) * 0.05).astype(np.float32) for p, v in flat.items()}
    mlx_bb.update(tree_unflatten([(p, mx.array(w)) for p, w in weights.items()]))
    ref_bb.load_state_dict({p: torch.tensor(w) for p, w in weights.items()})

    b, ctx, block = 1, 5, ARGS.block_size
    noise = (rng.standard_normal((b, block, ARGS.hidden_size)) * 0.3).astype(np.float32)
    target_hidden = (rng.standard_normal((b, ctx, ARGS.fc_in)) * 0.3).astype(np.float32)
    positions = np.arange(ctx + block)
    cos_np, sin_np = _torch_cos_sin(positions, ARGS.head_dim, ARGS.rope_theta)

    # rope_tables must reproduce the independent (numpy) formula
    mc, ms = rope_tables(mx.array(positions), ARGS.head_dim, ARGS.rope_theta)
    mx.eval(mc, ms)
    assert np.max(np.abs(np.array(mc) - cos_np)) < 1e-5
    assert np.max(np.abs(np.array(ms) - sin_np)) < 1e-5

    tctx = mlx_bb.project_context(mx.array(target_hidden))
    o_mlx = mlx_bb(mx.array(noise), tctx, mx.array(cos_np), mx.array(sin_np))
    mx.eval(o_mlx)
    with torch.no_grad():
        tctx_r = ref_bb.project_context(torch.tensor(target_hidden))
        o_ref = ref_bb(torch.tensor(noise), tctx_r, torch.tensor(cos_np), torch.tensor(sin_np)).numpy()

    assert np.max(np.abs(np.array(o_mlx) - o_ref)) <= 2e-3
