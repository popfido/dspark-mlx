# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
"""Sparse gather_mm/gather_qmm MoE dispatch: parity vs the naive dense combine, quantized
accuracy, and the per-expert -> stacked weight-loader path. Torch-free (no reference model)."""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from dspark_mlx.loading import _stack_moe_experts
from dspark_mlx.model.config import DSparkArgs
from dspark_mlx.model.moe import MoE, _swiglu

CFG = dict(
    vocab_size=64, dim=128, moe_inter_dim=64, n_routed_experts=8, n_shared_experts=1,
    n_activated_experts=2, n_hash_layers=0, score_func="sqrtsoftplus", route_scale=1.5,
    swiglu_limit=10.0,
)


def _build_moe(rng, dtype=mx.float32):
    args = DSparkArgs.from_dict(CFG)
    moe = MoE(args, 0)
    E, dim, inter = CFG["n_routed_experts"], CFG["dim"], CFG["moe_inter_dim"]
    moe.w1 = mx.array(rng.standard_normal((E, inter, dim)).astype(np.float32) * 0.1).astype(dtype)
    moe.w3 = mx.array(rng.standard_normal((E, inter, dim)).astype(np.float32) * 0.1).astype(dtype)
    moe.w2 = mx.array(rng.standard_normal((E, dim, inter)).astype(np.float32) * 0.1).astype(dtype)
    moe.gate.weight = mx.array(rng.standard_normal((E, dim)).astype(np.float32) * 0.1)
    moe.gate.bias = mx.array(rng.standard_normal((E,)).astype(np.float32) * 0.1)
    for n, shp in (("w1", (inter, dim)), ("w2", (dim, inter)), ("w3", (inter, dim))):
        getattr(moe.shared_experts, n).weight = mx.array(rng.standard_normal(shp).astype(np.float32) * 0.1).astype(dtype)
    return moe, args


def _dense_combine(x, w1, w2, w3, weights, indices, limit):
    """The original O(num_experts) reference: evaluate every expert, mask by routing weight."""
    y = mx.zeros(x.shape, dtype=mx.float32)
    for e in range(w1.shape[0]):
        h = _swiglu(x @ w1[e].T, x @ w3[e].T, limit)
        oe = h.astype(x.dtype) @ w2[e].T
        w_e = mx.sum(mx.where(indices == e, weights, 0.0), axis=-1, keepdims=True)
        y = y + oe.astype(mx.float32) * w_e
    return y


def test_sparse_matches_dense() -> None:
    rng = np.random.default_rng(0)
    moe, args = _build_moe(rng)
    x = mx.array(rng.standard_normal((6, CFG["dim"])).astype(np.float32) * 0.5)
    ids = mx.array(rng.integers(0, CFG["vocab_size"], size=(6,)).astype(np.int32))

    sparse = moe(x, ids)
    weights, indices = moe.gate(x, ids)
    dense = _dense_combine(x, moe.w1, moe.w2, moe.w3, weights, indices, args.swiglu_limit)
    dense = dense + moe.shared_experts(x).astype(mx.float32)
    mx.eval(sparse, dense)
    assert float(mx.max(mx.abs(sparse - dense))) < 1e-4


def test_quantized_paths_close() -> None:
    rng = np.random.default_rng(1)
    x = mx.array(rng.standard_normal((6, CFG["dim"])).astype(np.float32) * 0.5)
    ids = mx.array(rng.integers(0, CFG["vocab_size"], size=(6,)).astype(np.int32))

    moe_bf, _ = _build_moe(np.random.default_rng(2), dtype=mx.bfloat16)
    ref = moe_bf(x, ids)
    mx.eval(ref)
    scale = float(mx.max(mx.abs(ref))) + 1e-6

    for bits, tol in ((8, 0.05), (4, 0.40)):  # q8 near-lossless; q4 lossier (ok under lossless verify)
        moe_q, _ = _build_moe(np.random.default_rng(2), dtype=mx.bfloat16)
        moe_q.quantize(bits=bits, group_size=64)
        assert moe_q.quant == (bits, 64)
        assert "w1" not in moe_q and "w1_q" in moe_q  # weights replaced by the quantized triple
        out = moe_q(x, ids)
        mx.eval(out)
        rel = float(mx.max(mx.abs(out - ref))) / scale
        assert rel < tol, f"q{bits} rel err {rel:.3f} exceeds {tol}"


def test_loader_stacks_experts() -> None:
    E, dim, inter = 4, 8, 6
    rng = np.random.default_rng(3)
    per_expert = {e: mx.array(rng.standard_normal((inter, dim)).astype(np.float32)) for e in range(E)}
    params = {f"blocks.2.ffn.experts.{e}.w1.weight": per_expert[e] for e in range(E)}
    params["blocks.2.attn.q_proj.weight"] = mx.zeros((4, 4))  # unrelated key is untouched

    _stack_moe_experts(params)

    assert "blocks.2.ffn.w1" in params
    assert all(f"blocks.2.ffn.experts.{e}.w1.weight" not in params for e in range(E))
    assert params["blocks.2.ffn.w1"].shape == (E, inter, dim)
    for e in range(E):  # row e is expert e
        assert float(mx.max(mx.abs(params["blocks.2.ffn.w1"][e] - per_expert[e]))) == 0.0
    assert "blocks.2.attn.q_proj.weight" in params
