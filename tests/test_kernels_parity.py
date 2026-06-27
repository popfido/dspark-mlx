# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
#
# Parity of the MLX kernel ports against the CPU (torch) reference, plus independent
# known-answer anchors (dense-equivalence for sparse_attn; column-stochasticity for the
# Sinkhorn output) so a shared bug in port+reference cannot pass silently.
from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch

from cpu_kernels import hc_split_sinkhorn as ref_hc
from cpu_kernels import sparse_attn as ref_sparse_attn
from dspark_mlx.kernels import hc_split_sinkhorn as mlx_hc
from dspark_mlx.kernels import sparse_attn as mlx_sparse_attn


def _maxdiff(a: mx.array, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.array(a) - b)))


def test_sparse_attn_matches_reference() -> None:
    rng = np.random.default_rng(0)
    b, m, h, d, n, t = 2, 5, 4, 16, 40, 12
    q = (rng.standard_normal((b, m, h, d)) * 0.3).astype(np.float32)
    kv = (rng.standard_normal((b, n, d)) * 0.3).astype(np.float32)
    sink = (rng.standard_normal((h,)) * 0.1).astype(np.float32)
    idx = rng.integers(0, n, size=(b, m, t)).astype(np.int32)
    idx[:, :, -2:] = -1  # exercise the masked path
    scale = d ** -0.5

    ref = ref_sparse_attn(
        torch.tensor(q), torch.tensor(kv), torch.tensor(sink), torch.tensor(idx), scale
    ).numpy()
    out = mlx_sparse_attn(mx.array(q), mx.array(kv), mx.array(sink), mx.array(idx), scale)
    mx.eval(out)
    assert _maxdiff(out, ref) <= 1e-3


def test_sparse_attn_equals_dense_when_full_and_no_sink() -> None:
    # topk = every position in order + a hugely-negative sink => plain softmax attention.
    rng = np.random.default_rng(1)
    b, m, h, d, n = 2, 3, 4, 8, 10
    q = (rng.standard_normal((b, m, h, d)) * 0.5).astype(np.float32)
    kv = (rng.standard_normal((b, n, d)) * 0.5).astype(np.float32)
    sink = np.full((h,), -1e4, dtype=np.float32)
    idx = np.broadcast_to(np.arange(n, dtype=np.int32), (b, m, n)).copy()
    scale = d ** -0.5

    scores = np.einsum("bmhd,bnd->bmhn", q, kv) * scale
    scores -= scores.max(-1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(-1, keepdims=True)
    dense = np.einsum("bmhn,bnd->bmhd", w, kv).astype(np.float32)

    out = mlx_sparse_attn(mx.array(q), mx.array(kv), mx.array(sink), mx.array(idx), scale)
    mx.eval(out)
    assert _maxdiff(out, dense) <= 1e-3


def test_hc_sinkhorn_matches_reference() -> None:
    rng = np.random.default_rng(2)
    b, s, hc = 2, 4, 4
    mix_hc = (2 + hc) * hc
    mixes = (rng.standard_normal((b, s, mix_hc)) * 0.5).astype(np.float32)
    hc_scale = (rng.standard_normal((3,)) * 0.5).astype(np.float32)
    hc_base = (rng.standard_normal((mix_hc,)) * 0.5).astype(np.float32)

    r_pre, r_post, r_comb = ref_hc(
        torch.tensor(mixes), torch.tensor(hc_scale), torch.tensor(hc_base), hc, 20, 1e-6
    )
    m_pre, m_post, m_comb = mlx_hc(
        mx.array(mixes), mx.array(hc_scale), mx.array(hc_base), hc, 20, 1e-6
    )
    mx.eval(m_pre, m_post, m_comb)

    assert _maxdiff(m_pre, r_pre.numpy()) <= 1e-4
    assert _maxdiff(m_post, r_post.numpy()) <= 1e-4
    assert _maxdiff(m_comb, r_comb.numpy()) <= 1e-4


def test_hc_sinkhorn_columns_normalized() -> None:
    # The final Sinkhorn op is a column normalization, so each column sums to ~1.
    rng = np.random.default_rng(3)
    hc = 4
    mix_hc = (2 + hc) * hc
    mixes = (rng.standard_normal((6, mix_hc)) * 0.5).astype(np.float32)
    hc_scale = np.ones((3,), dtype=np.float32)
    hc_base = np.zeros((mix_hc,), dtype=np.float32)

    _, _, comb = mlx_hc(mx.array(mixes), mx.array(hc_scale), mx.array(hc_base), hc, 20, 1e-6)
    col_sums = np.array(mx.sum(comb, axis=-2))
    assert np.max(np.abs(col_sums - 1.0)) < 1e-2
