# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""MLX ports of the two reference ``inference/kernel.py`` ops the DSpark drafter needs.

The reference implements these as TileLang/CUDA kernels; here they are plain MLX array
ops (the draft block is tiny — K≈5 query positions — so a dense formulation is ample and
keeps everything on the Metal graph). The fp8/fp4 quant kernels are intentionally not
ported: the drafter runs in bf16/fp32 (see module docstring of :mod:`dspark_mlx.adapter`).
"""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx


def sparse_attn(
    q: mx.array,
    kv: mx.array,
    attn_sink: mx.array,
    topk_idxs: mx.array,
    softmax_scale: float,
) -> mx.array:
    """Top-k gathered single-KV-head attention with a denominator-only sink.

    Args:
        q: ``[b, m, h, d]`` queries (one shared KV head broadcast over ``h`` query heads).
        kv: ``[b, n, d]`` key/value rows.
        attn_sink: ``[h]`` learnable per-head sink logit (value vector is implicitly zero).
        topk_idxs: ``[b, m, t]`` int indices into ``n``; ``-1`` masks the slot.
        softmax_scale: score scale (the reference uses ``head_dim ** -0.5``).

    Returns:
        ``[b, m, h, d]`` attention output.
    """
    b, m, h, d = q.shape
    t = topk_idxs.shape[-1]

    mask = topk_idxs != -1                                  # [b, m, t]
    safe = mx.where(mask, topk_idxs, 0).astype(mx.int32)    # [b, m, t]
    flat = safe.reshape(b, m * t)                           # [b, m*t]
    idx_exp = mx.broadcast_to(flat[:, :, None], (b, m * t, d)).astype(mx.int32)
    kv_g = mx.take_along_axis(kv.astype(mx.float32), idx_exp, axis=1).reshape(b, m, t, d)

    qf = q.astype(mx.float32)
    scores = mx.matmul(qf, mx.swapaxes(kv_g, -1, -2)) * softmax_scale  # [b, m, h, t]
    scores = mx.where(mask[:, :, None, :], scores, -mx.array(float("inf")))

    smax = mx.max(scores, axis=-1, keepdims=True)           # [b, m, h, 1]
    ex = mx.exp(scores - smax)                              # masked -> 0
    num = mx.matmul(ex, kv_g)                               # [b, m, h, d]
    sink = mx.exp(attn_sink[None, None, :, None] - smax)    # [b, m, h, 1]
    denom = mx.sum(ex, axis=-1, keepdims=True) + sink
    return (num / denom).astype(q.dtype)


def hc_split_sinkhorn(
    mixes: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Hyper-Connection pre/post weights + a Sinkhorn-normalized combination matrix.

    ``mixes`` is ``[.., (2+hc)*hc]``; the first ``hc`` entries drive ``pre``, the next
    ``hc`` drive ``post``, and the remaining ``hc*hc`` form ``comb`` (row-major) which is
    softmax-normalized then Sinkhorn-iterated toward doubly-stochastic.
    """
    hc = hc_mult
    pre = mx.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * mx.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])

    comb = mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]
    comb = comb.reshape(*mixes.shape[:-1], hc, hc)
    comb = mx.softmax(comb, axis=-1) + eps
    comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + eps)
        comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    return pre, post, comb
