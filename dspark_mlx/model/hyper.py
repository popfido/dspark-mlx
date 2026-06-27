# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""Hyper-Connection mixing for the DSpark blocks (``inference/model.py`` Block.hc_*).

Instead of a plain residual, each block keeps ``hc_mult`` copies of the hidden state.
``hc_pre`` reduces the copies to one (Sinkhorn-weighted) before a sublayer; ``hc_post``
re-expands the sublayer output back to ``hc_mult`` copies and mixes them with the residual
via the doubly-stochastic ``comb`` matrix; ``hc_head`` is the final copies->one reduction
before the LM head. All three normalize the mix-projection by the input RMS (``norm_eps``)
and use ``hc_eps`` inside the Sinkhorn / sigmoid offsets.
"""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx

from ..kernels import hc_split_sinkhorn


def hc_pre(
    x: mx.array,
    hc_fn: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
) -> Tuple[mx.array, mx.array, mx.array]:
    """[b,s,hc,d] -> reduced [b,s,d] plus (post, comb) for the matching hc_post."""
    shape = x.shape
    xf = x.reshape(*x.shape[:2], -1).astype(mx.float32)          # [b, s, hc*d]
    rsqrt = mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + norm_eps)
    mixes = mx.matmul(xf, hc_fn.T) * rsqrt                       # [b, s, mix_hc]
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps)
    y = mx.sum(pre[..., None] * xf.reshape(shape), axis=2)       # [b, s, d]
    return y.astype(x.dtype), post, comb


def hc_post(x: mx.array, residual: mx.array, post: mx.array, comb: mx.array) -> mx.array:
    """Re-expand sublayer output ``x`` [b,s,d] to [b,s,hc,d], mixing residual via ``comb``."""
    term1 = post[..., None] * x[..., None, :]                   # [b, s, hc, d]
    # out[..,c,d] = sum_a comb[..,a,c] * residual[..,a,d]
    term2 = mx.sum(comb[..., :, :, None] * residual[..., :, None, :], axis=-3)
    return (term1 + term2).astype(x.dtype)


def hc_head(
    x: mx.array,
    hc_fn: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    norm_eps: float,
    hc_eps: float,
) -> mx.array:
    """Final copies->one reduction before the LM head. [b,s,hc,d] -> [b,s,d]."""
    shape = x.shape
    xf = x.reshape(*x.shape[:2], -1).astype(mx.float32)          # [b, s, hc*d]
    rsqrt = mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + norm_eps)
    mixes = mx.matmul(xf, hc_fn.T) * rsqrt                       # [b, s, hc]
    pre = mx.sigmoid(mixes * hc_scale + hc_base) + hc_eps        # [b, s, hc]
    y = mx.sum(pre[..., None] * xf.reshape(shape), axis=2)       # [b, s, d]
    return y.astype(x.dtype)
