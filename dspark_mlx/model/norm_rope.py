# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""RMSNorm + RoPE for the DSpark draft stack.

The drafter's ``DSparkAttention`` has ``compress_ratio == 0`` (no KV compression), so it
takes the reference's plain-sliding-window RoPE path: base ``rope_theta`` with YaRN
disabled (``original_seq_len == 0``). RoPE here uses the reference's *interleaved* complex
convention (consecutive dims form (real, imag) pairs), which differs from mlx-lm's default
RoPE — hence the explicit port rather than ``nn.RoPE``.
"""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """fp32 RMSNorm (weight stored fp32). ``with_scale=False`` is a weightless variant
    (e.g. Gemma4's v_norm) — pure normalization, no learnable weight."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.with_scale = with_scale
        if with_scale:
            self.weight = mx.ones((dim,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        xf = x.astype(mx.float32)
        var = mx.mean(xf * xf, axis=-1, keepdims=True)
        xf = xf * mx.rsqrt(var + self.eps)
        if self.with_scale:
            xf = self.weight * xf
        return xf.astype(dtype)


def precompute_rope(dim: int, seqlen: int, base: float = 10000.0) -> Tuple[mx.array, mx.array]:
    """No-YaRN RoPE tables. Returns ``(cos, sin)`` each ``[seqlen, dim/2]``."""
    freqs = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    t = mx.arange(seqlen).astype(mx.float32)
    theta = mx.outer(t, freqs)
    return mx.cos(theta), mx.sin(theta)


def apply_rotary_emb(
    x: mx.array, cos: mx.array, sin: mx.array, inverse: bool = False
) -> mx.array:
    """Interleaved-pair RoPE, matching ``inference/model.py::apply_rotary_emb``.

    ``x`` is ``[..., rd]`` with the position axis at index 1 (``[b, s, rd]`` or
    ``[b, s, h, rd]``); ``cos``/``sin`` are ``[s, rd/2]`` for those positions. ``inverse``
    conjugates (de-rotation), used for the output projection.
    """
    rd = x.shape[-1]
    half = rd // 2
    seqlen = x.shape[1]
    xf = x.astype(mx.float32).reshape(*x.shape[:-1], half, 2)
    xr = xf[..., 0]
    xi = xf[..., 1]
    if x.ndim == 3:
        c = cos.reshape(1, seqlen, half)
        s = sin.reshape(1, seqlen, half)
    else:
        c = cos.reshape(1, seqlen, 1, half)
        s = sin.reshape(1, seqlen, 1, half)
    if inverse:
        s = -s
    out_r = xr * c - xi * s
    out_i = xr * s + xi * c
    out = mx.stack([out_r, out_i], axis=-1).reshape(x.shape)
    return out.astype(x.dtype)
