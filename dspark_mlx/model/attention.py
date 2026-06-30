# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash/Pro-DSpark, deepseek-ai/DeepSpec)

"""DSparkAttention: the drafter's MLA attention (``inference/model.py::DSparkAttention``).

This is the ``compress_ratio == 0`` MLA variant — low-rank Q (``wq_a -> q_norm -> wq_b``)
with a per-head RMS, a single shared KV head, a sliding-window KV ring buffer, top-k sparse
attention with a denominator sink, and a grouped low-rank output projection. There is no KV
compressor/indexer (those belong to the host base model). The fp8 QAT round-trip on KV is
dropped (bf16/fp32 drafter).

Prefill (``start_pos == 0``) only seeds the window ring buffer from the main hidden and
returns the input untouched; decode (``start_pos > 0``) drafts the block.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..kernels import sparse_attn
from .config import DSparkArgs
from .norm_rope import RMSNorm, apply_rotary_emb, precompute_rope


class DSparkKVCache:
    """Sliding-window ring buffer for the drafter's shared single-head KV.

    A plain object (not ``nn.Module``) so it never lands in a parameter tree. Slot
    ``p % window_size`` holds the KV for absolute position ``p``; attention is order-
    agnostic over the window, so the ring ordering is irrelevant to correctness.
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.window: mx.array | None = None  # [b, win, head_dim]

    def prefill(self, main_kv: mx.array) -> None:
        b, seqlen, d = main_kv.shape
        win = self.window_size
        if seqlen <= win:
            pad = mx.zeros((b, win - seqlen, d), dtype=main_kv.dtype)
            self.window = mx.concatenate([main_kv, pad], axis=1)
        else:
            cutoff = seqlen % win
            last = main_kv[:, seqlen - win :]  # last `win` tokens
            # slot p%win: [last cutoff tokens] then [first win-cutoff tokens]
            self.window = mx.concatenate([last[:, win - cutoff :], last[:, : win - cutoff]], axis=1)

    def update(self, start_pos: int, v: mx.array) -> None:
        win = self.window_size
        sel = mx.arange(win) == (start_pos % win)
        self.window = mx.where(sel[None, :, None], v[:, None, :], self.window)

    def read(self) -> mx.array:
        return self.window


def _rope_last(x: mx.array, cos: mx.array, sin: mx.array, rd: int, inverse: bool = False) -> mx.array:
    """Rotate only the last ``rd`` dims (the rope slice); leave the nope dims untouched."""
    return mx.concatenate([x[..., :-rd], apply_rotary_emb(x[..., -rd:], cos, sin, inverse)], axis=-1)


def _dspark_topk_idxs(window_size: int, bsz: int, block_size: int, start_pos: int) -> mx.array:
    win = mx.arange(min(window_size, start_pos + 1))
    block = window_size + mx.arange(block_size)
    row = mx.concatenate([win, block]).astype(mx.int32)
    return mx.broadcast_to(row[None, None, :], (bsz, block_size, row.shape[0]))


class DSparkAttention(nn.Module):
    def __init__(self, args: DSparkArgs, max_seq_len: int = 8192):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.o_lora_rank = args.o_lora_rank
        self.n_groups = args.o_groups
        self.window_size = args.window_size
        self.eps = args.norm_eps
        self.softmax_scale = args.head_dim ** -0.5

        self.wq_a = nn.Linear(args.dim, args.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(args.q_lora_rank, args.norm_eps)
        self.wq_b = nn.Linear(args.q_lora_rank, args.n_heads * args.head_dim, bias=False)
        self.wkv = nn.Linear(args.dim, args.head_dim, bias=False)
        self.kv_norm = RMSNorm(args.head_dim, args.norm_eps)
        self.wo_a = nn.Linear(
            args.n_heads * args.head_dim // args.o_groups, args.o_groups * args.o_lora_rank, bias=False
        )
        self.wo_b = nn.Linear(args.o_groups * args.o_lora_rank, args.dim, bias=False)
        self.attn_sink = mx.zeros((args.n_heads,), dtype=mx.float32)

        cos, sin = precompute_rope(args.rope_head_dim, max_seq_len, args.rope_theta)
        self._cos, self._sin = cos, sin
        self.cache = DSparkKVCache(args.window_size)

    def __call__(self, x: mx.array, start_pos: int, main_x: mx.array) -> mx.array:
        rd = self.rope_head_dim
        b, seqlen, _ = main_x.shape

        main_kv = self.kv_norm(self.wkv(main_x))
        main_kv = _rope_last(main_kv, self._cos[start_pos : start_pos + seqlen],
                             self._sin[start_pos : start_pos + seqlen], rd)
        if start_pos == 0:
            self.cache.prefill(main_kv)
            return x

        _, block_size, _ = x.shape
        bcos = self._cos[start_pos + seqlen : start_pos + seqlen + block_size]
        bsin = self._sin[start_pos + seqlen : start_pos + seqlen + block_size]

        q = self.wq_b(self.q_norm(self.wq_a(x))).reshape(b, block_size, self.n_heads, self.head_dim)
        q = q * mx.rsqrt(mx.mean(q * q, axis=-1, keepdims=True) + self.eps)
        q = _rope_last(q, bcos, bsin, rd)
        kv = _rope_last(self.kv_norm(self.wkv(x)), bcos, bsin, rd)

        self.cache.update(start_pos, main_kv[:, 0])
        kv_full = mx.concatenate([self.cache.read(), kv], axis=1)
        topk = _dspark_topk_idxs(self.window_size, b, block_size, start_pos)
        o = sparse_attn(q, kv_full, self.attn_sink, topk, self.softmax_scale)
        o = _rope_last(o, bcos, bsin, rd, inverse=True)

        o = o.reshape(b, block_size, self.n_groups, -1)
        wo_a = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, -1)
        # out[..,g,r] = sum_d o[..,g,d] * wo_a[g,r,d]
        o = mx.sum(o[..., None, :] * wo_a[None, None, :, :, :], axis=-1)
        return self.wo_b(o.reshape(b, block_size, self.n_groups * self.o_lora_rank))

    def advance_window(self, main_x: mx.array, position: int) -> None:
        """Append one committed token's main KV to the window without drafting.

        Used by the generate loop to slide the window over tokens accepted in a block
        (forward_spec only adds the anchor). ``main_x`` is the projected main hidden
        ([b, dim]); this mirrors the decode path's window update for a single position.
        """
        rd = self.rope_head_dim
        main_kv = self.kv_norm(self.wkv(main_x))[:, None, :]  # [b, 1, head_dim]
        main_kv = _rope_last(main_kv, self._cos[position : position + 1],
                             self._sin[position : position + 1], rd)
        self.cache.update(position, main_kv[:, 0])
