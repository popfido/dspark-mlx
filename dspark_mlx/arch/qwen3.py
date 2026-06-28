# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (deepseek-ai/DeepSpec: dspark/qwen3/modeling.py)

"""Qwen3 DSpark draft backbone (standalone bf16 ``layers.*`` checkpoint).

A stack of stock Qwen3 decoder layers whose attention does the DSpark context/noise K/V
split: every layer cross-attends to ``[projected target context ‖ draft block]``. Unlike
DeepSeek-V4 (windowed MLA ring + sparse_attn), the context is the full accepted sequence
with standard GQA. RoPE is NeoX/half-split (Qwen convention), applied over
``[context_positions ‖ block_positions]`` — block q at the trailing positions, k at all.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Mapping, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..model.norm_rope import RMSNorm


@dataclass
class Qwen3DSparkArgs:
    vocab_size: int = 151936
    hidden_size: int = 2560
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9728
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    target_layer_ids: Tuple[int, ...] = (1, 9, 17, 25, 33)
    num_target_layers: int = 36
    block_size: int = 7
    mask_token_id: int = 151669
    markov_rank: int = 256
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True
    temperature: float = 0.0
    max_position_embeddings: int = 40960

    @property
    def fc_in(self) -> int:
        return self.hidden_size * len(self.target_layer_ids)

    @classmethod
    def from_dict(cls, params: Mapping) -> "Qwen3DSparkArgs":
        d = dict(params)
        rope = d.get("rope_parameters") or {}
        if "rope_theta" in rope and rope["rope_theta"]:
            d["rope_theta"] = rope["rope_theta"]
        names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in names}
        if "target_layer_ids" in kwargs:
            kwargs["target_layer_ids"] = tuple(kwargs["target_layer_ids"])
        return cls(**kwargs)


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x: [b, seq, heads, hd]; cos/sin: [seq, hd] (NeoX half-split convention)."""
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return x * cos + _rotate_half(x) * sin


def rope_tables(position_ids: mx.array, head_dim: int, theta: float) -> Tuple[mx.array, mx.array]:
    """cos/sin [len(position_ids), head_dim] for the given absolute positions."""
    inv_freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
    freqs = position_ids.astype(mx.float32)[:, None] * inv_freq[None, :]   # [seq, hd/2]
    emb = mx.concatenate([freqs, freqs], axis=-1)                          # [seq, hd]
    return mx.cos(emb), mx.sin(emb)


class Qwen3DSparkAttention(nn.Module):
    def __init__(self, args: Qwen3DSparkArgs):
        super().__init__()
        h, nh, nkv, hd = args.hidden_size, args.num_attention_heads, args.num_key_value_heads, args.head_dim
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.scale = hd ** -0.5
        self.q_proj = nn.Linear(h, nh * hd, bias=False)
        self.k_proj = nn.Linear(h, nkv * hd, bias=False)
        self.v_proj = nn.Linear(h, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, h, bias=False)
        self.q_norm = RMSNorm(hd, args.rms_norm_eps)
        self.k_norm = RMSNorm(hd, args.rms_norm_eps)

    def __call__(self, hidden: mx.array, target_ctx: mx.array, cos_full: mx.array, sin_full: mx.array) -> mx.array:
        b, q, _ = hidden.shape
        ctx = target_ctx.shape[1]
        qh = self.q_norm(self.q_proj(hidden).reshape(b, q, self.nh, self.hd))
        k = self.k_norm(
            mx.concatenate([self.k_proj(target_ctx), self.k_proj(hidden)], axis=1).reshape(
                b, ctx + q, self.nkv, self.hd
            )
        )
        v = mx.concatenate([self.v_proj(target_ctx), self.v_proj(hidden)], axis=1).reshape(
            b, ctx + q, self.nkv, self.hd
        )
        qh = _apply_rope(qh, cos_full[-q:], sin_full[-q:])  # block positions only
        k = _apply_rope(k, cos_full, sin_full)              # context ‖ block
        qh = qh.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(qh, k, v, scale=self.scale, mask=None)
        out = out.transpose(0, 2, 1, 3).reshape(b, q, self.nh * self.hd)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    def __init__(self, args: Qwen3DSparkArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DSparkLayer(nn.Module):
    def __init__(self, args: Qwen3DSparkArgs):
        super().__init__()
        self.self_attn = Qwen3DSparkAttention(args)
        self.mlp = Qwen3MLP(args)
        self.input_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, hidden: mx.array, target_ctx: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), target_ctx, cos, sin)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Qwen3Backbone(nn.Module):
    """fc/hidden_norm context projection + Qwen3 layers + final norm (the DSpark backbone)."""

    def __init__(self, args: Qwen3DSparkArgs):
        super().__init__()
        self.args = args
        self.fc = nn.Linear(args.fc_in, args.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.layers = [Qwen3DSparkLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def project_context(self, target_hidden: mx.array) -> mx.array:
        """Concat of target-layer hiddens [b, ctx, fc_in] -> projected context [b, ctx, hidden]."""
        return self.hidden_norm(self.fc(target_hidden))

    def __call__(self, noise_embed: mx.array, target_ctx: mx.array, cos_full: mx.array, sin_full: mx.array) -> mx.array:
        h = noise_embed
        for layer in self.layers:
            h = layer(h, target_ctx, cos_full, sin_full)
        return self.norm(h)
