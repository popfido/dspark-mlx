# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (deepseek-ai/DeepSpec: dspark/gemma4/modeling.py)

"""Gemma4 DSpark draft backbone (standalone bf16 ``layers.*`` checkpoint).

Stock Gemma4 decoder layers with the DSpark context/noise K/V split. Gemma deltas vs Qwen3:
K=V sharing (no v_proj; separate scaled k_norm + weightless v_norm), attention scale 1.0,
partial (proportional) RoPE — only ``partial_rotary_factor`` of head_dim rotates, the rest
pass through — four sandwich norms + a per-layer ``layer_scalar``, GeGLU (gelu-tanh) MLP, and
final-logit softcapping.
"""

from __future__ import annotations

import dataclasses
import re as _re
from dataclasses import dataclass
from typing import Mapping, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..model.heads import DSparkConfidenceHead, DSparkMarkovHead
from ..model.norm_rope import RMSNorm
from ..recipe import draft_block_decode
from .backbone import DraftArch
from .qwen3 import _apply_rope  # shared NeoX rotate_half application

_GEMMA4_LAYER_RE = _re.compile(r"layers\.(\d+)\.(.+)$")


@dataclass
class Gemma4DSparkArgs:
    vocab_size: int = 262144
    hidden_size: int = 3840
    num_hidden_layers: int = 5
    num_attention_heads: int = 16
    num_key_value_heads: int = 1          # global KV head count (k=v)
    head_dim: int = 512                   # global_head_dim
    intermediate_size: int = 15360
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    partial_rotary_factor: float = 0.25
    attention_k_eq_v: bool = True
    final_logit_softcapping: float = 30.0
    target_layer_ids: Tuple[int, ...] = (5, 17, 29, 41, 46)
    num_target_layers: int = 48
    block_size: int = 7
    mask_token_id: int = 4
    markov_rank: int = 256
    temperature: float = 0.0
    max_position_embeddings: int = 262144

    @property
    def fc_in(self) -> int:
        return self.hidden_size * len(self.target_layer_ids)

    @classmethod
    def from_dict(cls, params: Mapping) -> "Gemma4DSparkArgs":
        d = dict(params)
        rope = (d.get("rope_parameters") or {}).get("full_attention") or {}
        if rope.get("rope_theta"):
            d["rope_theta"] = rope["rope_theta"]
        if "partial_rotary_factor" in rope:
            d["partial_rotary_factor"] = rope["partial_rotary_factor"]
        if d.get("global_head_dim"):
            d["head_dim"] = d["global_head_dim"]
        if d.get("num_global_key_value_heads") is not None:
            d["num_key_value_heads"] = d["num_global_key_value_heads"]
        names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in names}
        if "target_layer_ids" in kwargs:
            kwargs["target_layer_ids"] = tuple(kwargs["target_layer_ids"])
        return cls(**kwargs)


def rope_tables(position_ids: mx.array, head_dim: int, theta: float, partial: float) -> Tuple[mx.array, mx.array]:
    """Proportional (partial) RoPE: first ``partial*head_dim`` dims rotate, rest are identity."""
    rope_angles = int(partial * head_dim // 2)
    inv_rot = 1.0 / (theta ** (mx.arange(0, 2 * rope_angles, 2).astype(mx.float32) / head_dim))
    nope = head_dim // 2 - rope_angles
    inv_freq = mx.concatenate([inv_rot, mx.zeros((nope,), dtype=mx.float32)]) if nope > 0 else inv_rot
    freqs = position_ids.astype(mx.float32)[:, None] * inv_freq[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb), mx.sin(emb)


class Gemma4DSparkAttention(nn.Module):
    def __init__(self, args: Gemma4DSparkArgs):
        super().__init__()
        h, nh, nkv, hd = args.hidden_size, args.num_attention_heads, args.num_key_value_heads, args.head_dim
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.k_eq_v = args.attention_k_eq_v
        self.q_proj = nn.Linear(h, nh * hd, bias=False)
        self.k_proj = nn.Linear(h, nkv * hd, bias=False)
        self.v_proj = None if self.k_eq_v else nn.Linear(h, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, h, bias=False)
        self.q_norm = RMSNorm(hd, args.rms_norm_eps)
        self.k_norm = RMSNorm(hd, args.rms_norm_eps)
        self.v_norm = RMSNorm(hd, args.rms_norm_eps, with_scale=False)

    def __call__(self, hidden: mx.array, target_ctx: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        b, q, _ = hidden.shape
        ctx = target_ctx.shape[1]
        qh = self.q_norm(self.q_proj(hidden).reshape(b, q, self.nh, self.hd))
        k_ctx, k_noise = self.k_proj(target_ctx), self.k_proj(hidden)
        v_ctx, v_noise = (k_ctx, k_noise) if self.k_eq_v else (self.v_proj(target_ctx), self.v_proj(hidden))
        k = self.k_norm(mx.concatenate([k_ctx, k_noise], axis=1).reshape(b, ctx + q, self.nkv, self.hd))
        v = self.v_norm(mx.concatenate([v_ctx, v_noise], axis=1).reshape(b, ctx + q, self.nkv, self.hd))
        qh = _apply_rope(qh, cos[-q:], sin[-q:])
        k = _apply_rope(k, cos, sin)  # v is not rotated
        qh = qh.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(qh, k, v, scale=1.0, mask=None)  # Gemma4 scale==1
        out = out.transpose(0, 2, 1, 3).reshape(b, q, self.nh * self.hd)
        return self.o_proj(out)


class Gemma4MLP(nn.Module):
    def __init__(self, args: Gemma4DSparkArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class Gemma4DSparkLayer(nn.Module):
    def __init__(self, args: Gemma4DSparkArgs):
        super().__init__()
        h, eps = args.hidden_size, args.rms_norm_eps
        self.self_attn = Gemma4DSparkAttention(args)
        self.mlp = Gemma4MLP(args)
        self.input_layernorm = RMSNorm(h, eps)
        self.post_attention_layernorm = RMSNorm(h, eps)
        self.pre_feedforward_layernorm = RMSNorm(h, eps)
        self.post_feedforward_layernorm = RMSNorm(h, eps)
        self.layer_scalar = mx.ones((1,), dtype=mx.float32)

    def __call__(self, hidden: mx.array, target_ctx: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        h = self.post_attention_layernorm(self.self_attn(self.input_layernorm(hidden), target_ctx, cos, sin))
        hidden = hidden + h
        h = self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(hidden)))
        hidden = hidden + h
        return hidden * self.layer_scalar


class Gemma4Backbone(nn.Module):
    def __init__(self, args: Gemma4DSparkArgs):
        super().__init__()
        self.fc = nn.Linear(args.fc_in, args.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.layers = [Gemma4DSparkLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def project_context(self, target_hidden: mx.array) -> mx.array:
        return self.hidden_norm(self.fc(target_hidden))

    def __call__(self, noise_embed: mx.array, target_ctx: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        h = noise_embed
        for layer in self.layers:
            h = layer(h, target_ctx, cos, sin)
        return self.norm(h)


class Gemma4DSparkDrafter(nn.Module):
    """Gemma4 DSpark drafter (DraftBackbone). Same loop as Qwen3 + partial RoPE + softcap."""

    def __init__(self, args: Gemma4DSparkArgs, max_seq_len: int = 8192):
        super().__init__()
        self.args = args
        self.block_size = args.block_size
        self.temperature = args.temperature
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.backbone = Gemma4Backbone(args)
        self.markov_head = DSparkMarkovHead(args.vocab_size, args.markov_rank)
        self.confidence_head = DSparkConfidenceHead(args.hidden_size + args.markov_rank, bias=True)
        self.reset()

    def reset(self) -> None:
        self._ctx = None
        self._next_pos = 0

    def _project_one(self, main_hidden: mx.array) -> mx.array:
        return self.backbone.project_context(main_hidden.reshape(main_hidden.shape[0], 1, -1))

    def _rope(self, length: int):
        return rope_tables(mx.arange(length), self.args.head_dim, self.args.rope_theta, self.args.partial_rotary_factor)

    def forward_spec(self, input_ids: mx.array, main_hidden: mx.array, start_pos: int = 0):
        if start_pos == 0:
            full = self.backbone.project_context(main_hidden)
            self._ctx = full[:, :-1]
            self._next_pos = full.shape[1] - 1
            return None
        self._ctx = mx.concatenate([self._ctx, self._project_one(main_hidden)], axis=1)
        self._next_pos = start_pos + 1
        b = input_ids.shape[0]
        anchor = input_ids.astype(mx.int32).reshape(b, 1)
        noise = mx.full((b, self.block_size - 1), self.args.mask_token_id, dtype=mx.int32)
        noise_embed = self.embed_tokens(mx.concatenate([anchor, noise], axis=1))
        cos, sin = self._rope(self._ctx.shape[1] + self.block_size)
        block_hidden = self.backbone(noise_embed, self._ctx, cos, sin)
        logits = self.lm_head(block_hidden.astype(mx.float32))
        sc = self.args.final_logit_softcapping
        if sc:
            logits = mx.tanh(logits / sc) * sc
        return draft_block_decode(
            logits, block_hidden, input_ids, self.markov_head, self.confidence_head,
            self.block_size, self.temperature,
        )

    def advance(self, main_hidden: mx.array, position: int) -> None:
        self._ctx = mx.concatenate([self._ctx, self._project_one(main_hidden)], axis=1)
        self._next_pos = position + 1


def gemma4_key_map(key):
    if key in ("embed_tokens.weight", "lm_head.weight"):
        return key
    if key in ("fc.weight", "hidden_norm.weight", "norm.weight"):
        return f"backbone.{key}"
    if key.startswith("markov_head.") or key.startswith("confidence_head."):
        return key
    m = _GEMMA4_LAYER_RE.match(key)
    if m:
        return f"backbone.layers.{m.group(1)}.{m.group(2)}"
    return None


def build(config, *, max_seq_len: int = 8192) -> Gemma4DSparkDrafter:
    return Gemma4DSparkDrafter(Gemma4DSparkArgs.from_dict(config), max_seq_len=max_seq_len)


GEMMA4 = DraftArch(name="gemma4", model_types=("gemma4", "gemma4_text"), build=build, key_map=gemma4_key_map)
