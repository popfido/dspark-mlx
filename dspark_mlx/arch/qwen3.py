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
    cos = cos[None, :, None, :].astype(x.dtype)
    sin = sin[None, :, None, :].astype(x.dtype)
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

    # --- cached path (Phase 3b): context K/V are precomputed once and reused every block ---
    # k_norm + RoPE are position-wise, so caching post-RoPE context K/V is exactly equivalent
    # to recomputing k_proj/v_proj over the whole context each block.

    def context_kv(self, proj_ctx: mx.array, cos: mx.array, sin: mx.array):
        """Project committed context [b, n, hidden] -> RoPE'd K, V [b, nkv, n, hd] for the cache."""
        b, n, _ = proj_ctx.shape
        k = self.k_norm(self.k_proj(proj_ctx).reshape(b, n, self.nkv, self.hd))
        v = self.v_proj(proj_ctx).reshape(b, n, self.nkv, self.hd)
        k = _apply_rope(k, cos, sin)
        return k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)

    def attend_cached(self, noise, ctx_k, ctx_v, cos, sin):
        """Attend the noise block over [cached context K/V ‖ this block]."""
        b, q, _ = noise.shape
        qh = self.q_norm(self.q_proj(noise).reshape(b, q, self.nh, self.hd))
        nk = self.k_norm(self.k_proj(noise).reshape(b, q, self.nkv, self.hd))
        nv = self.v_proj(noise).reshape(b, q, self.nkv, self.hd)
        qh = _apply_rope(qh, cos, sin).transpose(0, 2, 1, 3)
        nk = _apply_rope(nk, cos, sin).transpose(0, 2, 1, 3)
        nv = nv.transpose(0, 2, 1, 3)
        k = nk if ctx_k is None else mx.concatenate([ctx_k, nk], axis=2)
        v = nv if ctx_v is None else mx.concatenate([ctx_v, nv], axis=2)
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

    # cached path: context K/V come from the global projected context (no input_layernorm);
    # only the noise block runs through input_layernorm, matching the reference layer.
    def context_kv(self, proj_ctx: mx.array, cos: mx.array, sin: mx.array):
        return self.self_attn.context_kv(proj_ctx, cos, sin)

    def forward_cached(self, hidden, ctx_k, ctx_v, cos, sin) -> mx.array:
        hidden = hidden + self.self_attn.attend_cached(self.input_layernorm(hidden), ctx_k, ctx_v, cos, sin)
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


# --- drafter (DraftBackbone) + checkpoint mapping + registry descriptor ---

import re as _re  # noqa: E402

from ..model.heads import DSparkConfidenceHead, DSparkMarkovHead  # noqa: E402
from ..recipe import draft_block_decode  # noqa: E402
from .backbone import DraftArch  # noqa: E402

_QWEN3_LAYER_RE = _re.compile(r"layers\.(\d+)\.(.+)$")


class Qwen3DSparkDrafter(nn.Module):
    """Qwen3 DSpark drafter conforming to DraftBackbone (forward_spec / advance).

    Holds the shared embed/lm_head, the Qwen3 backbone, and the Markov + confidence heads.
    Context is the full accepted sequence's projected target hidden, accumulated across the
    decode (no windowing); the draft block cross-attends to it. (Append-based growth is O(ctx)
    per step — a preallocated buffer is a perf follow-up.)
    """

    def __init__(self, args: Qwen3DSparkArgs, max_seq_len: int = 8192):
        super().__init__()
        self.args = args
        self.block_size = args.block_size
        self.temperature = args.temperature
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.backbone = Qwen3Backbone(args)
        self.markov_head = DSparkMarkovHead(args.vocab_size, args.markov_rank)
        self.confidence_head = DSparkConfidenceHead(args.hidden_size + args.markov_rank, bias=True)
        self.reset()

    def reset(self) -> None:
        self._ctx = None        # [b, ctx_len, hidden] projected target context (legacy forward_spec path)
        self._next_pos = 0
        self._ctx_k = None      # per-layer cached context K [b, nkv, ctx_len, hd] (eager path)
        self._ctx_v = None
        self._committed = 0     # committed context length (excludes the live anchor)

    def _project_one(self, main_hidden: mx.array) -> mx.array:
        return self.backbone.project_context(main_hidden.reshape(main_hidden.shape[0], 1, -1))

    def forward_spec(self, input_ids: mx.array, main_hidden: mx.array, start_pos: int = 0):
        if start_pos == 0:
            # prefill: ingest 0..L-2; the last prompt token becomes the first anchor.
            full = self.backbone.project_context(main_hidden)
            self._ctx = full[:, :-1]
            self._next_pos = full.shape[1] - 1
            return None

        # decode: append the anchor's projected hidden, then draft the block.
        self._ctx = mx.concatenate([self._ctx, self._project_one(main_hidden)], axis=1)
        self._next_pos = start_pos + 1
        b = input_ids.shape[0]
        anchor = input_ids.astype(mx.int32).reshape(b, 1)
        noise = mx.full((b, self.block_size - 1), self.args.mask_token_id, dtype=mx.int32)
        noise_embed = self.embed_tokens(mx.concatenate([anchor, noise], axis=1))

        full_pos = mx.arange(self._ctx.shape[1] + self.block_size)  # contiguous 0..ctx+block-1
        cos, sin = rope_tables(full_pos, self.args.head_dim, self.args.rope_theta)
        block_hidden = self.backbone(noise_embed, self._ctx, cos, sin)
        logits = self.lm_head(block_hidden.astype(mx.float32))
        return draft_block_decode(
            logits, block_hidden, input_ids, self.markov_head, self.confidence_head,
            self.block_size, self.temperature,
        )

    def advance(self, main_hidden: mx.array, position: int) -> None:
        self._ctx = mx.concatenate([self._ctx, self._project_one(main_hidden)], axis=1)
        self._next_pos = position + 1

    # --- reference-matched eager interface (one base forward per cycle) ---
    #
    # The context holds the raw target hiddens of *committed* positions only; the live anchor
    # is never in the context -- it is the block's query at its own absolute position `start`
    # (DeepSpec: noise block at position_ids[start : start+block_size]). This is the fix for
    # the legacy forward_spec path, which wrongly appended the anchor hidden into the context
    # and shifted the block to start+1.

    def extend_context(self, new_hiddens: mx.array) -> None:
        """Project newly committed positions [b, n, fc_in] once and append to each layer's K/V cache."""
        n = new_hiddens.shape[1]
        if n == 0:
            return
        proj = self.backbone.project_context(new_hiddens)  # global hidden_norm(fc(.)), shared by all layers
        pos = mx.arange(self._committed, self._committed + n)
        cos, sin = rope_tables(pos, self.args.head_dim, self.args.rope_theta)
        layers = self.backbone.layers
        if self._ctx_k is None:
            self._ctx_k = [None] * len(layers)
            self._ctx_v = [None] * len(layers)
        for i, layer in enumerate(layers):
            k, v = layer.context_kv(proj, cos, sin)
            self._ctx_k[i] = k if self._ctx_k[i] is None else mx.concatenate([self._ctx_k[i], k], axis=2)
            self._ctx_v[i] = v if self._ctx_v[i] is None else mx.concatenate([self._ctx_v[i], v], axis=2)
        self._committed += n

    def draft(self, anchor_token: mx.array):
        """Draft a block from the cached context; anchor is the query at position `start`."""
        b = anchor_token.shape[0]
        start = self._committed
        anchor = anchor_token.astype(mx.int32).reshape(b, 1)
        noise = mx.full((b, self.block_size - 1), self.args.mask_token_id, dtype=mx.int32)
        noise_embed = self.embed_tokens(mx.concatenate([anchor, noise], axis=1))
        block_pos = mx.arange(start, start + self.block_size)  # block at its own absolute positions
        cos, sin = rope_tables(block_pos, self.args.head_dim, self.args.rope_theta)
        h = noise_embed
        ck = self._ctx_k or [None] * len(self.backbone.layers)
        cv = self._ctx_v or [None] * len(self.backbone.layers)
        for i, layer in enumerate(self.backbone.layers):
            h = layer.forward_cached(h, ck[i], cv[i], cos, sin)
        block_hidden = self.backbone.norm(h)
        logits = self.lm_head(block_hidden.astype(mx.float32))
        return draft_block_decode(
            logits, block_hidden, anchor_token, self.markov_head, self.confidence_head,
            self.block_size, self.temperature,
        )


def qwen3_key_map(key):
    if key in ("embed_tokens.weight", "lm_head.weight"):
        return key
    if key in ("fc.weight", "hidden_norm.weight", "norm.weight"):
        return f"backbone.{key}"
    if key.startswith("markov_head.") or key.startswith("confidence_head."):
        return key
    m = _QWEN3_LAYER_RE.match(key)
    if m:
        return f"backbone.layers.{m.group(1)}.{m.group(2)}"
    return None


def build(config, *, max_seq_len: int = 8192) -> Qwen3DSparkDrafter:
    return Qwen3DSparkDrafter(Qwen3DSparkArgs.from_dict(config), max_seq_len=max_seq_len)


QWEN3 = DraftArch(name="qwen3", model_types=("qwen3",), build=build, key_map=qwen3_key_map)
