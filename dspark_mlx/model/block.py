# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""DSparkBlock: one DSpark draft stage (``inference/model.py::DSparkBlock`` / ``Block``).

Decode runs the Hyper-Connection block: hc_pre -> attn_norm -> DSparkAttention -> hc_post,
then hc_pre -> ffn_norm -> MoE -> hc_post. Prefill (``start_pos == 0``) only seeds the
attention's window KV from the main hidden and returns the input unchanged. ``forward_embed``
(stage 0 only) projects the concatenated main-layer hiddens and embeds the draft block
(anchor token + noise placeholders), expanding to ``hc_mult`` copies.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .attention import DSparkAttention
from .config import DSparkArgs
from .hyper import hc_post, hc_pre
from .moe import MoE
from .norm_rope import RMSNorm


class DSparkBlock(nn.Module):
    def __init__(self, args: DSparkArgs, stage_id: int, max_seq_len: int = 8192):
        super().__init__()
        self.stage_id = stage_id
        self.hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.norm_eps = args.norm_eps
        self.block_size = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        layer_id = args.n_layers + stage_id

        self.attn = DSparkAttention(args, max_seq_len)
        self.ffn = MoE(args, layer_id)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * args.dim
        self.hc_attn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.hc_ffn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.hc_attn_base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.hc_ffn_base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.hc_attn_scale = mx.zeros((3,), dtype=mx.float32)
        self.hc_ffn_scale = mx.zeros((3,), dtype=mx.float32)

        if stage_id == 0:
            self.main_proj = nn.Linear(
                args.dim * len(args.dspark_target_layer_ids), args.dim, bias=False
            )
            self.main_norm = RMSNorm(args.dim, args.norm_eps)

    def _hc_pre(self, x, fn, scale, base):
        return hc_pre(x, fn, scale, base, self.hc_mult, self.hc_sinkhorn_iters, self.norm_eps, self.hc_eps)

    def __call__(self, x: mx.array, start_pos: int, input_ids: mx.array, main_x: mx.array) -> mx.array:
        if start_pos == 0:
            return self.attn(x, start_pos, main_x)  # prefill: seed window KV only

        residual = x
        h, post, comb = self._hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        h = self.attn(self.attn_norm(h), start_pos, main_x)
        x = hc_post(h, residual, post, comb)

        residual = x
        h, post, comb = self._hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        h = self.ffn(self.ffn_norm(h), input_ids)
        return hc_post(h, residual, post, comb)

    def forward_embed(self, main_hidden: mx.array, input_ids: mx.array, embed: nn.Module):
        """Stage-0 only: project main hidden + embed the draft block (anchor + noise)."""
        main_x = self.main_norm(self.main_proj(main_hidden))
        b = input_ids.shape[0]
        anchor = input_ids.reshape(b, 1).astype(mx.int32)
        noise = mx.full((b, self.block_size - 1), self.noise_token_id, dtype=mx.int32)
        draft_ids = mx.concatenate([anchor, noise], axis=1)  # [b, block_size]
        x = embed(draft_ids)  # [b, block_size, dim]
        x = mx.broadcast_to(x[:, :, None, :], (b, self.block_size, self.hc_mult, x.shape[-1]))
        return x, main_x
