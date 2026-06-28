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
from .heads import DSparkConfidenceHead, DSparkMarkovHead
from .hyper import hc_head, hc_post, hc_pre
from .moe import MoE
from .norm_rope import RMSNorm


def sample(logits: mx.array, temperature: float) -> mx.array:
    """Match the reference sampler: argmax at temperature 0, else Gumbel-max."""
    if temperature == 0:
        return mx.argmax(logits, axis=-1).astype(mx.int32)
    probs = mx.softmax(logits.astype(mx.float32) / max(temperature, 1e-5), axis=-1)
    g = mx.random.uniform(shape=probs.shape).astype(mx.float32)
    return mx.argmax(probs / (-mx.log(g + 1e-20)), axis=-1).astype(mx.int32)


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
        self.temperature = args.temperature
        self.markov_rank = args.dspark_markov_rank
        self.is_last = stage_id == args.n_mtp_layers - 1
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

        if self.is_last:
            self.norm = RMSNorm(args.dim, args.norm_eps)
            self.markov_head = DSparkMarkovHead(args.vocab_size, args.dspark_markov_rank)
            self.confidence_head = DSparkConfidenceHead(args.dim + args.dspark_markov_rank)
            self.hc_head_fn = mx.zeros((self.hc_mult, hc_dim), dtype=mx.float32)
            self.hc_head_base = mx.zeros((self.hc_mult,), dtype=mx.float32)
            self.hc_head_scale = mx.zeros((1,), dtype=mx.float32)

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

    def advance(self, main_x: mx.array, position: int) -> None:
        """Slide this block's window over one committed token (no draft compute)."""
        self.attn.advance_window(main_x, position)

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

    def forward_head(self, x: mx.array, input_ids: mx.array, head: nn.Module):
        """Last-stage only: produce draft tokens, biased logits, and confidence.

        ``x`` is the [b, block_size, hc, dim] stack output. Reduces it (hc_head), reads the
        LM head for base logits, then walks the block left-to-right adding the Markov
        transition bias and sampling each next token. Confidence reads the (pre-norm)
        reduced hidden with the per-position Markov embedding.
        """
        x = hc_head(x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm_eps, self.hc_eps)
        logits = head(self.norm(x).astype(mx.float32))  # [b, block_size, vocab]

        prev = input_ids.astype(mx.int32)  # output_ids[:, 0] = anchor
        out_ids, biased_cols, markov_embeds = [prev], [], []
        for i in range(self.block_size):
            bias, membed = self.markov_head(prev)        # [b, vocab], [b, rank]
            li = logits[:, i] + bias
            biased_cols.append(li)
            markov_embeds.append(membed)
            prev = sample(li, self.temperature)
            out_ids.append(prev)
        output_ids = mx.stack(out_ids, axis=1)           # [b, block_size + 1]
        logits_out = mx.stack(biased_cols, axis=1)       # [b, block_size, vocab]
        markov_embed = mx.stack(markov_embeds, axis=1)   # [b, block_size, rank]
        confidence = self.confidence_head(x, markov_embed)  # [b, block_size]
        return output_ids, logits_out, confidence
