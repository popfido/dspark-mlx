# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash/Pro-DSpark, deepseek-ai/DeepSpec)

"""Architecture-agnostic DSpark recipe pieces shared by every backbone.

The block-head logic (Markov-biased intra-block sampling + confidence) is identical across
DeepSeek-V4 / Qwen3 / Gemma4 — only the way each arch produces the block hidden + base
logits differs. Each backbone computes ``logits`` and a ``conf_hidden`` its own way, then
calls :func:`draft_block_decode`.
"""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


def sample(logits: mx.array, temperature: float) -> mx.array:
    """argmax at temperature 0, else Gumbel-max (matches the reference sampler)."""
    if temperature == 0:
        return mx.argmax(logits, axis=-1).astype(mx.int32)
    probs = mx.softmax(logits.astype(mx.float32) / max(temperature, 1e-5), axis=-1)
    g = mx.random.uniform(shape=probs.shape).astype(mx.float32)
    return mx.argmax(probs / (-mx.log(g + 1e-20)), axis=-1).astype(mx.int32)


def draft_block_decode(
    logits: mx.array,
    conf_hidden: mx.array,
    anchor: mx.array,
    markov_head: nn.Module,
    confidence_head: nn.Module,
    block_size: int,
    temperature: float,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Walk the block left-to-right: add the Markov bias, sample each token, score confidence.

    Args:
        logits: ``[b, block_size, vocab]`` base draft logits (pre-Markov).
        conf_hidden: ``[b, block_size, dim]`` hidden the confidence head reads.
        anchor: ``[b]`` the block's seed token (output position 0).
        markov_head / confidence_head: the DSpark heads.

    Returns ``(output_ids[b, block_size+1], biased_logits[b, block_size, vocab],
    confidence[b, block_size])``.
    """
    prev = anchor.astype(mx.int32)
    out_ids, biased, markov_embeds = [prev], [], []
    for i in range(block_size):
        bias, membed = markov_head(prev)
        li = logits[:, i] + bias
        biased.append(li)
        markov_embeds.append(membed)
        prev = sample(li, temperature)
        out_ids.append(prev)
    output_ids = mx.stack(out_ids, axis=1)
    logits_out = mx.stack(biased, axis=1)
    markov_embed = mx.stack(markov_embeds, axis=1)
    confidence = confidence_head(conf_hidden, markov_embed)
    return output_ids, logits_out, confidence
