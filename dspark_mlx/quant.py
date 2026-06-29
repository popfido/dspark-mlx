# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""Quantize a DSpark drafter in place.

DeepSeek-V4 ships its draft as fp8 (the DSpark layers live inside the fp8 base checkpoint),
so a low-precision drafter is the intended design, not a hack. Quantizing the draft's compute
Linears (including the big ``lm_head``) is acceptance-lossless at 8-bit and ~3-4% at 4-bit on
Qwen3 / Gemma4 — and since the draft is a large share of each decode cycle (≈36% bf16 base,
≈46% 8-bit base), it lifts speedup across the board (Qwen3-4B bf16 base 1.21× → 1.38× q8 /
1.52× q4; 8-bit base 1.27× → 1.51× q8). The Markov + confidence heads stay full precision —
they shape the draft distribution directly and are tiny, so quantizing them isn't worth the
acceptance risk.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

_SKIP = ("markov_head", "confidence_head")


def quantize_drafter(drafter: nn.Module, bits: int = 8, group_size: int = 64) -> nn.Module:
    """Quantize the drafter's compute Linears (incl. ``lm_head``) to ``bits``; keep the Markov +
    confidence heads (and embeddings / norms) in full precision. Mutates and returns ``drafter``."""

    def predicate(path: str, module: nn.Module):
        return isinstance(module, nn.Linear) and not any(s in path for s in _SKIP)

    nn.quantize(drafter, group_size=group_size, bits=bits, class_predicate=predicate)
    mx.eval(drafter.parameters())
    return drafter
