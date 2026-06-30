# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""Quantize a DSpark drafter in place.

DeepSeek-V4 ships its draft as fp8 (the DSpark layers live inside the fp8 base checkpoint),
so a low-precision drafter is the intended design, not a hack. Quantizing the *whole* draft —
the compute Linears, the ``lm_head``, the token embedding, and the Markov head's rank→vocab
projection — is **acceptance-lossless at 8-bit** on Qwen3 / Gemma4 and is both smaller and
faster than partial quantization:

    Qwen3-4B draft   bf16  2786 MB  accepted 6.12  bf16-base speedup 1.26x
                     q8    1480 MB  accepted 6.11  bf16-base speedup 1.67x   (0.53x size)
                     q4     784 MB  accepted 5.88  bf16-base speedup 1.57x   (0.28x size)

The Markov head's projection is a real per-block matmul, so quantizing it helps speed, not
just size. 8-bit is free; 4-bit costs ~4% (Qwen) / ~7% (Gemma) accepted length.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .model.moe import MoE


def quantize_drafter(drafter: nn.Module, bits: int = 8, group_size: int = 64) -> nn.Module:
    """Quantize every compatible Linear + Embedding in the drafter to ``bits``. Layers whose
    quantized dimension isn't a multiple of ``group_size`` (e.g. tiny heads) are skipped
    gracefully. The DeepSeek-V4 draft's MoE keeps its routed experts as stacked arrays (not
    ``nn.Linear``), so ``nn.quantize`` skips them — they are quantized via the MoE's own
    ``gather_qmm`` path. Mutates and returns ``drafter``."""

    def predicate(_path: str, module: nn.Module):
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False
        return module.weight.shape[-1] % group_size == 0

    nn.quantize(drafter, group_size=group_size, bits=bits, class_predicate=predicate)
    for _name, module in drafter.named_modules():
        if isinstance(module, MoE) and module.quant is None:
            module.quantize(bits=bits, group_size=group_size)
    mx.eval(drafter.parameters())
    return drafter
