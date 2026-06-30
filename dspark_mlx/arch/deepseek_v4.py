# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash/Pro-DSpark, deepseek-ai/DeepSpec)

"""DeepSeek-V4-{Flash,Pro}-DSpark backbone descriptor.

The windowed MLA + hash-MoE + Hyper-Connections realization, drafting from the ``mtp.*``
namespace of the bundled fp8/fp4 checkpoint. The model code lives under ``dspark_mlx.model``
(its parity tests pin it); this module just registers it as a DraftArch.

Both the **Flash** and the larger **Pro** checkpoints share this exact draft architecture —
Pro only scales the dims (``DSparkArgs.from_dict`` absorbs them) and adds DeepSeek Sparse
Attention (the indexer/compressor) to its *base* layers; the draft excludes it (the reference
``DSparkAttention`` asserts ``compress_ratio == 0``), so the same descriptor covers both.
"""

from __future__ import annotations

from typing import Optional

from ..loading import map_checkpoint_key
from ..model.config import DSparkArgs
from ..model.drafter import DSparkDrafter
from .backbone import DraftArch, DraftBackbone


def build(config: dict, *, max_seq_len: int = 8192) -> DraftBackbone:
    return DSparkDrafter(DSparkArgs.from_dict(config), max_seq_len=max_seq_len)


def key_map(key: str) -> Optional[str]:
    return map_checkpoint_key(key)  # mtp.N.* -> blocks.N.*, embed/head pass through


DEEPSEEK_V4 = DraftArch(
    name="deepseek_v4",
    model_types=("deepseek_v4",),
    build=build,
    key_map=key_map,
)
