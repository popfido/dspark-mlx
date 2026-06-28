# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""Architecture registry: pick a DSpark backbone from a base model's config.

Add a new architecture by implementing a ``DraftArch`` (build + key_map) in
``dspark_mlx/arch/<name>.py`` and appending it here — nothing else changes (mirrors
dflash-mlx's ``TARGET_BACKENDS``).
"""

from __future__ import annotations

from typing import Any, List

from .arch.backbone import DraftArch, config_model_type
from .arch.deepseek_v4 import DEEPSEEK_V4
from .arch.gemma4 import GEMMA4
from .arch.qwen3 import QWEN3

ARCH_REGISTRY: List[DraftArch] = [
    DEEPSEEK_V4,
    QWEN3,
    GEMMA4,
]


def resolve_arch(config: Any) -> DraftArch:
    """Return the DraftArch whose ``model_types`` covers ``config.model_type``."""
    model_type = config_model_type(config)
    for arch in ARCH_REGISTRY:
        if arch.supports(model_type):
            return arch
    known = sorted({mt for a in ARCH_REGISTRY for mt in a.model_types})
    raise ValueError(
        f"No DSpark backbone registered for model_type={model_type!r}; known: {known}"
    )
