# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""The per-architecture seam for DSpark drafters.

DSpark ships one recipe (EAGLE-style context projection + Markov bias + confidence head +
block drafting) realized over different base-model decoder layers — DeepSeek-V4 (windowed
MLA + MoE + Hyper-Connections, bundled fp8/fp4 ``mtp.*`` checkpoint), Qwen3 and Gemma4
(standalone bf16 ``layers.*`` checkpoints, full-context GQA). ``generate()`` drives any of
them through the ``DraftBackbone`` interface; a ``DraftArch`` descriptor registers how to
build and load each one (see :mod:`dspark_mlx.registry`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Tuple, runtime_checkable

import mlx.core as mx


@runtime_checkable
class DraftBackbone(Protocol):
    """A loaded DSpark drafter for one base architecture (what ``generate`` consumes)."""

    block_size: int

    def forward_spec(
        self, input_ids: mx.array, main_hidden: mx.array, start_pos: int = 0
    ) -> Optional[Tuple[mx.array, mx.array, mx.array]]:
        """Prefill (start_pos==0) seeds context; decode drafts (ids, logits, confidence)."""
        ...

    def advance(self, main_hidden: mx.array, position: int) -> None:
        """Slide the drafter's context over one committed token."""
        ...


@dataclass(frozen=True)
class DraftArch:
    """Registry entry: how to build + load a DSpark drafter for a base architecture."""

    name: str
    model_types: Tuple[str, ...]
    build: Callable[..., DraftBackbone]  # (config: dict, *, max_seq_len) -> DraftBackbone
    key_map: Callable[[str], Optional[str]]  # checkpoint key -> drafter param path (or None)

    def supports(self, model_type: Optional[str]) -> bool:
        return model_type in self.model_types


def config_model_type(config: Any) -> Optional[str]:
    """Read ``model_type`` from a dict-like or attribute-like config."""
    if isinstance(config, dict):
        return config.get("model_type")
    return getattr(config, "model_type", None)
