# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""Load a DSpark drafter from a DeepSeek-V4-Flash-DSpark checkpoint.

The draft stack ships under the ``mtp.*`` namespace (DSpark extends DeepSeek native MTP),
alongside the shared ``embed.weight`` / ``head.weight``. Each quantized weight has a sibling
``.scale`` (block-wise e8m0): attention/projection weights are fp8(e4m3) with a 2D 128x128
scale; MoE expert weights are fp4(e2m1) with a 1D per-32 scale along K. Norms / sinks /
hyper-connection params are unquantized.

Loading flow (the drafter runs in bf16/fp32, so we dequantize):
  host reads safetensors + casts fp8/fp4 -> fp32  ->  ``dequant_*`` apply the block scale
  ->  ``load_drafter`` maps ``mtp.N.* -> blocks.N.*`` and assigns.

The safetensors read + fp8/fp4 cast is left to the host (e.g. omlx already dequantizes the
DeepSeek-V4 base model); this module owns the DSpark-specific mapping + dequant math.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten

from .model.drafter import DSparkDrafter

_MTP_RE = re.compile(r"mtp\.(\d+)\.(.+)$")
_BLOCKS_RE = re.compile(r"blocks\.(\d+)\.(.+)$")

# Keys that distinguish a DSpark checkpoint from a vanilla DeepSeek native-MTP checkpoint
# (both write mtp.*, but only DSpark adds these).
_DSPARK_MARKERS = (".markov_head.", ".confidence_head.", ".main_proj.")


def is_dspark_checkpoint(keys: Iterable[str], config: Optional[dict] = None) -> bool:
    """True if the checkpoint carries the DSpark drafter (vs plain MTP / no drafter)."""
    if config and config.get("dspark_block_size", 0):
        return True
    return any(any(mark in k for mark in _DSPARK_MARKERS) for k in keys)


def map_checkpoint_key(key: str) -> Optional[str]:
    """Checkpoint key -> DSparkDrafter param path; None for base-model keys to skip."""
    if key in ("embed.weight", "head.weight"):
        return key
    m = _MTP_RE.match(key)
    return f"blocks.{m.group(1)}.{m.group(2)}" if m else None


def drafter_path_to_checkpoint_key(path: str) -> str:
    """Inverse of map_checkpoint_key for a drafter param path (blocks.N.* -> mtp.N.*)."""
    m = _BLOCKS_RE.match(path)
    return f"mtp.{m.group(1)}.{m.group(2)}" if m else path


def load_drafter(
    drafter, weights: Dict[str, mx.array], key_map=map_checkpoint_key
) -> List[str]:
    """Assign already-dequantized weights (keyed by checkpoint key) into ``drafter``.

    ``key_map`` translates each checkpoint key to a drafter param path (per-architecture:
    ``mtp.*`` for DeepSeek-V4, ``layers.*`` for Qwen3/Gemma4). ``.scale`` entries are skipped
    (consumed at dequant time); base-model keys are skipped and returned. Rope tables are
    computed (absent from checkpoints) and left untouched.
    """
    params: Dict[str, mx.array] = {}
    skipped: List[str] = []
    for key, value in weights.items():
        if key.endswith(".scale"):
            continue
        path = key_map(key)
        if path is None:
            skipped.append(key)
            continue
        params[path] = value if isinstance(value, mx.array) else mx.array(value)
    drafter.update(tree_unflatten(list(params.items())))
    return skipped


def dequant_fp8_blockwise(weight: np.ndarray, scale: np.ndarray, block: int = 128) -> np.ndarray:
    """fp8 weight (already cast to f32) with a 2D [ceil(out/block) x ceil(in/block)] e8m0 scale."""
    out, inn = weight.shape
    full = np.repeat(np.repeat(scale, block, axis=0), block, axis=1)
    return (weight * full[:out, :inn]).astype(np.float32)


def dequant_fp4_groupwise(weight: np.ndarray, scale: np.ndarray, group: int = 32) -> np.ndarray:
    """fp4 weight (already unpacked + f32-cast) with a 1D [out x ceil(in/group)] e8m0 scale."""
    out, inn = weight.shape
    full = np.repeat(scale, group, axis=1)
    return (weight * full[:, :inn]).astype(np.float32)
