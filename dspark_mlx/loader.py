# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""Load a DSpark draft + host base model from Hugging Face refs (or local paths).

The published standalone drafts (``deepseek-ai/dspark_{qwen3_*,gemma4_*}_block7``) ship a
``config.json`` + safetensors; the matching base is the *deployed instruct* model. This module
resolves both, builds the drafter via the arch registry, and wires the right host adapter
(mlx-lm for Qwen3, mlx-vlm's ``gemma4_unified`` text tower for Gemma4).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Tuple

import mlx.core as mx

from .loading import load_drafter
from .quant import quantize_drafter
from .registry import resolve_arch

#: name -> (draft HF ref, default base HF ref). Bases are the deployed instruct models.
KNOWN_MODELS: Dict[str, Tuple[str, str]] = {
    "qwen3-4b": ("deepseek-ai/dspark_qwen3_4b_block7", "Qwen/Qwen3-4B"),
    "qwen3-8b": ("deepseek-ai/dspark_qwen3_8b_block7", "Qwen/Qwen3-8B"),
    "qwen3-14b": ("deepseek-ai/dspark_qwen3_14b_block7", "Qwen/Qwen3-14B"),
    "gemma4-12b": ("deepseek-ai/dspark_gemma4_12b_block7", "google/gemma-4-12b-it"),
}


def resolve_model(name: str, base: str | None = None) -> Tuple[str, str]:
    """Map a short name (``qwen3-4b``) or a raw draft ref to ``(draft_ref, base_ref)``."""
    if name in KNOWN_MODELS:
        draft_ref, default_base = KNOWN_MODELS[name]
        return draft_ref, base or default_base
    if base is None:
        raise ValueError(f"unknown model {name!r}; pass --base, or use one of {sorted(KNOWN_MODELS)}")
    return name, base


def _snapshot(ref: str) -> str:
    if os.path.isdir(ref):
        return ref
    from huggingface_hub import snapshot_download

    return snapshot_download(ref, allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model"])


def load_draft(ref: str, quant_bits: int = 0, max_seq_len: int = 8192) -> Tuple[Any, Dict]:
    """Build a drafter from an HF ref / local dir. ``quant_bits`` in {0,8,4} quantizes it."""
    local = _snapshot(ref)
    config = json.load(open(os.path.join(local, "config.json")))
    arch = resolve_arch(config)
    drafter = arch.build(config, max_seq_len=max_seq_len)
    weights: Dict[str, mx.array] = {}
    for shard in sorted(glob.glob(os.path.join(local, "*.safetensors"))):
        weights.update(mx.load(shard))
    skipped = load_drafter(drafter, weights, key_map=arch.key_map)
    if skipped:
        raise ValueError(f"unmapped draft keys for {ref}: {skipped[:8]} ...")
    if quant_bits:
        quantize_drafter(drafter, bits=quant_bits)
    mx.eval(drafter.parameters())
    return drafter, config


def load_host(base_ref: str, draft_config: Dict, precision: str = "bf16"):
    """Load the base + build a BaseModelAdapter. Returns ``(model, tokenizer, adapter)``."""
    target_layer_ids = tuple(draft_config["target_layer_ids"])
    model_type = str(draft_config.get("model_type", ""))

    if model_type.startswith("gemma4"):
        from mlx_vlm import load as vlm_load

        from .hosts.gemma4_unified import Gemma4UnifiedHostAdapter, nn_quantize

        model, processor = vlm_load(base_ref)
        if precision == "8bit":
            nn_quantize(model)
        tokenizer = getattr(processor, "tokenizer", processor)
        return model, tokenizer, Gemma4UnifiedHostAdapter(model, target_layer_ids)

    from mlx_lm import load as lm_load

    from .hosts.mlx_lm import MlxLmHostAdapter

    model, tokenizer = lm_load(base_ref)
    return model, tokenizer, MlxLmHostAdapter(model, target_layer_ids)
