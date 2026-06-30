# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash/Pro-DSpark, deepseek-ai/DeepSpec)

"""Load a DSpark draft + host base model from Hugging Face refs (or local paths).

The published standalone drafts (``deepseek-ai/dspark_{qwen3_*,gemma4_*}_block7``) ship a
``config.json`` + safetensors; the matching base is the *deployed instruct* model. This module
resolves both, builds the drafter via the arch registry, and wires the right host adapter
(mlx-lm for Qwen3, mlx-vlm's ``gemma4_unified`` text tower for Gemma4).

The native ``deepseek-ai/DeepSeek-V4-{Flash,Pro}-DSpark`` checkpoints are different: the DSpark
draft (``mtp.*``) is *bundled inside* the multi-hundred-GB fp8/fp4 base and lives in only a few
safetensors shards, and the reference dims are in ``inference/config.json`` (the root
``config.json`` uses HF ``hidden_size`` naming). So for those we read ``inference/config.json``
and download only the shards holding ``mtp.*`` + the shared ``embed``/``head`` — not the base.
Running such a draft still needs a ``deepseek_v4`` host adapter (not yet implemented; the base is
unrunnable on a single machine), so :func:`load_host` raises for it — draft load is for
inspection / parity today.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Set, Tuple

import mlx.core as mx

from .loading import load_drafter
from .quant import quantize_drafter
from .registry import resolve_arch

#: name -> (draft HF ref, default base HF ref). For Qwen3/Gemma4 the base is the deployed
#: instruct model; for DeepSeek-V4 the draft lives inside the base checkpoint, so both are the
#: same repo (and the base is not locally runnable yet — draft load only).
KNOWN_MODELS: Dict[str, Tuple[str, str]] = {
    "qwen3-4b": ("deepseek-ai/dspark_qwen3_4b_block7", "Qwen/Qwen3-4B"),
    "qwen3-8b": ("deepseek-ai/dspark_qwen3_8b_block7", "Qwen/Qwen3-8B"),
    "qwen3-14b": ("deepseek-ai/dspark_qwen3_14b_block7", "Qwen/Qwen3-14B"),
    "gemma4-12b": ("deepseek-ai/dspark_gemma4_12b_block7", "google/gemma-4-12b-it"),
    "deepseek-v4-flash": ("deepseek-ai/DeepSeek-V4-Flash-DSpark", "deepseek-ai/DeepSeek-V4-Flash-DSpark"),
    "deepseek-v4-pro": ("deepseek-ai/DeepSeek-V4-Pro-DSpark", "deepseek-ai/DeepSeek-V4-Pro-DSpark"),
}

#: shared (non-``mtp.*``) tensors the draft also needs from the bundled DeepSeek checkpoint.
_DRAFT_SHARED: Tuple[str, ...] = ("embed.weight", "head.weight")


def resolve_model(name: str, base: str | None = None) -> Tuple[str, str]:
    """Map a short name (``qwen3-4b``) or a raw draft ref to ``(draft_ref, base_ref)``."""
    if name in KNOWN_MODELS:
        draft_ref, default_base = KNOWN_MODELS[name]
        return draft_ref, base or default_base
    if base is None:
        raise ValueError(f"unknown model {name!r}; pass --base, or use one of {sorted(KNOWN_MODELS)}")
    return name, base


def _snapshot(ref: str, *, weights: bool = True) -> str:
    """Snapshot an HF repo (or return a local dir). ``weights=False`` fetches metadata only
    (configs / index / tokenizer) — used to peek at a DeepSeek checkpoint without the base."""
    if os.path.isdir(ref):
        return ref
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "tokenizer*", "*.model", "*.txt"]
    if weights:
        patterns.append("*.safetensors")
    return snapshot_download(ref, allow_patterns=patterns)


def _is_deepseek_checkpoint(local: str) -> bool:
    """A native DeepSeek-V4 checkpoint carries the reference ``inference/config.json``."""
    return os.path.exists(os.path.join(local, "inference", "config.json"))


def _deepseek_config(local: str) -> Dict:
    """Read the reference ``inference/config.json`` (``dim``/``n_layers`` naming) and inject the
    ``model_type`` it omits (the root ``config.json`` carries ``model_type: deepseek_v4``)."""
    cfg = json.load(open(os.path.join(local, "inference", "config.json")))
    cfg.setdefault("model_type", "deepseek_v4")
    return cfg


def _draft_shards(weight_map: Dict[str, str]) -> Tuple[Set[str], List[str]]:
    """The keys + safetensors shards that hold the draft (``mtp.*`` + shared ``embed``/``head``)."""
    keep = {k for k in weight_map if k.startswith("mtp.") or k in _DRAFT_SHARED}
    shards = sorted({weight_map[k] for k in keep})
    return keep, shards


def _deepseek_draft_weights(ref: str, local: str) -> Dict[str, mx.array]:
    """Load only the draft tensors, fetching just the shards that hold them (not the 100s-of-GB base)."""
    index = json.load(open(os.path.join(local, "model.safetensors.index.json")))
    keep, shards = _draft_shards(index["weight_map"])
    weights: Dict[str, mx.array] = {}
    for shard in shards:
        if os.path.isdir(ref):
            path = os.path.join(ref, shard)
        else:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(ref, shard)
        weights.update({k: v for k, v in mx.load(path).items() if k in keep})
    return weights


def load_draft(ref: str, quant_bits: int = 0, max_seq_len: int = 8192) -> Tuple[Any, Dict]:
    """Build a drafter from an HF ref / local dir. ``quant_bits`` in {0,8,4} quantizes it.

    DeepSeek-V4 checkpoints (``inference/config.json`` present) read the reference config and pull
    only the ``mtp.*`` shards; everything else is a standalone draft repo (full snapshot).
    """
    meta = _snapshot(ref, weights=False)
    if _is_deepseek_checkpoint(meta):
        config = _deepseek_config(meta)
        weights = _deepseek_draft_weights(ref, meta)
    else:
        local = meta if os.path.isdir(ref) else _snapshot(ref)
        config = json.load(open(os.path.join(local, "config.json")))
        weights = {}
        for shard in sorted(glob.glob(os.path.join(local, "*.safetensors"))):
            weights.update(mx.load(shard))

    arch = resolve_arch(config)
    drafter = arch.build(config, max_seq_len=max_seq_len)
    skipped = load_drafter(drafter, weights, key_map=arch.key_map)
    if skipped:
        raise ValueError(f"unmapped draft keys for {ref}: {skipped[:8]} ...")
    if quant_bits:
        quantize_drafter(drafter, bits=quant_bits)
    mx.eval(drafter.parameters())
    return drafter, config


def load_host(base_ref: str, draft_config: Dict, precision: str = "bf16"):
    """Load the base + build a BaseModelAdapter. Returns ``(model, tokenizer, adapter)``."""
    model_type = str(draft_config.get("model_type", ""))

    if model_type.startswith("deepseek_v4"):
        raise NotImplementedError(
            "DeepSeek-V4 host base is not wired yet: there is no MLX host adapter for the "
            "windowed-MLA + DSA base, and the checkpoint is multi-hundred-GB (unrunnable on a "
            "single machine). dspark-mlx loads the DeepSeek-V4 draft (mtp.*) for inspection / "
            "parity; running it needs a deepseek_v4 host that supplies target hidden states at "
            "dspark_target_layer_ids."
        )

    if model_type.startswith("gemma4"):
        from mlx_vlm import load as vlm_load

        from .hosts.gemma4_unified import Gemma4UnifiedHostAdapter, nn_quantize

        model, processor = vlm_load(base_ref)
        if precision == "8bit":
            nn_quantize(model)
        tokenizer = getattr(processor, "tokenizer", processor)
        return model, tokenizer, Gemma4UnifiedHostAdapter(model, tuple(draft_config["target_layer_ids"]))

    from mlx_lm import load as lm_load

    from .hosts.mlx_lm import MlxLmHostAdapter

    model, tokenizer = lm_load(base_ref)
    return model, tokenizer, MlxLmHostAdapter(model, tuple(draft_config["target_layer_ids"]))
