# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""Host adapter for ``google/gemma-4-12b`` (a multimodal ``gemma4_unified`` model) via mlx-vlm.

Gemma4's text decoder scales embeddings by sqrt(hidden), interleaves sliding/full attention
with per-layer masks, and softcaps logits -- all intricate to reimplement. mlx-vlm's
``LanguageModel.__call__`` already does it correctly and exposes ``capture_layer_ids`` to
return intermediate hidden states, so this adapter reuses that forward and just collects the
target-layer hiddens. KV is mlx-vlm's own prompt cache (RotatingKVCache on sliding layers,
KVCache on full); rollback uses ``.trim`` (safe while the generation stays within the 1024
sliding window, which covers ordinary benchmark lengths).
"""

from __future__ import annotations

import json
from typing import Any, Sequence, Tuple

import mlx.core as mx

from .mlx_lm import MlxLmHostAdapter


class Gemma4UnifiedHostAdapter(MlxLmHostAdapter):
    def __init__(self, model: Any, target_layer_ids: Sequence[int]) -> None:
        from mlx_lm.models.cache import make_prompt_cache

        self.model = model
        self.lm = model.language_model
        self.target_layer_ids = tuple(target_layer_ids)
        self._capture = list(self.target_layer_ids)
        self._make = lambda: make_prompt_cache(self.lm)
        self.reset()

    def reset(self) -> None:
        self._cache = self._make()

    def _run(self, inputs: mx.array) -> Tuple[mx.array, mx.array]:
        out = self.lm(inputs, cache=self._cache, capture_layer_ids=self._capture)
        main = mx.concatenate(out.hidden_states, axis=-1)  # captured in capture_layer_ids (increasing) order
        return out.logits, main


def load_gemma4_unified_adapter(repo: str, config_path: str, precision: str = "bf16"):
    """Load the gemma4_unified base + build (model, tokenizer, adapter)."""
    from mlx_vlm import load

    model, processor = load(repo)
    if precision == "8bit":
        mx.eval(model.parameters())
        nn_quantize(model)
    tokenizer = getattr(processor, "tokenizer", processor)
    config = json.load(open(config_path))
    adapter = Gemma4UnifiedHostAdapter(model, config["target_layer_ids"])
    return model, tokenizer, adapter


def nn_quantize(model) -> None:
    import mlx.nn as nn

    nn.quantize(model, group_size=64, bits=8)
    mx.eval(model.parameters())
