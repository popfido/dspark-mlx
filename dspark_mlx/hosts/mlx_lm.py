# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""A reference :class:`~dspark_mlx.adapter.BaseModelAdapter` over an mlx-lm decoder model.

mlx-lm models only expose the final ``norm(h)`` from their ``__call__``; DSpark needs the
residual-stream output of specific intermediate layers (the same tensors HF returns as
``hidden_states[layer_id + 1]``). So this adapter re-runs the model's own layer loop and
captures ``h`` right after each target layer, concatenated in ``target_layer_ids`` order to
form ``main_hidden``. Logits are recomputed through the model's own ``norm`` + head, so they
are bit-identical to a plain ``model(...)`` forward — emitted tokens equal base greedy
decoding (lossless).

KV is a standard mlx-lm prompt cache (one cache per layer); a partially-accepted block is
rolled back with :meth:`mlx_lm.models.cache.KVCache.trim`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import mlx.core as mx

from ..adapter import BlockOut, StepOut


@dataclass
class HostHooks:
    """The few model internals the adapter needs, decoupled from any one model class."""

    embed: Callable[[mx.array], mx.array]            # token ids [b, L] -> hidden [b, L, H]
    layers: Sequence[Any]                            # decoder blocks, called layer(h, mask, cache)
    norm: Callable[[mx.array], mx.array]             # final norm
    logits: Callable[[mx.array], mx.array]           # normed hidden -> logits [b, L, V]
    make_cache: Callable[[], List[Any]]              # fresh per-layer KV cache
    make_mask: Callable[[mx.array, Any], Any]        # (h, cache[0]) -> attention mask


def build_mlx_lm_hooks(model: Any) -> HostHooks:
    """Hooks for a standard mlx-lm model (``model.model.{embed_tokens,layers,norm}``)."""
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import make_prompt_cache

    inner = model.model
    args = getattr(model, "args", None)
    tied = getattr(args, "tie_word_embeddings", False) or not hasattr(model, "lm_head")
    logits = (lambda out: inner.embed_tokens.as_linear(out)) if tied else (lambda out: model.lm_head(out))
    return HostHooks(
        embed=inner.embed_tokens,
        layers=list(inner.layers),
        norm=inner.norm,
        logits=logits,
        make_cache=lambda: make_prompt_cache(model),
        make_mask=lambda h, c0: create_attention_mask(h, c0),
    )


class MlxLmHostAdapter:
    """BaseModelAdapter backed by an mlx-lm (or hook-compatible) decoder model."""

    def __init__(
        self,
        model: Any,
        target_layer_ids: Sequence[int],
        hooks: Optional[HostHooks] = None,
    ) -> None:
        self.model = model
        self.target_layer_ids = tuple(target_layer_ids)
        self._tset = set(self.target_layer_ids)
        n = len(getattr(model, "model", model).layers) if hooks is None else len(hooks.layers)
        bad = [i for i in self.target_layer_ids if not 0 <= i < n]
        if bad:
            raise ValueError(f"target_layer_ids {bad} out of range for {n}-layer base model")
        self.hooks = hooks or build_mlx_lm_hooks(model)
        self.reset()

    def reset(self) -> None:
        """Drop all KV — start a fresh sequence."""
        self._cache = self.hooks.make_cache()

    @property
    def offset(self) -> int:
        return self._cache[0].offset

    def _run(self, inputs: mx.array) -> Tuple[mx.array, mx.array]:
        """One base forward over ``inputs`` [b, L]; returns (logits [b,L,V], main_hidden [b,L,D])."""
        h = self.hooks.embed(inputs)
        mask = self.hooks.make_mask(h, self._cache[0])
        caps = {}
        for i, (layer, c) in enumerate(zip(self.hooks.layers, self._cache)):
            h = layer(h, mask, c)
            if i in self._tset:
                caps[i] = h
        main = mx.concatenate([caps[i] for i in self.target_layer_ids], axis=-1)
        return self.hooks.logits(self.hooks.norm(h)), main

    def prefill(self, tokens: mx.array) -> StepOut:
        logits, main = self._run(tokens)
        return StepOut(logits=logits[:, -1, :], main_hidden=main)

    def decode_step(self, token: mx.array) -> StepOut:
        logits, main = self._run(token.reshape(1, -1))
        return StepOut(logits=logits[:, -1, :], main_hidden=main[:, -1, :])

    def verify_forward(self, block_tokens: mx.array) -> BlockOut:
        logits, main = self._run(block_tokens)
        return BlockOut(
            per_pos_logits=logits,
            per_pos_main_hidden=main,
            main_hidden_last=main[:, -1, :],
        )

    def kv_snapshot(self) -> Any:
        return self.offset

    def kv_rollback(self, n_keep: int) -> None:
        for c in self._cache:
            drop = c.offset - n_keep
            if drop > 0:
                c.trim(drop)
