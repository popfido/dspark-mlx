# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash/Pro-DSpark, deepseek-ai/DeepSpec)

"""The seam between dspark-mlx (drafter + verify/accept loop) and a host base model.

dspark-mlx is target-agnostic: it owns the DSpark draft stack and the lossless
accept policy, but never the base model. The host (e.g. omlx over its
``patches/deepseek_v4`` model) implements :class:`BaseModelAdapter` so the drafter
can (a) read the ``main_hidden`` it conditions on, (b) get the base distribution for
each candidate token during verify, and (c) snapshot/roll back base KV when a block is
only partially accepted.

Logit conventions (one decode cycle):
- ``prefill`` / ``decode_step`` return ``StepOut.logits`` = ``p_1``, the base
  distribution for the *first* drafted token. It is free — already computed by the
  step that produced the anchor — so the verify forward never recomputes it.
- ``verify_forward`` runs ONE base forward over the K drafted tokens and returns the
  K base distributions ``p_2 .. p_{K+1}`` (``p_{K+1}`` is the bonus position).
- The generate loop concatenates ``[p_1] + [p_2..p_{K+1}]`` into the ``[K+1, V]`` block
  the accept policy consumes (see :mod:`dspark_mlx.verify`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Tuple, runtime_checkable

import mlx.core as mx


@dataclass
class StepOut:
    """Output of a single base forward at the anchor position."""

    logits: mx.array        # [b, V] base distribution for the next (first drafted) token
    main_hidden: mx.array   # [b, dim * len(target_layer_ids)] concat of target-layer hiddens


@dataclass
class BlockOut:
    """Output of the base verify forward over a K-token draft block."""

    per_pos_logits: mx.array       # [b, K, V] base distributions p_2 .. p_{K+1}
    per_pos_main_hidden: mx.array  # [b, K, D] main hidden at each verified position
    main_hidden_last: mx.array     # [b, D] convenience alias for the last verified position


@runtime_checkable
class BaseModelAdapter(Protocol):
    """Host contract. Implementations own the base model and its KV cache."""

    #: Main-model layer indices whose hidden states are concatenated into ``main_hidden``.
    target_layer_ids: Tuple[int, ...]

    def prefill(self, tokens: mx.array) -> StepOut:
        """Process the prompt; return logits for the first generated token + main_hidden."""
        ...

    def decode_step(self, token: mx.array) -> StepOut:
        """Advance one token; return its next-token logits + main_hidden."""
        ...

    def verify_forward(self, block_tokens: mx.array) -> BlockOut:
        """Run one base forward over K draft tokens; return p_2..p_{K+1} + main_hidden_last.

        Appends K entries to the base KV cache speculatively; the caller rolls back the
        rejected tail via :meth:`kv_rollback`.
        """
        ...

    def kv_snapshot(self) -> Any:
        """Opaque handle capturing base KV state before a speculative block."""
        ...

    def kv_rollback(self, n_keep: int) -> None:
        """Drop speculatively-appended KV beyond ``n_keep`` accepted tokens."""
        ...
