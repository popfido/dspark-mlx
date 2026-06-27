# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""Lossless verify/accept policy for DSpark block drafting.

Speculative decoding accepts the drafted block against the base model's own
distribution so the emitted stream is indistinguishable from non-speculative decoding:

- :func:`greedy_accept` (temperature 0): accept the longest prefix of draft tokens that
  matches the base argmax; the next emitted token is the base argmax at the first
  divergence (a *correction* if a draft token was rejected, or the *bonus* if the whole
  block was accepted). Output is bit-identical to plain greedy decoding.
- :func:`speculative_sample_accept` (temperature > 0): the Leviathan/Chen rejection rule —
  accept ``d_i`` with probability ``min(1, p_i(d_i)/q_i(d_i))``; on rejection resample the
  correction from ``norm(relu(p_i − q_i))``. The emitted distribution equals the base.

The DSpark confidence head is *advisory only* here — it does not influence acceptance in
either mode. It gates adaptive draft length in the (later) lossy mode.

These run once per block on the host (acceptance inherently needs the token ids), so they
operate on small NumPy arrays; the generate loop converts the adapter's MLX logits at the
block boundary. ``base_logits`` carries ``K+1`` rows: ``p_1`` (free, from the anchor step)
followed by ``p_2..p_{K+1}`` (the verify forward).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AcceptResult:
    tokens: np.ndarray  # emitted tokens: accepted prefix + 1 correction/bonus, shape [n_accepted+1]
    n_accepted: int     # number of draft tokens accepted, in [0, K]
    n_emitted: int      # always n_accepted + 1


def greedy_accept(draft_tokens: np.ndarray, base_logits: np.ndarray) -> AcceptResult:
    """Lossless greedy acceptance.

    Args:
        draft_tokens: ``[K]`` int — the drafted tokens ``d_1..d_K``.
        base_logits: ``[K+1, V]`` — base distributions ``p_1..p_{K+1}`` (logits; argmax only).

    Returns:
        :class:`AcceptResult` whose ``tokens`` equal what plain greedy decoding would emit.
    """
    draft_tokens = np.asarray(draft_tokens)
    k = int(draft_tokens.shape[0])
    if base_logits.shape[0] != k + 1:
        raise ValueError(f"base_logits must have K+1={k + 1} rows, got {base_logits.shape[0]}")
    base_argmax = np.asarray(base_logits).argmax(axis=-1)  # [K+1]

    n = 0
    while n < k and int(draft_tokens[n]) == int(base_argmax[n]):
        n += 1
    next_token = int(base_argmax[n])  # correction (n<K) or bonus (n==K); valid since argmax has K+1 rows
    tokens = np.concatenate([draft_tokens[:n].astype(np.int64), np.array([next_token], dtype=np.int64)])
    return AcceptResult(tokens=tokens, n_accepted=n, n_emitted=n + 1)


def speculative_sample_accept(
    draft_tokens: np.ndarray,
    draft_probs: np.ndarray,
    base_probs: np.ndarray,
    rng: np.random.Generator,
) -> AcceptResult:
    """Lossless speculative sampling (temperature > 0).

    Args:
        draft_tokens: ``[K]`` int — drafted tokens, each previously sampled from its ``q_i``.
        draft_probs: ``[K, V]`` — drafter distributions ``q_1..q_K`` (probabilities, rows sum to 1).
        base_probs: ``[K+1, V]`` — base distributions ``p_1..p_{K+1}`` (probabilities).
        rng: NumPy generator for the accept coin and resampling.

    Returns:
        :class:`AcceptResult` whose emitted token is distributed exactly as the base.
    """
    draft_tokens = np.asarray(draft_tokens)
    k = int(draft_tokens.shape[0])
    if base_probs.shape[0] != k + 1:
        raise ValueError(f"base_probs must have K+1={k + 1} rows, got {base_probs.shape[0]}")

    accepted: list[int] = []
    for i in range(k):
        di = int(draft_tokens[i])
        p_i = base_probs[i]
        q_i = draft_probs[i]
        qd = float(q_i[di])
        ratio = 1.0 if qd <= 0.0 else min(1.0, float(p_i[di]) / qd)
        if rng.random() < ratio:
            accepted.append(di)
            continue
        corr = _resample_residual(p_i, q_i, rng)
        return _result(accepted, corr)

    bonus = int(rng.choice(base_probs.shape[-1], p=_normalize(base_probs[k])))
    return _result(accepted, bonus)


def _resample_residual(p_i: np.ndarray, q_i: np.ndarray, rng: np.random.Generator) -> int:
    resid = np.maximum(p_i - q_i, 0.0)
    total = float(resid.sum())
    dist = resid / total if total > 0.0 else _normalize(p_i)
    return int(rng.choice(dist.shape[-1], p=dist))


def _normalize(dist: np.ndarray) -> np.ndarray:
    total = float(dist.sum())
    return dist / total if total > 0.0 else np.full_like(dist, 1.0 / dist.shape[-1])


def _result(accepted: list[int], last: int) -> AcceptResult:
    tokens = np.array([*accepted, last], dtype=np.int64)
    return AcceptResult(tokens=tokens, n_accepted=len(accepted), n_emitted=len(accepted) + 1)
