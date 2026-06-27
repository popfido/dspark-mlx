# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
#
# The lossless guarantees of the DSpark verify/accept policy, on toy bases:
#   - greedy: speculative output is BIT-IDENTICAL to plain greedy, for ANY drafter.
#   - sampling: the emitted-token marginal equals the base distribution.
# Plus a structural check that a host adapter satisfies the BaseModelAdapter contract.
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from dspark_mlx.adapter import BaseModelAdapter, BlockOut, StepOut
from dspark_mlx.verify import greedy_accept, speculative_sample_accept


# ---- toy order-1 Markov base/drafter: logits for next token depend only on last token ----

def _markov(vocab: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((vocab, vocab)).astype(np.float64)


def _plain_greedy(M: np.ndarray, start: int, steps: int) -> list[int]:
    seq = [start]
    t = start
    for _ in range(steps):
        t = int(M[t].argmax())
        seq.append(t)
    return seq


def _spec_greedy(M: np.ndarray, G: np.ndarray, start: int, steps: int, k: int) -> list[int]:
    seq = [start]
    t = start
    while len(seq) - 1 < steps:
        # draft k tokens greedily from the (imperfect) drafter G
        d, dt = [], t
        for _ in range(k):
            dt = int(G[dt].argmax())
            d.append(dt)
        d = np.array(d)
        # order-1 base distributions: p_i = M[context before position i]
        ctx = [t, *d.tolist()]                       # [t, d1, .., dK]
        base_logits = np.stack([M[c] for c in ctx])  # [K+1, V] = p_1 .. p_{K+1}
        res = greedy_accept(d, base_logits)
        for tok in res.tokens.tolist():
            seq.append(int(tok))
        t = seq[-1]
    return seq[: steps + 1]


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("seed_pair", [(0, 0), (0, 7), (1, 99), (3, 3)])
def test_greedy_is_lossless(k: int, seed_pair: tuple[int, int]) -> None:
    vocab, steps, start = 16, 40, 5
    sm, sg = seed_pair
    M = _markov(vocab, sm)
    G = _markov(vocab, sg)  # sg==sm -> perfect drafter (all accept); else imperfect
    assert _spec_greedy(M, G, start, steps, k) == _plain_greedy(M, start, steps)


def test_greedy_accept_counts() -> None:
    # d = [3, 1, 4]; base argmax = [3, 1, 9, 2] -> accept 2, correct to 9
    base = np.full((4, 16), -10.0)
    base[0, 3] = 1.0
    base[1, 1] = 1.0
    base[2, 9] = 1.0
    base[3, 2] = 1.0
    res = greedy_accept(np.array([3, 1, 4]), base)
    assert res.n_accepted == 2
    assert res.tokens.tolist() == [3, 1, 9]


def test_sampling_marginal_matches_base() -> None:
    # K=1: emitted token must be distributed as the base p, regardless of drafter q.
    rng = np.random.default_rng(0)
    vocab = 4
    p = np.array([0.5, 0.2, 0.2, 0.1])
    q = np.array([0.1, 0.1, 0.4, 0.4])  # deliberately mismatched drafter
    p_block = np.stack([p, p])          # [K+1=2, V]: p_1 (verify) + p_2 (bonus, unused at K=1 reject)
    counts = np.zeros(vocab)
    trials = 40000
    for _ in range(trials):
        d1 = rng.choice(vocab, p=q)                  # drafter samples d_1 ~ q
        res = speculative_sample_accept(
            np.array([d1]), q[None, :], p_block, rng
        )
        counts[int(res.tokens[0])] += 1
    emp = counts / trials
    assert np.max(np.abs(emp - p)) < 0.02, f"emp={emp} p={p}"


def test_adapter_contract_is_satisfiable() -> None:
    class _ToyAdapter:
        target_layer_ids = (0,)

        def prefill(self, tokens: mx.array) -> StepOut:
            return StepOut(mx.zeros((1, 4)), mx.zeros((1, 4)))

        def decode_step(self, token: mx.array) -> StepOut:
            return StepOut(mx.zeros((1, 4)), mx.zeros((1, 4)))

        def verify_forward(self, block_tokens: mx.array) -> BlockOut:
            return BlockOut(mx.zeros((1, 3, 4)), mx.zeros((1, 4)))

        def kv_snapshot(self):
            return None

        def kv_rollback(self, n_keep: int) -> None:
            return None

    assert isinstance(_ToyAdapter(), BaseModelAdapter)
