# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
#
# The reference-matched eager loop (one base forward/cycle) must stay lossless. Reuse the tiny
# real mlx-lm Qwen3 base + tiny drafter from the host test: lossless holds for any drafter, so
# the eager loop's output must equal base greedy regardless of draft quality.
from __future__ import annotations

import mlx.core as mx
import numpy as np

from dspark_mlx.events import SummaryEvent, TokenEvent
from dspark_mlx.hosts.mlx_lm import MlxLmHostAdapter
from dspark_mlx.loop import generate_eager
from tests.test_host_mlx_lm import TIDS, VOCAB, _base_greedy, _tiny_drafter, _tiny_qwen3_base


def test_eager_loop_is_lossless() -> None:
    mx.random.seed(1234)
    base = _tiny_qwen3_base()
    drafter = _tiny_drafter()
    adapter = MlxLmHostAdapter(base, target_layer_ids=TIDS)

    prompt = np.random.default_rng(0).integers(0, VOCAB, size=(1, 7)).astype(np.int32)
    events = list(generate_eager(adapter, drafter, prompt, max_new_tokens=24))
    tokens = [e.token for e in events if isinstance(e, TokenEvent)]

    assert isinstance(events[-1], SummaryEvent)
    assert len(tokens) == 24
    assert tokens == _base_greedy(adapter, prompt, 24)
    assert events[-1].n_drafted > 0


def test_eager_confidence_gating_stays_lossless() -> None:
    """Confidence-gated draft length only changes speculation depth, never the output."""
    mx.random.seed(1234)
    base = _tiny_qwen3_base()
    drafter = _tiny_drafter()
    adapter = MlxLmHostAdapter(base, target_layer_ids=TIDS)
    prompt = np.random.default_rng(0).integers(0, VOCAB, size=(1, 7)).astype(np.int32)

    tokens = [e.token for e in generate_eager(adapter, drafter, prompt, max_new_tokens=24,
                                              confidence_threshold=0.5)
              if isinstance(e, TokenEvent)]
    assert tokens == _base_greedy(adapter, prompt, 24)


def test_eager_loop_lossless_with_quantized_draft() -> None:
    """A quantized drafter only changes draft quality (acceptance), never the output."""
    from dspark_mlx.quant import quantize_drafter

    mx.random.seed(1234)
    base = _tiny_qwen3_base()
    drafter = quantize_drafter(_tiny_drafter(), bits=8, group_size=64)
    adapter = MlxLmHostAdapter(base, target_layer_ids=TIDS)
    prompt = np.random.default_rng(0).integers(0, VOCAB, size=(1, 7)).astype(np.int32)
    tokens = [e.token for e in generate_eager(adapter, drafter, prompt, max_new_tokens=24)
              if isinstance(e, TokenEvent)]
    assert tokens == _base_greedy(adapter, prompt, 24)


def test_eager_matches_legacy_token_stream() -> None:
    """Eager and legacy loops are both lossless, so they emit the same tokens (b=1, greedy)."""
    from dspark_mlx.generate import generate

    mx.random.seed(7)
    base = _tiny_qwen3_base()
    adapter = MlxLmHostAdapter(base, target_layer_ids=TIDS)
    prompt = np.random.default_rng(3).integers(0, VOCAB, size=(1, 6)).astype(np.int32)

    adapter.reset()
    d1 = _tiny_drafter()
    eager = [e.token for e in generate_eager(adapter, d1, prompt, max_new_tokens=20)
             if isinstance(e, TokenEvent)]
    adapter.reset()
    d2 = _tiny_drafter()
    legacy = [e.token for e in generate(adapter, d2, prompt, max_new_tokens=20)
              if isinstance(e, TokenEvent)]
    assert eager == legacy
