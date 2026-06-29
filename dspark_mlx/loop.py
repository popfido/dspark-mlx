# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""Reference-matched ("eager") DSpark decode loop: one base forward per cycle.

The legacy :func:`dspark_mlx.generate.generate` runs two base forwards per cycle (a verify
over the K drafts plus a standalone decode of the anchor). The DeepSpec reference instead
folds the anchor into the verify forward -- ``verify_input_ids = [anchor, d_1..d_K]`` -- and
reads the anchor's hidden + p_1 back from position 0 (see ``draft_ops.build_dspark_proposal``
and the evaluator's ``_update``). That halves the per-cycle forward overhead and, crucially,
conditions the drafter exactly as it was trained (committed context excludes the live anchor;
the anchor is the block's query at its own position).

Invariant: the base KV cache holds the committed sequence EXCLUDING the current anchor. Each
cycle verifies [anchor, drafts] (positions start..start+K), commits [anchor, accepted], and
rolls the rest back. Output is identical to base greedy (b=1).
"""

from __future__ import annotations

from typing import Iterator, Optional

import mlx.core as mx
import numpy as np

from .events import SummaryEvent, TokenEvent
from .generate import _f32
from .verify import greedy_accept


def generate_eager(
    adapter,
    drafter,
    prompt_tokens,
    max_new_tokens: int,
    eos_id: Optional[int] = None,
) -> Iterator[object]:
    prompt = np.asarray(prompt_tokens).reshape(1, -1)
    P = prompt.shape[1]
    block = drafter.block_size

    step = adapter.prefill(mx.array(prompt.astype(np.int32)))   # base cache [0..P-1]
    adapter.kv_rollback(P - 1)                                  # drop the anchor (pos P-1)
    drafter.reset()
    drafter.extend_context(step.main_hidden[:, : P - 1])        # committed context [0..P-2]
    anchor = int(prompt[0, -1])
    start = P - 1                                               # anchor's absolute position

    n_emitted = n_drafted = n_accepted = 0
    while n_emitted < max_new_tokens:
        out = drafter.draft(mx.array([anchor], dtype=mx.int32))
        d = np.array(out[0])[0, 1:]                             # [block] drafts
        verify = np.concatenate([[anchor], d]).astype(np.int32)  # [anchor, d_1..d_block]
        blk = adapter.verify_forward(mx.array(verify[None, :]))
        base_logits = _f32(blk.per_pos_logits)[0]              # [block+1, V] = p_1..p_{block+1}
        res = greedy_accept(d, base_logits)
        m = res.n_accepted
        n_drafted += int(d.shape[0])
        n_accepted += m

        adapter.kv_rollback(start + m + 1)                     # keep [0..start+m]
        drafter.extend_context(blk.per_pos_main_hidden[:, : m + 1])  # commit anchor + m accepted

        stop = False
        for tok in res.tokens.tolist():
            yield TokenEvent(token=int(tok), n_accepted=m)
            n_emitted += 1
            if (eos_id is not None and int(tok) == eos_id) or n_emitted >= max_new_tokens:
                stop = True
                break
        if stop:
            break

        anchor = int(res.tokens[-1])
        start = start + m + 1

    yield SummaryEvent(n_emitted=n_emitted, n_drafted=n_drafted, n_accepted=n_accepted)
