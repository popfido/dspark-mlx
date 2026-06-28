# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""Lossless greedy speculative decode loop tying the drafter, base adapter, and accept.

One cycle (anchor x_t at position t, base prediction p_1 = P(x_{t+1}) in hand):
  1. drafter.forward_spec drafts d_1..d_K for x_{t+1}..x_{t+K} (and slides its window to t).
  2. adapter.verify_forward runs ONE base forward over the block -> p_2..p_{K+1}.
  3. greedy_accept against [p_1, p_2..p_{K+1}] -> m accepted + a correction/bonus.
  4. roll the base KV back to the accepted prefix; slide the drafter window over the
     m accepted tokens (the next forward_spec adds the new anchor).
  5. one base decode_step on the last emitted token -> next p_1, main hidden, position.

Output is identical to greedy decoding from the base model alone (the drafter only affects
how many tokens are confirmed per base forward, never which tokens are emitted). b=1.
"""

from __future__ import annotations

from typing import Iterator, Optional

import mlx.core as mx
import numpy as np

from .adapter import BaseModelAdapter
from .events import SummaryEvent, TokenEvent
from .model.drafter import DSparkDrafter
from .verify import greedy_accept


def generate(
    adapter: BaseModelAdapter,
    drafter: DSparkDrafter,
    prompt_tokens,
    max_new_tokens: int,
    eos_id: Optional[int] = None,
) -> Iterator[object]:
    prompt = np.asarray(prompt_tokens).reshape(1, -1)
    prompt_len = prompt.shape[1]

    step = adapter.prefill(mx.array(prompt.astype(np.int32)))
    main_hidden = step.main_hidden                       # [1, prompt_len, D]
    p1 = np.array(step.logits)[0]                        # [V]
    anchor = int(prompt[0, -1])
    drafter.forward_spec(mx.array([anchor], dtype=mx.int32), main_hidden, start_pos=0)
    main_hidden = main_hidden[:, -1:].reshape(1, 1, -1)  # h_{t}, t = prompt_len - 1
    t = prompt_len - 1

    n_emitted = n_drafted = n_accepted = 0
    while n_emitted < max_new_tokens:
        out = drafter.forward_spec(mx.array([anchor], dtype=mx.int32), main_hidden, start_pos=t)
        d = np.array(out[0])[0, 1:]                       # [K] drafts for x_{t+1}..x_{t+K}
        block = adapter.verify_forward(mx.array(d[None, :].astype(np.int32)))
        base_logits = np.concatenate([p1[None, :], np.array(block.per_pos_logits)[0]], axis=0)
        res = greedy_accept(d, base_logits)
        m = res.n_accepted
        n_drafted += int(d.shape[0])
        n_accepted += m

        adapter.kv_rollback(t + 1 + m)
        pph = block.per_pos_main_hidden                   # [1, K, D]
        for j in range(m):
            drafter.advance(pph[:, j], t + 1 + j)

        stop = False
        for tok in res.tokens.tolist():
            yield TokenEvent(token=int(tok), n_accepted=m)
            n_emitted += 1
            if (eos_id is not None and int(tok) == eos_id) or n_emitted >= max_new_tokens:
                stop = True
                break
        if stop:
            break

        last = int(res.tokens[-1])
        step = adapter.decode_step(mx.array([last], dtype=mx.int32))
        p1 = np.array(step.logits)[0]
        main_hidden = step.main_hidden.reshape(1, 1, -1)
        t = t + m + 1
        anchor = last

    yield SummaryEvent(n_emitted=n_emitted, n_drafted=n_drafted, n_accepted=n_accepted)
