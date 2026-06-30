import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx")
sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx/bench")
from run_dspark import _load_base, _load_drafter  # noqa: E402

from dspark_mlx.events import SummaryEvent  # noqa: E402
from dspark_mlx.loop import generate_eager  # noqa: E402
from datasets import load_dataset  # noqa: E402

GEMMA_TMPL = "<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"

model, tok, adapter = _load_base("gemma4", "bf16")
drafter = _load_drafter("gemma4")
eos = getattr(tok, "eos_token_id", None)
block = drafter.block_size
qs = [ex["question"] for ex in load_dataset("openai/gsm8k", "main", split="test").select(range(8))]


def tau_for(encode_fn, label):
    taus, accs, blks = [], 0, 0
    for q in qs:
        ids = encode_fn(q)
        adapter.reset(); drafter.reset()
        ev = list(generate_eager(adapter, drafter, [ids], max_new_tokens=200, eos_id=eos))
        s = next(e for e in ev if isinstance(e, SummaryEvent))
        b = s.n_drafted // block
        if b:
            taus.append(s.n_accepted / b + 1)
        accs += s.n_accepted; blks += b
    print(f"[{label}] macro_tau={np.mean(taus):.2f} micro_tau={accs/blks + 1:.2f} acc={100*accs/(blks*block):.0f}%")


tau_for(lambda q: tok.encode(q), "raw")
tau_for(lambda q: tok.encode(GEMMA_TMPL.format(q=q)), "gemma-chat-format")
