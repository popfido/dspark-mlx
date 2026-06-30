import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx")
sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx/bench")
from run_dspark import _load_base, _load_drafter  # noqa: E402

from dspark_mlx.events import SummaryEvent  # noqa: E402
from dspark_mlx.loop import generate_eager  # noqa: E402
from datasets import load_dataset  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
model, tok, adapter = _load_base("qwen3", "bf16")
drafter = _load_drafter("qwen3")
eos = getattr(tok, "eos_token_id", None)
block = drafter.block_size
_ds = load_dataset("openai/gsm8k", "main", split="test")
print(f"GSM8K test split size: {len(_ds)}; evaluating N={N}", flush=True)
qs = [ex["question"] for ex in _ds.select(range(N))]


def enc(q, thinking):
    out = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                  tokenize=True, enable_thinking=thinking)
    return [int(t) for t in (out["input_ids"] if hasattr(out, "keys") else out)]


def run(thinking):
    taus, accs, blks = [], 0, 0
    for q in qs:
        ids = enc(q, thinking)
        adapter.reset(); drafter.reset()
        ev = list(generate_eager(adapter, drafter, [ids], max_new_tokens=256, eos_id=eos))
        s = next(e for e in ev if isinstance(e, SummaryEvent))
        b = s.n_drafted // block
        if b:
            taus.append(s.n_accepted / b + 1)
        accs += s.n_accepted; blks += b
    print(f"[thinking={thinking}] macro_tau={np.mean(taus):.2f} micro_tau={accs/blks + 1:.2f} "
          f"acc={100*accs/(blks*block):.0f}% plen0={len(enc(qs[0], thinking))}")


run(True)
run(False)
