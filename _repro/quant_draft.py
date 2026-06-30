import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx")
sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx/bench")
from run_dspark import _load_base, _load_drafter  # noqa: E402

from dspark_mlx.events import SummaryEvent  # noqa: E402
from dspark_mlx.loop import generate_eager  # noqa: E402
from datasets import load_dataset  # noqa: E402

ARCH = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
NO_THINK = "--no-think" in sys.argv
N = 20
model, tok, adapter = _load_base(ARCH, "bf16")
eos = getattr(tok, "eos_token_id", None)
qs = [ex["question"] for ex in load_dataset("openai/gsm8k", "main", split="test").select(range(N))]


def enc(q):
    kw = {"enable_thinking": False} if NO_THINK else {}
    out = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True, **kw)
    return [int(t) for t in (out["input_ids"] if hasattr(out, "keys") else out)]


# quantize the compute Linears + lm_head, but keep the Markov/confidence heads in bf16
def _pred(path, m):
    if not isinstance(m, nn.Linear):
        return False
    return "markov_head" not in path and "confidence_head" not in path


def measure(drafter, label):
    block = drafter.block_size
    taus, accs, blks = [], 0, 0
    for q in qs:
        ids = enc(q)
        adapter.reset(); drafter.reset()
        ev = list(generate_eager(adapter, drafter, [ids], max_new_tokens=256, eos_id=eos))
        s = next(e for e in ev if isinstance(e, SummaryEvent))
        b = s.n_drafted // block
        if b:
            taus.append(s.n_accepted / b + 1)
        accs += s.n_accepted; blks += b
    print(f"[{ARCH} draft={label:>5}] accepted_len={accs/blks + 1:.2f}  acc={100*accs/(blks*block):.0f}%", flush=True)


for bits in [None, 8, 4]:
    drafter = _load_drafter(ARCH)
    if bits is not None:
        nn.quantize(drafter, group_size=64, bits=bits, class_predicate=_pred)
        mx.eval(drafter.parameters())
    measure(drafter, "bf16" if bits is None else f"q{bits}")
