import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx")
sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx/bench")
from run_dspark import _load_base, _load_drafter  # noqa: E402

from dspark_mlx.events import SummaryEvent  # noqa: E402
from dspark_mlx.loop import generate_eager  # noqa: E402
from dspark_mlx.quant import quantize_drafter  # noqa: E402
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


def mb(d):
    return sum(v.nbytes for _, v in tree_flatten(d.parameters())) / 1e6


def measure(make, label):
    d = make()
    block = d.block_size
    taus, accs, blks = [], 0, 0
    for q in qs:
        ids = enc(q)
        adapter.reset(); d.reset()
        ev = list(generate_eager(adapter, d, [ids], max_new_tokens=256, eos_id=eos))
        s = next(e for e in ev if isinstance(e, SummaryEvent))
        b = s.n_drafted // block
        if b:
            taus.append(s.n_accepted / b + 1)
        accs += s.n_accepted; blks += b
    print(f"[{ARCH} {label:28s}] size={mb(d):6.0f} MB  accepted_len={accs/blks + 1:.2f}  acc={100*accs/(blks*block):.0f}%", flush=True)


def full_quant(bits):
    d = _load_drafter(ARCH)
    nn.quantize(d, group_size=64, bits=bits)
    mx.eval(d.parameters())
    return d


measure(lambda: _load_drafter(ARCH), "bf16 (none)")
measure(lambda: quantize_drafter(_load_drafter(ARCH), bits=8), "partial-q8 (Linears only)")
measure(lambda: full_quant(8), "ALL-q8 (incl embed+heads)")
measure(lambda: full_quant(4), "ALL-q4 (incl embed+heads)")
