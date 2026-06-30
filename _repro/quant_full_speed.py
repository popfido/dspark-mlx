import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx")
sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx/bench")
from run_dspark import _base_greedy, _dspark, _load_base, _load_drafter  # noqa: E402
from dspark_mlx.quant import quantize_drafter  # noqa: E402

model, tok, adapter = _load_base("qwen3", "bf16")
p = tok.apply_chat_template([{"role": "user", "content": "A robe takes 2 bolts of blue fiber and half that much white. How many bolts total? Show your work."}],
                            add_generation_prompt=True, tokenize=True, enable_thinking=False)
prompt = [int(t) for t in (p["input_ids"] if hasattr(p, "keys") else p)]


def mb(d):
    return sum(v.nbytes for _, v in tree_flatten(d.parameters())) / 1e6


def full(bits):
    d = _load_drafter("qwen3"); nn.quantize(d, group_size=64, bits=bits); mx.eval(d.parameters()); return d


makers = [
    ("bf16", lambda: _load_drafter("qwen3")),
    ("partial-q8", lambda: quantize_drafter(_load_drafter("qwen3"), bits=8)),
    ("ALL-q8", lambda: full(8)),
    ("ALL-q4", lambda: full(4)),
]
print("== qwen3 bf16 base, no-think ==")
for label, make in makers:
    d = make()
    _dspark(adapter, d, prompt, 8, None, "eager"); _base_greedy(adapter, prompt, 8)
    ds = _dspark(adapter, d, prompt, 160, None, "eager")
    base = _base_greedy(adapter, prompt, len(ds.tokens))
    print(f"  draft={label:11s} size={mb(d):6.0f}MB  acc/blk={ds.n_accepted/max(1,ds.n_blocks):.2f}  "
          f"DSpark={ds.tps:5.1f} tok/s  speedup={base.seconds/ds.seconds:.2f}x", flush=True)
