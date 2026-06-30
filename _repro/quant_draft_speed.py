import sys

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx")
sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx/bench")
from run_dspark import _base_greedy, _dspark, _load_base, _load_drafter  # noqa: E402

ARCH = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
BASE_PREC = sys.argv[2] if len(sys.argv) > 2 else "bf16"
NO_THINK = "--no-think" in sys.argv

model, tok, adapter = _load_base(ARCH, BASE_PREC)
kw = {"enable_thinking": False} if NO_THINK else {}
p = tok.apply_chat_template([{"role": "user", "content": "A robe takes 2 bolts of blue fiber and half that much white. How many bolts total? Show your work."}],
                            add_generation_prompt=True, tokenize=True, **kw)
prompt = [int(t) for t in (p["input_ids"] if hasattr(p, "keys") else p)]


def _pred(path, m):
    return isinstance(m, nn.Linear) and "markov_head" not in path and "confidence_head" not in path


print(f"== {ARCH} base={BASE_PREC} {'no-think' if NO_THINK else ''} ==")
for dbits in [None, 8, 4]:
    drafter = _load_drafter(ARCH)
    if dbits is not None:
        nn.quantize(drafter, group_size=64, bits=dbits, class_predicate=_pred)
        mx.eval(drafter.parameters())
    _dspark(adapter, drafter, prompt, 8, None, "eager")  # warmup
    _base_greedy(adapter, prompt, 8)
    ds = _dspark(adapter, drafter, prompt, 160, None, "eager")
    base = _base_greedy(adapter, prompt, len(ds.tokens))
    lbl = "bf16" if dbits is None else f"q{dbits}"
    print(f"  draft={lbl:>4}  acc/blk={ds.n_accepted/max(1,ds.n_blocks):.2f}  "
          f"base={base.tps:5.1f}  DSpark={ds.tps:5.1f} tok/s  speedup={base.seconds/ds.seconds:.2f}x", flush=True)
