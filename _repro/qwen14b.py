import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx")
sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx/bench")
from run_dspark import _base_greedy, _dspark, _load_base, _load_drafter  # noqa: E402

from dspark_mlx.events import SummaryEvent  # noqa: E402
from dspark_mlx.loop import generate_eager  # noqa: E402
from datasets import load_dataset  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
model, tok, adapter = _load_base("qwen3_14b", "bf16")
drafter = _load_drafter("qwen3_14b")
eos = getattr(tok, "eos_token_id", None)
block = drafter.block_size
qs = [ex["question"] for ex in load_dataset("openai/gsm8k", "main", split="test").select(range(N))]


def enc(q, think):
    out = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                  tokenize=True, **({} if think else {"enable_thinking": False}))
    return [int(t) for t in (out["input_ids"] if hasattr(out, "keys") else out)]


def eval_tau(think):
    taus, accs, blks = [], 0, 0
    for q in qs:
        ids = enc(q, think)
        adapter.reset(); drafter.reset()
        ev = list(generate_eager(adapter, drafter, [ids], max_new_tokens=256, eos_id=eos))
        s = next(e for e in ev if isinstance(e, SummaryEvent))
        b = s.n_drafted // block
        if b:
            taus.append(s.n_accepted / b + 1)
        accs += s.n_accepted; blks += b
    print(f"[Qwen3-14B GSM8K think={think}] macro_tau={np.mean(taus):.2f} micro_tau={accs/blks + 1:.2f} "
          f"acc={100 * accs / (blks * block):.0f}%  (N={N})", flush=True)


eval_tau(True)
eval_tau(False)

# speedup (thinking off, the high-acceptance setting)
prompt = enc("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total?", False)
_dspark(adapter, drafter, prompt, 8, None, "eager")
_base_greedy(adapter, prompt, 8)
ds = _dspark(adapter, drafter, prompt, 160, None, "eager")
base = _base_greedy(adapter, prompt, len(ds.tokens))
print(f"[Qwen3-14B speedup, think=off] accepted/block={ds.n_accepted/max(1,ds.n_blocks):.2f} "
      f"base={base.tps:.1f} tok/s DSpark={ds.tps:.1f} tok/s speedup={base.seconds/ds.seconds:.2f}x", flush=True)
