import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx")
sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx/bench")
from run_dspark import _load_base, _load_drafter  # noqa: E402

from dspark_mlx.events import SummaryEvent, TokenEvent  # noqa: E402
from dspark_mlx.loop import generate_eager  # noqa: E402
from dspark_mlx.quant import quantize_drafter  # noqa: E402
from datasets import load_dataset  # noqa: E402

NO_THINK = "--no-think" in sys.argv
ARCH = next((a for a in sys.argv[1:] if not a.startswith("--")), "qwen3")
N = 10
MAXT = 200
model, tok, adapter = _load_base(ARCH, "bf16")
eos = getattr(tok, "eos_token_id", None)
qs = [ex["question"] for ex in load_dataset("openai/gsm8k", "main", split="test").select(range(N))]


def enc(q):
    kw = {"enable_thinking": False} if NO_THINK else {}
    out = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True, **kw)
    return [int(t) for t in (out["input_ids"] if hasattr(out, "keys") else out)]


def base_greedy(ids, n):
    adapter.reset()
    t0 = time.perf_counter()
    s = adapter.prefill(mx.array([ids], dtype=mx.int32))
    t = int(mx.argmax(s.logits[0].astype(mx.float32)).item()); out = [t]
    while len(out) < n:
        s = adapter.decode_step(mx.array([t], dtype=mx.int32))
        t = int(mx.argmax(s.logits[0].astype(mx.float32)).item()); out.append(t)
    return time.perf_counter() - t0, len(out)


def dspark(drafter, ids, n):
    adapter.reset(); drafter.reset()
    t0 = time.perf_counter()
    ev = list(generate_eager(adapter, drafter, [ids], max_new_tokens=n, eos_id=eos))
    sec = time.perf_counter() - t0
    nt = sum(1 for e in ev if isinstance(e, TokenEvent))
    s = next(e for e in ev if isinstance(e, SummaryEvent))
    return sec, nt, (s.n_accepted / max(1, s.n_drafted // drafter.block_size) + 1)


for dbits in [None, 8]:
    drafter = _load_drafter(ARCH)
    if dbits:
        quantize_drafter(drafter, bits=dbits)
    speedups, taus = [], []
    for q in qs:
        ids = enc(q)
        dspark(drafter, ids, 8); base_greedy(ids, 8)  # warmup per prompt
        ds_sec, nt, tau = dspark(drafter, ids, MAXT)
        b_sec, _ = base_greedy(ids, nt)
        speedups.append(b_sec / ds_sec); taus.append(tau)
    lbl = "bf16" if dbits is None else f"q{dbits}"
    print(f"[DSpark draft={lbl} {'no-think' if NO_THINK else 'thinking-on'}] "
          f"GSM8K median speedup={np.median(speedups):.2f}x  tokens/cycle={np.median(taus):.2f}  (N={N})", flush=True)
