import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx")
sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx/bench")
from run_dspark import _load_base, _load_drafter  # noqa: E402
from eval_accept import DATASETS, _encode  # noqa: E402

from dspark_mlx.events import SummaryEvent, TokenEvent  # noqa: E402
from dspark_mlx.loop import generate_eager  # noqa: E402
from dspark_mlx.quant import quantize_drafter  # noqa: E402

ARCH = next((a for a in sys.argv[1:] if not a.startswith("--")), "gemma4")
NO_THINK = "--no-think" in sys.argv
N = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--n=")), 10))
QUANT = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--quant=")), 8))
MAXT = 200
ORDER = ["gsm8k", "math500", "humaneval", "mbpp"]

model, tok, adapter = _load_base(ARCH, "bf16")
drafter = _load_drafter(ARCH)
if QUANT:
    quantize_drafter(drafter, bits=QUANT)
mx.eval(drafter.parameters())
eos = getattr(tok, "eos_token_id", None)
block = drafter.block_size


def base_greedy(ids, n):
    adapter.reset()
    t0 = time.perf_counter()
    s = adapter.prefill(mx.array([ids], dtype=mx.int32))
    t = int(mx.argmax(s.logits[0].astype(mx.float32)).item())
    cnt = 1
    while cnt < n:
        s = adapter.decode_step(mx.array([t], dtype=mx.int32))
        t = int(mx.argmax(s.logits[0].astype(mx.float32)).item())
        cnt += 1
    return time.perf_counter() - t0


def dspark(ids, n):
    adapter.reset(); drafter.reset()
    t0 = time.perf_counter()
    ev = list(generate_eager(adapter, drafter, [ids], max_new_tokens=n, eos_id=eos))
    sec = time.perf_counter() - t0
    nt = sum(1 for e in ev if isinstance(e, TokenEvent))
    s = next(e for e in ev if isinstance(e, SummaryEvent))
    b = s.n_drafted // block
    return nt, sec, (s.n_accepted / b + 1 if b else 0.0)


print(f"== {ARCH} cross-domain (N={N}, draft q{QUANT}, {'no-think' if NO_THINK else 'chat'}) ==", flush=True)
print(f"  {'dataset':10s} {'domain':5s} {'accepted_len':>12} {'accel':>7}", flush=True)
for name in ORDER:
    prompts = DATASETS[name](N)
    taus, speeds = [], []
    for q in prompts:
        ids = _encode(tok, q, chat=True, think=not NO_THINK)
        dspark(ids, 8); base_greedy(ids, 8)  # warmup
        nt, ds_sec, tau = dspark(ids, MAXT)
        b_sec = base_greedy(ids, nt)
        taus.append(tau); speeds.append(b_sec / ds_sec)
    domain = "code" if name in ("humaneval", "mbpp") else "math"
    print(f"  {name:10s} {domain:5s} {np.mean(taus):>12.2f} {np.median(speeds):>6.2f}x", flush=True)
