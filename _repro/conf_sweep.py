import json
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx")
from dspark_mlx.events import SummaryEvent, TokenEvent
from dspark_mlx.hosts.mlx_lm import MlxLmHostAdapter
from dspark_mlx.loading import load_drafter
from dspark_mlx.loop import generate_eager
from dspark_mlx.registry import resolve_arch
from mlx_lm import load

R = "/Users/Fido/workspace/omlx/_research/dspark_multi"
cfg = json.load(open(f"{R}/qwen3_4b_config.json"))
model, tok = load("Qwen/Qwen3-4B")
arch = resolve_arch(cfg)
drafter = arch.build(cfg, max_seq_len=8192)
load_drafter(drafter, mx.load(f"{R}/ckpt/qwen3_4b.safetensors"), key_map=arch.key_map)
mx.eval(drafter.parameters())
adapter = MlxLmHostAdapter(model, target_layer_ids=cfg["target_layer_ids"])

prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Explain why speculative decoding speeds up LLM inference."}],
    add_generation_prompt=True, tokenize=True,
)
N = 128


def base_greedy(n):
    adapter.reset()
    s = adapter.prefill(mx.array([prompt], dtype=mx.int32))
    t = int(np.argmax(np.array(s.logits[0].astype(mx.float32)))); out = [t]
    while len(out) < n:
        s = adapter.decode_step(mx.array([t], dtype=mx.int32))
        t = int(np.argmax(np.array(s.logits[0].astype(mx.float32)))); out.append(t)
    return out


def run(threshold, n):
    adapter.reset(); drafter.reset()
    t0 = time.perf_counter()
    ev = list(generate_eager(adapter, drafter, [prompt], max_new_tokens=n, confidence_threshold=threshold))
    sec = time.perf_counter() - t0
    toks = [e.token for e in ev if isinstance(e, TokenEvent)]
    summ = next(e for e in ev if isinstance(e, SummaryEvent))
    blocks = sum(1 for _ in range(1))  # placeholder; recompute below
    return toks, sec, summ


# warmup + base
run(0.0, 8); base = base_greedy(8)
t0 = time.perf_counter(); base = base_greedy(N); base_sec = time.perf_counter() - t0
base_ref = base
print(f"base greedy: {N/base_sec:5.1f} tok/s")
print(f"{'thr':>4} {'tok/s':>7} {'speedup':>7} {'mean_acc/blk':>12} {'mean_draft/blk':>13} {'lossless':>9}")
for thr in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
    toks, sec, summ = run(thr, N)
    blocks = max(1, summ.n_emitted // 1)  # one block per cycle; derive from drafted
    # cycles = number of verify calls; reconstruct: each cycle emits >=1 token. Use accepted/drafted.
    # Approximate blocks by counting via a fresh instrumented pass is overkill; report rates.
    tps = len(toks) / sec
    diffs = sum(toks[i] != base_ref[i] for i in range(min(len(toks), len(base_ref))))
    print(f"{thr:>4.1f} {tps:>7.1f} {base_sec/sec:>6.2f}x  n_draft={summ.n_drafted:>4} "
          f"n_acc={summ.n_accepted:>3}  acc_rate={100*summ.n_accepted/max(1,summ.n_drafted):4.0f}%  "
          f"diff={diffs:>2}/{len(toks)}")
