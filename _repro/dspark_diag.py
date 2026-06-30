import json
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx")
from dspark_mlx.hosts.mlx_lm import MlxLmHostAdapter
from dspark_mlx.loading import load_drafter
from dspark_mlx.registry import resolve_arch
from dspark_mlx.verify import greedy_accept
from mlx_lm import load

R = "/Users/Fido/workspace/omlx/_research/dspark_multi"
cfg = json.load(open(f"{R}/qwen3_4b_config.json"))
model, tok = load("Qwen/Qwen3-4B")
arch = resolve_arch(cfg)
drafter = arch.build(cfg, max_seq_len=8192)
load_drafter(drafter, mx.load(f"{R}/ckpt/qwen3_4b.safetensors"), key_map=arch.key_map)
mx.eval(drafter.parameters())
adapter = MlxLmHostAdapter(model, target_layer_ids=cfg["target_layer_ids"])
K = drafter.block_size

PROMPT = ("The history of computing is a story of abstraction. Each generation built tools that "
          "hid the complexity of the one before it, and in doing so")
prompt = tok.encode(PROMPT)
P = len(prompt)


def f32(a):
    return np.array(a.astype(mx.float32))


import time
T = {"draft": 0.0, "verify": 0.0, "decode": 0.0, "advance": 0.0}


def run_dspark_verbose(n):
    """Mirror generate.py but record per-block accepted counts + component wall time."""
    adapter.reset(); drafter.reset()
    s = adapter.prefill(mx.array([prompt], dtype=mx.int32))
    main_h = s.main_hidden
    p1 = f32(s.logits)[0]
    anchor = prompt[-1]
    drafter.forward_spec(mx.array([anchor], dtype=mx.int32), main_h, start_pos=0)
    main_h = main_h[:, -1:].reshape(1, 1, -1)
    t = P - 1
    emitted, ms = [], []
    while len(emitted) < n:
        t0 = time.perf_counter()
        out = drafter.forward_spec(mx.array([anchor], dtype=mx.int32), main_h, start_pos=t)
        d = np.array(out[0])[0, 1:]
        T["draft"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        block = adapter.verify_forward(mx.array(d[None, :].astype(np.int32)))
        base_logits = np.concatenate([p1[None, :], f32(block.per_pos_logits)[0]], axis=0)
        T["verify"] += time.perf_counter() - t0
        res = greedy_accept(d, base_logits)
        m = res.n_accepted
        ms.append(m)
        adapter.kv_rollback(t + 1 + m)
        t0 = time.perf_counter()
        pph = block.per_pos_main_hidden
        for j in range(m):
            drafter.advance(pph[:, j], t + 1 + j)
        mx.eval(drafter._ctx)
        T["advance"] += time.perf_counter() - t0
        for tk in res.tokens.tolist():
            emitted.append(int(tk))
        last = int(res.tokens[-1])
        t0 = time.perf_counter()
        s = adapter.decode_step(mx.array([last], dtype=mx.int32))
        p1 = f32(s.logits)[0]
        T["decode"] += time.perf_counter() - t0
        main_h = s.main_hidden.reshape(1, 1, -1)
        t = t + m + 1
        anchor = last
    return emitted[:n], ms


def seq_greedy(n):
    adapter.reset()
    s = adapter.prefill(mx.array([prompt], dtype=mx.int32))
    t = int(np.argmax(f32(s.logits)[0])); out = [t]
    while len(out) < n:
        s = adapter.decode_step(mx.array([t], dtype=mx.int32))
        t = int(np.argmax(f32(s.logits)[0])); out.append(t)
    return out


N = 96
ds, ms = run_dspark_verbose(N)
tot = sum(T.values())
print("component time (s):", {k: round(v, 3) for k, v in T.items()},
      "| %:", {k: f"{100*v/tot:.0f}" for k, v in T.items()})
ref = seq_greedy(N)
import collections
hist = collections.Counter(ms)
print(f"blocks={len(ms)}  mean accepted/block={np.mean(ms):.2f}  (K={K})")
print("per-block m:", ms)
print("histogram m->count:", dict(sorted(hist.items())))
diffs = [i for i in range(N) if ds[i] != ref[i]]
print(f"\nlossless: {len(diffs)}/{N} differ; first at {diffs[0] if diffs else None}")
if diffs:
    # re-derive base logits at the first divergence to inspect the top-2 gap (near-tie => bf16 jitter)
    i = diffs[0]
    adapter.reset()
    s = adapter.prefill(mx.array([prompt + ref[:i]], dtype=mx.int32))
    lg = f32(s.logits)[0]
    order = np.argsort(lg)[::-1]
    print(f"  first diff @ {i}: ds={ds[i]} ref={ref[i]}; top1={order[0]}({lg[order[0]]:.4f}) "
          f"top2={order[1]}({lg[order[1]]:.4f}) gap={lg[order[0]]-lg[order[1]]:.4f}")
    print(f"  ds token logit={lg[ds[i]]:.4f} ref token logit={lg[ref[i]]:.4f}")
