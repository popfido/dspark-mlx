"""Simulation: sparse gather_mm dispatch vs the current dense expert-loop for the DSpark
DeepSeek-V4 draft MoE (256 experts, top-6, +1 shared). The real DeepSeek-V4 base is too large
to run locally, but the MoE block's perf is self-contained — synthetic stacked weights at the
real draft dims (dim=4096, inter=2048) reproduce the dispatch cost exactly.

  current path (model/moe.py): loop e in range(256): expert_e(x) * route_weight_e   -> O(experts)
  optimized path:              gather_mm over only the top-6 routed experts/token     -> O(topk)

Reports parity (sparse == dense within fp tol) + wall-clock speedup at the draft block sizes.

    python _repro/moe_gather_mm_sim.py
"""

import argparse
import time

import mlx.core as mx
import numpy as np

# DeepSeek-V4 DSpark draft MoE dims (dspark_mlx/model/config.py defaults)
DIM = 4096
INTER = 2048
N_EXPERTS = 256
TOPK = 6
LIMIT = 10.0  # swiglu_limit


def _silu(x):
    return x * mx.sigmoid(x)


def _expert_act(gate, up, limit):
    """DSpark asymmetric clamp: gate max-only, up both-sides (model/moe.py:Expert)."""
    gate = gate.astype(mx.float32)
    up = up.astype(mx.float32)
    if limit > 0:
        up = mx.clip(up, -limit, limit)
        gate = mx.minimum(gate, limit)
    return _silu(gate) * up


def make_weights(dim, inter, n_experts, dtype, key_seed):
    """Stacked expert weights [E, out, in] (nn.Linear layout) + gate + shared."""
    rng = np.random.default_rng(key_seed)
    s = 1.0 / np.sqrt(dim)

    def w(out, inn):
        return mx.array((rng.standard_normal((n_experts, out, inn)) * s).astype(np.float32)).astype(dtype)

    Wg = w(inter, dim)   # gate_proj  [E, inter, dim]
    Wu = w(inter, dim)   # up_proj
    Wd = mx.array((rng.standard_normal((n_experts, dim, inter)) / np.sqrt(inter)).astype(np.float32)).astype(dtype)
    gate_w = mx.array((rng.standard_normal((n_experts, dim)) * s).astype(np.float32))  # router (fp32)
    # shared expert (single)
    sg = mx.array((rng.standard_normal((inter, dim)) * s).astype(np.float32)).astype(dtype)
    su = mx.array((rng.standard_normal((inter, dim)) * s).astype(np.float32)).astype(dtype)
    sd = mx.array((rng.standard_normal((dim, inter)) / np.sqrt(inter)).astype(np.float32)).astype(dtype)
    return Wg, Wu, Wd, gate_w, (sg, su, sd)


def route(x, gate_w, topk):
    """sqrtsoftplus score routing (model/moe.py:Gate, score branch, no bias)."""
    scores = mx.matmul(x.astype(mx.float32), gate_w.T)
    sp = mx.maximum(scores, 0.0) + mx.log1p(mx.exp(-mx.abs(scores)))
    scores = mx.sqrt(sp)
    idx = mx.argsort(scores, axis=-1)[..., -topk:]
    w = mx.take_along_axis(scores, idx, axis=-1)
    w = w / mx.sum(w, axis=-1, keepdims=True)
    return w * 1.5, idx.astype(mx.int32)  # route_scale


def shared(x, sw, limit):
    sg, su, sd = sw
    h = _expert_act(x @ sg.T, x @ su.T, limit)
    return (h.astype(x.dtype) @ sd.T).astype(mx.float32)


def dense_moe(x, Wg, Wu, Wd, gate_w, sw, topk, limit):
    """Current path: evaluate every expert, mask by routing weight (model/moe.py:MoE)."""
    w, idx = route(x, gate_w, topk)
    y = mx.zeros(x.shape, dtype=mx.float32)
    for e in range(Wg.shape[0]):
        g = x @ Wg[e].T
        u = x @ Wu[e].T
        h = _expert_act(g, u, limit)
        oe = h.astype(x.dtype) @ Wd[e].T
        w_e = mx.sum(mx.where(idx == e, w, 0.0), axis=-1, keepdims=True)
        y = y + oe.astype(mx.float32) * w_e
    return y + shared(x, sw, limit)


def sparse_moe(x, Wg, Wu, Wd, gate_w, sw, topk, limit, sort=False):
    """Optimized path: gather_mm over only the top-k routed experts per token."""
    w, idx = route(x, gate_w, topk)          # [N, topk]
    xe = mx.expand_dims(x, (-2, -3))          # [N, 1, 1, dim]
    g = mx.gather_mm(xe, Wg.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=sort)  # [N,topk,1,inter]
    u = mx.gather_mm(xe, Wu.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=sort)
    h = _expert_act(g, u, limit).astype(x.dtype)                                     # [N,topk,1,inter]
    o = mx.gather_mm(h, Wd.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=sort)   # [N,topk,1,dim]
    o = o.squeeze(-2).astype(mx.float32)      # [N, topk, dim]
    y = mx.sum(o * mx.expand_dims(w, -1), axis=-2)
    return y + shared(x, sw, limit)


def _time(fn, iters=20):
    fn(); mx.eval(fn())  # warmup + compile
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t0) / iters * 1e3  # ms/call


def parity_check():
    dim, inter, E, topk = 64, 128, 8, 2
    Wg, Wu, Wd, gw, sw = make_weights(dim, inter, E, mx.float32, key_seed=0)
    x = mx.array(np.random.default_rng(1).standard_normal((5, dim)).astype(np.float32))
    d = dense_moe(x, Wg, Wu, Wd, gw, sw, topk, LIMIT)
    s = sparse_moe(x, Wg, Wu, Wd, gw, sw, topk, LIMIT)
    mx.eval(d, s)
    err = float(mx.max(mx.abs(d - s)))
    rel = err / float(mx.max(mx.abs(d)))
    print(f"parity (E={E},dim={dim}): max_abs_err={err:.2e} rel={rel:.2e} -> "
          f"{'OK' if rel < 1e-4 else 'MISMATCH'}")
    return rel < 1e-4


def perf(dtype, Ms, iters):
    print(f"\n=== DeepSeek-V4 draft MoE perf | dim={DIM} inter={INTER} experts={N_EXPERTS} "
          f"top-{TOPK} | dtype={dtype} ===")
    Wg, Wu, Wd, gw, sw = make_weights(DIM, INTER, N_EXPERTS, dtype, key_seed=2)
    mx.eval(Wg, Wu, Wd, gw, *sw)
    print(f"  {'M (tokens)':>10} {'dense ms':>9} {'sparse ms':>10} {'speedup':>8} {'parity rel':>11}")
    for M in Ms:
        x = mx.array(np.random.default_rng(M).standard_normal((M, DIM)).astype(np.float32)).astype(dtype)
        mx.eval(x)
        d = dense_moe(x, Wg, Wu, Wd, gw, sw, TOPK, LIMIT)
        s = sparse_moe(x, Wg, Wu, Wd, gw, sw, TOPK, LIMIT)
        mx.eval(d, s)
        rel = float(mx.max(mx.abs(d - s))) / (float(mx.max(mx.abs(d))) + 1e-9)
        dense_ms = _time(lambda: dense_moe(x, Wg, Wu, Wd, gw, sw, TOPK, LIMIT), iters)
        sparse_ms = _time(lambda: sparse_moe(x, Wg, Wu, Wd, gw, sw, TOPK, LIMIT), iters)
        print(f"  {M:>10} {dense_ms:>9.2f} {sparse_ms:>10.2f} {dense_ms / sparse_ms:>7.1f}x "
              f"{rel:>11.1e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 5, 8, 40],
                    help="token counts: 1=decode, 5=draft block, 8=verify, 40=batch")
    args = ap.parse_args()
    if not parity_check():
        raise SystemExit("parity failed -- fix the sparse forward before trusting perf")
    perf(mx.bfloat16 if args.dtype == "bf16" else mx.float32, args.ms, args.iters)


if __name__ == "__main__":
    main()
