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


def quantize_stack(W, bits, gs=64):
    """mx.quantize each [E, out, in] stack along the in dim -> (w_q, scales, biases)."""
    return mx.quantize(W, group_size=gs, bits=bits)


def sparse_moe_q(x, q1, q3, q2, gate_w, sw, topk, limit, bits, gs=64, sort=False):
    """Quantized optimized path: gather_qmm over only the top-k routed experts (q8/q4)."""
    w, idx = route(x, gate_w, topk)
    xe = mx.expand_dims(x, (-2, -3))
    kw = dict(rhs_indices=idx, transpose=True, group_size=gs, bits=bits, sorted_indices=sort)
    g = mx.gather_qmm(xe, *q1, **kw)          # [N,topk,1,inter]
    u = mx.gather_qmm(xe, *q3, **kw)
    h = _expert_act(g, u, limit).astype(x.dtype)
    o = mx.gather_qmm(h, *q2, **kw)           # [N,topk,1,dim]
    o = o.squeeze(-2).astype(mx.float32)
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


def _relerr(a, b):
    return float(mx.max(mx.abs(a - b))) / (float(mx.max(mx.abs(a))) + 1e-9)


def perf(dtype, Ms, iters):
    print(f"\n=== DeepSeek-V4 draft MoE perf | dim={DIM} inter={INTER} experts={N_EXPERTS} "
          f"top-{TOPK} | dtype={dtype} ===")
    Wg, Wu, Wd, gw, sw = make_weights(DIM, INTER, N_EXPERTS, dtype, key_seed=2)
    mx.eval(Wg, Wu, Wd, gw, *sw)
    # quantized expert stacks (the real DeepSeek-V4 draft ships fp8/fp4 experts)
    q8 = [quantize_stack(W, 8) for W in (Wg, Wu, Wd)]
    q4 = [quantize_stack(W, 4) for W in (Wg, Wu, Wd)]
    mx.eval(q8, q4)
    bytes_bf16 = sum(W.nbytes for W in (Wg, Wu, Wd))
    bytes_q8 = sum(t.nbytes for q in q8 for t in q)
    bytes_q4 = sum(t.nbytes for q in q4 for t in q)
    print(f"  expert mem: bf16 {bytes_bf16/1e9:.1f}GB | q8 {bytes_q8/1e9:.1f}GB "
          f"({bytes_bf16/bytes_q8:.1f}x) | q4 {bytes_q4/1e9:.1f}GB ({bytes_bf16/bytes_q4:.1f}x)")
    print(f"  {'M':>4} | {'dense':>7} | {'mm ms':>6} {'x':>5} {'err':>7} | "
          f"{'q8 ms':>6} {'x':>5} {'err':>7} | {'q4 ms':>6} {'x':>5} {'err':>7}")
    for M in Ms:
        x = mx.array(np.random.default_rng(M).standard_normal((M, DIM)).astype(np.float32)).astype(dtype)
        mx.eval(x)
        d = dense_moe(x, Wg, Wu, Wd, gw, sw, TOPK, LIMIT)
        s = sparse_moe(x, Wg, Wu, Wd, gw, sw, TOPK, LIMIT)
        s8 = sparse_moe_q(x, q8[0], q8[1], q8[2], gw, sw, TOPK, LIMIT, 8)
        s4 = sparse_moe_q(x, q4[0], q4[1], q4[2], gw, sw, TOPK, LIMIT, 4)
        mx.eval(d, s, s8, s4)
        e_mm, e8, e4 = _relerr(d, s), _relerr(d, s8), _relerr(d, s4)
        t_d = _time(lambda: dense_moe(x, Wg, Wu, Wd, gw, sw, TOPK, LIMIT), iters)
        t_mm = _time(lambda: sparse_moe(x, Wg, Wu, Wd, gw, sw, TOPK, LIMIT), iters)
        t8 = _time(lambda: sparse_moe_q(x, q8[0], q8[1], q8[2], gw, sw, TOPK, LIMIT, 8), iters)
        t4 = _time(lambda: sparse_moe_q(x, q4[0], q4[1], q4[2], gw, sw, TOPK, LIMIT, 4), iters)
        print(f"  {M:>4} | {t_d:>7.1f} | {t_mm:>6.2f} {t_d/t_mm:>4.0f}x {e_mm:>7.1e} | "
              f"{t8:>6.2f} {t_d/t8:>4.0f}x {e8:>7.1e} | {t4:>6.2f} {t_d/t4:>4.0f}x {e4:>7.1e}",
              flush=True)


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
