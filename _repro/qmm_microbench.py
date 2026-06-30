import time

import mlx.core as mx

# Representative Qwen3-4B matmul shapes (the quantized weights the verify forward hits).
SHAPES = {
    "down_proj  (9728->2560)": (9728, 2560),
    "o_proj     (2560->2560)": (2560, 2560),
    "gate_proj  (2560->9728)": (2560, 9728),
    "lm_head    (2560->152k)": (2560, 151936),
}
MS = [1, 4, 8, 16, 32, 64, 128, 256, 512]


def bench(fn, iters=60):
    for _ in range(8):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def qmm_fn(x, wq, s, b, gs, bits):
    return lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=gs, bits=bits)


for name, (K, N) in SHAPES.items():
    w = mx.random.normal((N, K)).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=64, bits=8)
    wbf = w  # bf16 reference
    print(f"\n## {name}   [8-bit q vs bf16, per-row efficiency]")
    print(f"  {'M':>4} {'qmm us':>9} {'q TFLOP/s':>10} {'q us/row':>9} {'bf16 us':>9} {'bf16 TFLOP/s':>12}")
    base_perrow = None
    for M in MS:
        x = mx.random.normal((M, K)).astype(mx.bfloat16)
        tq = bench(qmm_fn(x, wq, scales, biases, 64, 8))
        tb = bench(lambda: x @ wbf.T)
        flop = 2 * M * K * N
        qtf, btf = flop / tq / 1e12, flop / tb / 1e12
        perrow = tq / M * 1e6
        if base_perrow is None:
            base_perrow = perrow
        print(f"  {M:>4} {tq * 1e6:>9.1f} {qtf:>10.2f} {perrow:>9.2f} {tb * 1e6:>9.1f} {btf:>12.2f}")
