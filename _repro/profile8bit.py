import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx")
sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx/bench")
from run_dspark import _load_base, _load_drafter  # noqa: E402

from dspark_mlx.generate import _f32  # noqa: E402
from dspark_mlx.verify import greedy_accept  # noqa: E402

prec = sys.argv[1] if len(sys.argv) > 1 else "8bit"
model, tok, adapter = _load_base("qwen3", prec)
drafter = _load_drafter("qwen3")
block = drafter.block_size
prompt = tok.apply_chat_template([{"role": "user", "content": "Explain why speculative decoding speeds up LLM inference."}],
                                 add_generation_prompt=True, tokenize=True)
prompt = [int(t) for t in (prompt["input_ids"] if hasattr(prompt, "keys") else prompt)]
P = len(prompt)
T = {"draft": 0.0, "verify": 0.0, "base_decode_equiv": 0.0}


def run(n, measure=True):
    adapter.reset(); drafter.reset()
    step = adapter.prefill(mx.array([prompt], dtype=mx.int32))
    adapter.kv_rollback(P - 1)
    drafter.extend_context(step.main_hidden[:, :P - 1])
    anchor, start, emitted, ms = int(prompt[-1]), P - 1, 0, []
    while emitted < n:
        t0 = time.perf_counter()
        out = drafter.draft(mx.array([anchor], dtype=mx.int32))
        d = np.array(out[0])[0, 1:]
        if measure:
            T["draft"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        verify = np.concatenate([[anchor], d]).astype(np.int32)
        blk = adapter.verify_forward(mx.array(verify[None, :]))
        base_logits = _f32(blk.per_pos_logits)[0]
        if measure:
            T["verify"] += time.perf_counter() - t0
        res = greedy_accept(d, base_logits); m = res.n_accepted
        adapter.kv_rollback(start + m + 1)
        drafter.extend_context(blk.per_pos_main_hidden[:, :m + 1])
        ms.append(m)
        emitted += len(res.tokens.tolist())
        anchor, start = int(res.tokens[-1]), start + m + 1
    return ms


# also time a pure M=1 base decode (what plain greedy pays per token)
def time_base_decode(steps=40):
    adapter.reset()
    s = adapter.prefill(mx.array([prompt], dtype=mx.int32))
    tok1 = int(mx.argmax(s.logits[0].astype(mx.float32)).item())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        s = adapter.decode_step(mx.array([tok1], dtype=mx.int32))
        tok1 = int(mx.argmax(s.logits[0].astype(mx.float32)).item())
    mx.synchronize()
    return (time.perf_counter() - t0) / steps


run(8, measure=False)  # warmup
for k in T:
    T[k] = 0.0
ms = run(96)
base_dec = time_base_decode()
blocks = len(ms)
print(f"\n[{prec}] blocks={blocks} mean_accepted/block={np.mean(ms):.2f}")
print(f"  draft  total={T['draft']:.2f}s  per-cycle={T['draft']/blocks*1e3:.1f} ms")
print(f"  verify total={T['verify']:.2f}s  per-cycle={T['verify']/blocks*1e3:.1f} ms  (M={block+1} tokens)")
print(f"  base decode (M=1) per-token = {base_dec*1e3:.1f} ms  ->  verify(8) / decode(1) = {T['verify']/blocks/base_dec:.2f}x")
print(f"  draft : verify : split = {100*T['draft']/(T['draft']+T['verify']):.0f}% : {100*T['verify']/(T['draft']+T['verify']):.0f}%")
