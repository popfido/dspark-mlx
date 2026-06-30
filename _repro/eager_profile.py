import json
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx")
from dspark_mlx.generate import _f32
from dspark_mlx.hosts.mlx_lm import MlxLmHostAdapter
from dspark_mlx.loading import load_drafter
from dspark_mlx.recipe import draft_block_decode
from dspark_mlx.arch.qwen3 import rope_tables
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

prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Explain why speculative decoding speeds up LLM inference."}],
    add_generation_prompt=True, tokenize=True,
)
P, block = len(prompt), drafter.block_size
T = {"draft_backbone": 0.0, "draft_head": 0.0, "draft_decode": 0.0, "verify": 0.0, "commit": 0.0}


def timed_draft(anchor):
    """Replicate Qwen3DSparkDrafter.draft with internal timing."""
    d = drafter
    b = 1
    start = d._committed
    a = mx.array([anchor], dtype=mx.int32).reshape(b, 1)
    noise = mx.full((b, block - 1), d.args.mask_token_id, dtype=mx.int32)
    noise_embed = d.embed_tokens(mx.concatenate([a, noise], axis=1))
    bp = mx.arange(start, start + block)
    cos, sin = rope_tables(bp, d.args.head_dim, d.args.rope_theta)
    t0 = time.perf_counter()
    h = noise_embed
    ck, cv = d._ctx_k, d._ctx_v
    for i, layer in enumerate(d.backbone.layers):
        h = layer.forward_cached(h, ck[i], cv[i], cos, sin)
    bh = d.backbone.norm(h)
    mx.eval(bh)
    T["draft_backbone"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    logits = d.lm_head(bh.astype(mx.float32))
    mx.eval(logits)
    T["draft_head"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    out = draft_block_decode(logits, bh, mx.array([anchor], dtype=mx.int32),
                             d.markov_head, d.confidence_head, block, d.temperature)
    mx.eval(out[0])
    T["draft_decode"] += time.perf_counter() - t0
    return out


def run(n):
    adapter.reset(); drafter.reset()
    step = adapter.prefill(mx.array([prompt], dtype=mx.int32))
    adapter.kv_rollback(P - 1)
    drafter.extend_context(step.main_hidden[:, :P - 1])
    anchor, start, emitted, ms = int(prompt[-1]), P - 1, 0, []
    while emitted < n:
        out = timed_draft(anchor)
        d = np.array(out[0])[0, 1:]
        t0 = time.perf_counter()
        verify = np.concatenate([[anchor], d]).astype(np.int32)
        blk = adapter.verify_forward(mx.array(verify[None, :]))
        base_logits = _f32(blk.per_pos_logits)[0]
        T["verify"] += time.perf_counter() - t0
        res = greedy_accept(d, base_logits); m = res.n_accepted
        t0 = time.perf_counter()
        adapter.kv_rollback(start + m + 1)
        drafter.extend_context(blk.per_pos_main_hidden[:, :m + 1])
        mx.eval(drafter._ctx_k[0])
        T["commit"] += time.perf_counter() - t0
        ms.append(m)
        emitted += len(res.tokens.tolist())
        anchor, start = int(res.tokens[-1]), start + m + 1
    return ms


run(8)
for k in T:
    T[k] = 0.0
ms = run(128)
tot = sum(T.values())
print("eager+cache component time (s):", {k: round(v, 3) for k, v in T.items()},
      "| %:", {k: f"{100 * v / tot:.0f}" for k, v in T.items()})
print(f"blocks={len(ms)} mean_accepted/block={np.mean(ms):.2f} total={tot:.2f}s")
