import json
import sys

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx")
from dspark_mlx.events import TokenEvent
from dspark_mlx.hosts.mlx_lm import MlxLmHostAdapter
from dspark_mlx.loading import load_drafter
from dspark_mlx.loop import generate_eager
from dspark_mlx.registry import resolve_arch
from mlx_lm import load

R = "/Users/Fido/workspace/omlx/_research/dspark_multi"
cfg = json.load(open(f"{R}/qwen3_4b_config.json"))
model, tok = load("Qwen/Qwen3-4B")
arch = resolve_arch(cfg)


def fresh_drafter():
    d = arch.build(cfg, max_seq_len=8192)
    load_drafter(d, mx.load(f"{R}/ckpt/qwen3_4b.safetensors"), key_map=arch.key_map)
    mx.eval(d.parameters())
    return d


def acc_stats(adapter, drafter, prompt_ids, n):
    adapter.reset(); drafter.reset()
    ev = list(generate_eager(adapter, drafter, [prompt_ids], max_new_tokens=n))
    summ = ev[-1]
    blocks = summ.n_drafted // drafter.block_size
    toks = [e.token for e in ev if isinstance(e, TokenEvent)]
    return toks, summ.n_accepted / blocks, blocks


def seq_greedy(adapter, prompt_ids, n):
    adapter.reset()
    s = adapter.prefill(mx.array([prompt_ids], dtype=mx.int32))
    t = int(np.argmax(np.array(s.logits[0].astype(mx.float32)))); out = [t]
    while len(out) < n:
        s = adapter.decode_step(mx.array([t], dtype=mx.int32))
        t = int(np.argmax(np.array(s.logits[0].astype(mx.float32)))); out.append(t)
    return out


# ---- Test B (bf16): raw vs chat-formatted prompt acceptance ----
RAW = ("The history of computing is a story of abstraction. Each generation built tools that "
       "hid the complexity of the one before it, and in doing so")
CHAT = tok.apply_chat_template(
    [{"role": "user", "content": "Explain why speculative decoding speeds up LLM inference."}],
    add_generation_prompt=True, tokenize=True,
)
adapter = MlxLmHostAdapter(model, target_layer_ids=cfg["target_layer_ids"])
for name, ids in [("raw", tok.encode(RAW)), ("chat", CHAT)]:
    _, mean_acc, blocks = acc_stats(adapter, fresh_drafter(), ids, 96)
    print(f"[bf16 {name:4s}] prompt={len(ids):3d} blocks={blocks:2d} mean_accepted/block={mean_acc:.2f}")

# ---- Test A (float32): eager must equal sequential greedy (no bf16 tie artifact) ----
model.update(tree_map(lambda x: x.astype(mx.float32) if x.dtype == mx.bfloat16 else x, model.parameters()))
mx.eval(model.parameters())
adapter = MlxLmHostAdapter(model, target_layer_ids=cfg["target_layer_ids"])
ids = tok.encode(RAW)
eager_toks, mean_acc, _ = acc_stats(adapter, fresh_drafter(), ids, 48)
ref = seq_greedy(adapter, ids, 48)
diffs = [i for i in range(48) if eager_toks[i] != ref[i]]
print(f"[f32 raw ] mean_accepted/block={mean_acc:.2f}  lossless: {len(diffs)}/48 differ "
      f"(first {diffs[0] if diffs else None}) -> {'CLEAN' if not diffs else 'DIVERGES'}")
