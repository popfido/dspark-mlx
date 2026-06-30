import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx")
sys.path.insert(0, "/Users/Fido/workspace/dspark-mlx/bench")
from run_dspark import _load_drafter  # noqa: E402
from dspark_mlx.quant import quantize_drafter  # noqa: E402

ARCH = sys.argv[1] if len(sys.argv) > 1 else "qwen3"


def mb(params):
    return sum(v.nbytes for _, v in tree_flatten(params)) / 1e6


def breakdown(d):
    return {
        "embed_tokens": mb(d.embed_tokens.parameters()),
        "lm_head": mb(d.lm_head.parameters()),
        "backbone(layers+fc+norm)": mb(d.backbone.parameters()),
        "markov_head": mb(d.markov_head.parameters()),
        "confidence_head": mb(d.confidence_head.parameters()),
    }


d = _load_drafter(ARCH)
bd = breakdown(d)
total = mb(d.parameters())
print(f"\n## {ARCH} draft — bf16 component sizes (MB)")
for k, v in bd.items():
    print(f"  {k:28s} {v:8.1f}  ({100*v/total:4.1f}%)")
print(f"  {'TOTAL':28s} {total:8.1f}")

# (a) current quantize_drafter: compute Linears (incl lm_head), skip embed + markov/confidence
d_a = quantize_drafter(_load_drafter(ARCH), bits=8)
# (b) quantize EVERYTHING (Linear + Embedding, incl heads) — the default nn.quantize
d_b = _load_drafter(ARCH)
nn.quantize(d_b, group_size=64, bits=8)
mx.eval(d_b.parameters())
# (c) everything at 4-bit
d_c = _load_drafter(ARCH)
nn.quantize(d_c, group_size=64, bits=4)
mx.eval(d_c.parameters())
print(f"\n## total size (MB) vs bf16 {total:.0f}")
print(f"  quantize_drafter q8 (Linears only)      {mb(d_a.parameters()):8.1f}  ({mb(d_a.parameters())/total:.2f}x)")
print(f"  quantize ALL q8 (Linear+Embedding)      {mb(d_b.parameters()):8.1f}  ({mb(d_b.parameters())/total:.2f}x)")
print(f"  quantize ALL q4                         {mb(d_c.parameters()):8.1f}  ({mb(d_c.parameters())/total:.2f}x)")
