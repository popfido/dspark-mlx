# CPU (torch) reference implementations of the reference inference/kernel.py ops that
# the DSpark draft stack needs. Dependency-free of CUDA/tilelang so they run on CPU and
# double as (a) parity targets for the MLX ports and (b) the ``kernel`` module stub used
# to import the real reference model.py for block-level parity (see P1b-3).
#
# Faithful to inference/kernel.py:
#   sparse_attn          (L355-368 wrapper, L277-352 kernel): top-k gathered MLA attention
#                        with a denominator-only attention sink.
#   hc_split_sinkhorn    (L430-438 wrapper, L371-427 kernel): row-softmax -> col-norm ->
#                        (iters-1) row/col Sinkhorn normalizations.
# The fp8/fp4 quant ops are reduced to identity (inplace) / NotImplemented: the MLX drafter
# runs in bf16/fp32, so the fp8 QAT round-trip is intentionally dropped on both sides.
from __future__ import annotations

import torch


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """q:[b,m,h,d] kv:[b,n,d] attn_sink:[h] topk_idxs:[b,m,t] int (-1=mask) -> o:[b,m,h,d]."""
    b, m, h, d = q.shape
    t = topk_idxs.shape[-1]
    qf, kvf = q.float(), kv.float()
    mask = topk_idxs != -1                                   # [b,m,t]
    safe = topk_idxs.clamp_min(0).long()                     # [b,m,t]
    kv_g = torch.stack([kvf[bi][safe[bi]] for bi in range(b)], dim=0)  # [b,m,t,d]

    scores = torch.einsum("bmhd,bmtd->bmht", qf, kv_g) * softmax_scale
    scores = scores.masked_fill(~mask.unsqueeze(2), float("-inf"))
    smax = scores.max(dim=-1, keepdim=True).values           # [b,m,h,1]
    ex = torch.exp(scores - smax)                            # masked -> 0
    num = torch.einsum("bmht,bmtd->bmhd", ex, kv_g)          # [b,m,h,d]
    denom = ex.sum(-1, keepdim=True) + torch.exp(attn_sink.view(1, 1, h, 1).float() - smax)
    return (num / denom).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """mixes:[..,(2+hc)*hc] -> pre:[..,hc], post:[..,hc], comb:[..,hc,hc]."""
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]
    comb = comb.reshape(*mixes.shape[:-1], hc, hc)

    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


# --- quant ops: identity (drop fp8 QAT) so the real model.py imports + runs in fp32 ---

def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    if inplace:
        return x
    return x, x.new_ones(*x.shape[:-1], x.size(-1) // block_size)


def fp4_act_quant(x, block_size=32, inplace=False):
    if inplace:
        return x
    return x, x.new_ones(*x.shape[:-1], x.size(-1) // block_size)


def fp8_gemm(*args, **kwargs):  # pragma: no cover - bf16/fp32 path never calls this
    raise NotImplementedError("fp8_gemm is not used in the bf16/fp32 reference path")


def fp4_gemm(*args, **kwargs):  # pragma: no cover - bf16/fp32 path never calls this
    raise NotImplementedError("fp4_gemm is not used in the bf16/fp32 reference path")
