# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
#
# Parity of the MLX DSpark layers against the real DeepSeek reference model.py (kernels
# stubbed to CPU, weights fp32). Skips when the reference code isn't available locally.
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from dspark_mlx.model.norm_rope import RMSNorm, apply_rotary_emb, precompute_rope
from reference_model import load_reference_model

REF = load_reference_model()
needs_ref = pytest.mark.skipif(REF is None, reason="reference inference/model.py not available")


def _maxdiff(a: mx.array, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.array(a) - b)))


@needs_ref
def test_rmsnorm_parity() -> None:
    import torch

    rng = np.random.default_rng(0)
    dim = 64
    w = (rng.standard_normal(dim) * 0.1 + 1.0).astype(np.float32)
    x = (rng.standard_normal((2, 5, dim)) * 0.5).astype(np.float32)

    ref = REF.RMSNorm(dim)
    ref.weight.data = torch.tensor(w)
    r = ref(torch.tensor(x)).detach().numpy()

    norm = RMSNorm(dim)
    norm.weight = mx.array(w)
    o = norm(mx.array(x))
    mx.eval(o)
    assert _maxdiff(o, r) <= 1e-4


@needs_ref
@pytest.mark.parametrize("ndim", [3, 4])
def test_rope_parity(ndim: int) -> None:
    import torch

    rng = np.random.default_rng(1)
    rd, seqlen, base = 64, 8, 10000.0
    shape = (2, seqlen, rd) if ndim == 3 else (2, seqlen, 4, rd)
    x = (rng.standard_normal(shape) * 0.5).astype(np.float32)

    # reference: precompute_freqs_cis(dim, seqlen, original_seq_len=0, base, factor, bf, bs)
    fc = REF.precompute_freqs_cis(rd, seqlen, 0, base, 1.0, 32, 1)
    r = REF.apply_rotary_emb(torch.tensor(x).clone(), fc).detach().numpy()

    cos, sin = precompute_rope(rd, seqlen, base)
    o = apply_rotary_emb(mx.array(x), cos, sin)
    mx.eval(o)
    assert _maxdiff(o, r) <= 1e-4


@needs_ref
def test_rope_inverse_roundtrips() -> None:
    rng = np.random.default_rng(2)
    rd, seqlen, base = 32, 6, 10000.0
    x = (rng.standard_normal((2, seqlen, 3, rd)) * 0.5).astype(np.float32)
    cos, sin = precompute_rope(rd, seqlen, base)
    rotated = apply_rotary_emb(mx.array(x), cos, sin)
    back = apply_rotary_emb(rotated, cos, sin, inverse=True)
    mx.eval(back)
    assert _maxdiff(back, x) <= 1e-4


def _set_expert(ref_e, mlx_e, rng, inter, dim) -> None:
    import torch

    for name in ("w1", "w2", "w3"):
        out_in = (inter, dim) if name in ("w1", "w3") else (dim, inter)
        w = (rng.standard_normal(out_in) * 0.1).astype(np.float32)
        getattr(ref_e, name).weight.data = torch.tensor(w)
        getattr(mlx_e, name).weight = mx.array(w)


@needs_ref
def test_moe_parity() -> None:
    import torch

    from dspark_mlx.model.config import DSparkArgs
    from dspark_mlx.model.moe import MoE

    rng = np.random.default_rng(3)
    e_count, dim, inter, topk = 8, 32, 16, 2
    cfg = dict(
        vocab_size=64, dim=dim, moe_inter_dim=inter, n_routed_experts=e_count,
        n_shared_experts=1, n_activated_experts=topk, n_hash_layers=0,
        score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=10.0,
    )
    ref = REF.MoE(0, REF.ModelArgs(**cfg))
    mlx = MoE(DSparkArgs.from_dict(cfg), 0)

    gate_w = (rng.standard_normal((e_count, dim)) * 0.1).astype(np.float32)
    gate_b = (rng.standard_normal((e_count,)) * 0.1).astype(np.float32)
    ref.gate.weight.data = torch.tensor(gate_w)
    ref.gate.bias.data = torch.tensor(gate_b)
    mlx.gate.weight = mx.array(gate_w)
    mlx.gate.bias = mx.array(gate_b)
    for i in range(e_count):
        _set_expert(ref.experts[i], mlx.experts[i], rng, inter, dim)
    _set_expert(ref.shared_experts, mlx.shared_experts, rng, inter, dim)

    x = (rng.standard_normal((2, 3, dim)) * 0.5).astype(np.float32)
    ids = rng.integers(0, 64, size=(2, 3)).astype(np.int64)
    with torch.no_grad():
        r = ref(torch.tensor(x), torch.tensor(ids)).detach().numpy()
    o = mlx(mx.array(x), mx.array(ids.astype(np.int32)))
    mx.eval(o)
    assert _maxdiff(o, r) <= 1e-3


@needs_ref
def test_hyper_connection_parity() -> None:
    import torch

    from dspark_mlx.model.hyper import hc_head, hc_post, hc_pre

    rng = np.random.default_rng(4)
    dim, hc = 32, 4
    args = REF.ModelArgs(
        dim=dim, hc_mult=hc, hc_sinkhorn_iters=20, n_heads=4, head_dim=16,
        rope_head_dim=8, q_lora_rank=16, o_lora_rank=16, o_groups=2, window_size=8,
        n_routed_experts=4, moe_inter_dim=16, n_activated_experts=2, n_hash_layers=0,
        vocab_size=64, max_batch_size=2, max_seq_len=64,
    )
    block = REF.Block(0, args)  # layer 0 -> compress_ratio 0, light attention dims

    mix_hc, hc_dim = (2 + hc) * hc, hc * dim
    fn = (rng.standard_normal((mix_hc, hc_dim)) * 0.1).astype(np.float32)
    sc = (rng.standard_normal((3,)) * 0.1).astype(np.float32)
    ba = (rng.standard_normal((mix_hc,)) * 0.1).astype(np.float32)
    block.hc_attn_fn.data = torch.tensor(fn)
    block.hc_attn_scale.data = torch.tensor(sc)
    block.hc_attn_base.data = torch.tensor(ba)

    x = (rng.standard_normal((2, 3, hc, dim)) * 0.5).astype(np.float32)
    with torch.no_grad():
        ry, rpost, rcomb = block.hc_pre(
            torch.tensor(x), block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
        )
    my_y, my_post, my_comb = hc_pre(
        mx.array(x), mx.array(fn), mx.array(sc), mx.array(ba), hc, 20, args.norm_eps, args.hc_eps
    )
    mx.eval(my_y, my_post, my_comb)
    assert _maxdiff(my_y, ry.numpy()) <= 1e-3
    assert _maxdiff(my_post, rpost.numpy()) <= 1e-4
    assert _maxdiff(my_comb, rcomb.numpy()) <= 1e-4

    residual = (rng.standard_normal((2, 3, hc, dim)) * 0.5).astype(np.float32)
    xp = (rng.standard_normal((2, 3, dim)) * 0.5).astype(np.float32)
    with torch.no_grad():
        rpo = block.hc_post(torch.tensor(xp), torch.tensor(residual), rpost, rcomb)
    my_po = hc_post(mx.array(xp), mx.array(residual), my_post, my_comb)
    mx.eval(my_po)
    assert _maxdiff(my_po, rpo.numpy()) <= 1e-3

    head_fn = (rng.standard_normal((hc, hc_dim)) * 0.1).astype(np.float32)
    head_sc = (rng.standard_normal((1,)) * 0.1).astype(np.float32)
    head_ba = (rng.standard_normal((hc,)) * 0.1).astype(np.float32)
    with torch.no_grad():
        rh = block.hc_head(
            torch.tensor(x), torch.tensor(head_fn), torch.tensor(head_sc), torch.tensor(head_ba)
        )
    my_h = hc_head(
        mx.array(x), mx.array(head_fn), mx.array(head_sc), mx.array(head_ba),
        args.norm_eps, args.hc_eps,
    )
    mx.eval(my_h)
    assert _maxdiff(my_h, rh.numpy()) <= 1e-3
