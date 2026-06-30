# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
#
# Parity of the assembled DSpark draft stack (attention + block + forward_embed) against
# the real DeepSeek reference, including the stateful sliding-window KV ring buffer (both
# the simple seqlen<=window case and the wraparound). Skips without the reference.
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from reference_model import load_reference_model

REF = load_reference_model()
needs_ref = pytest.mark.skipif(REF is None, reason="reference inference/model.py not available")

ATTN_ARGS = dict(
    dim=32, n_heads=4, head_dim=16, rope_head_dim=8, q_lora_rank=16, o_lora_rank=8,
    o_groups=2, window_size=8, norm_eps=1e-6, rope_theta=10000.0, n_layers=2,
    n_mtp_layers=1, compress_ratios=[0, 0, 0], max_batch_size=2, max_seq_len=64,
    n_routed_experts=4, moe_inter_dim=16, n_activated_experts=2, n_shared_experts=1,
    n_hash_layers=0, score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=10.0,
    vocab_size=64, hc_mult=4, hc_sinkhorn_iters=20,
    dspark_block_size=3, dspark_noise_token_id=63, dspark_target_layer_ids=[0, 1],
    dspark_markov_rank=8,
)


def _maxdiff(a: mx.array, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.array(a) - b)))


def _set_linear(ref_mod, mlx_mod, shape, rng):
    import torch

    w = (rng.standard_normal(shape) * 0.1).astype(np.float32)
    ref_mod.weight.data = torch.tensor(w)
    mlx_mod.weight = mx.array(w)


def _copy_attention(ref, mlx, args, rng):
    import torch

    dim, h, hd, ql, ol, g = args["dim"], args["n_heads"], args["head_dim"], args["q_lora_rank"], args["o_lora_rank"], args["o_groups"]
    _set_linear(ref.wq_a, mlx.wq_a, (ql, dim), rng)
    _set_linear(ref.wq_b, mlx.wq_b, (h * hd, ql), rng)
    _set_linear(ref.wkv, mlx.wkv, (hd, dim), rng)
    _set_linear(ref.wo_a, mlx.wo_a, (g * ol, h * hd // g), rng)
    _set_linear(ref.wo_b, mlx.wo_b, (dim, g * ol), rng)
    for ref_n, mlx_n in ((ref.q_norm, mlx.q_norm), (ref.kv_norm, mlx.kv_norm)):
        w = (rng.standard_normal(ref_n.weight.shape) * 0.1 + 1.0).astype(np.float32)
        ref_n.weight.data = torch.tensor(w)
        mlx_n.weight = mx.array(w)
    sink = (rng.standard_normal((h,)) * 0.1).astype(np.float32)
    ref.attn_sink.data = torch.tensor(sink)
    mlx.attn_sink = mx.array(sink)


@needs_ref
@pytest.mark.parametrize("prompt_len", [5, 12])  # <= window, and wraparound (> window)
def test_dspark_attention_parity(prompt_len: int) -> None:
    import torch

    from dspark_mlx.model.attention import DSparkAttention
    from dspark_mlx.model.config import DSparkArgs

    rng = np.random.default_rng(prompt_len)
    args = ATTN_ARGS
    ref = REF.DSparkAttention(0, REF.ModelArgs(**args))  # layer 0 -> compress_ratio 0
    mlx = DSparkAttention(DSparkArgs.from_dict(args), max_seq_len=args["max_seq_len"])
    _copy_attention(ref, mlx, args, rng)

    b, dim, block_size, win = 2, args["dim"], args["dspark_block_size"], args["window_size"]
    main_prompt = (rng.standard_normal((b, prompt_len, dim)) * 0.5).astype(np.float32)
    dummy = (rng.standard_normal((b, block_size, dim)) * 0.5).astype(np.float32)
    x_dec = (rng.standard_normal((b, block_size, dim)) * 0.5).astype(np.float32)
    main_dec = (rng.standard_normal((b, 1, dim)) * 0.5).astype(np.float32)

    with torch.no_grad():
        ref(torch.tensor(dummy), 0, torch.tensor(main_prompt))  # prefill -> ref.kv_cache
        r = ref(torch.tensor(x_dec), prompt_len, torch.tensor(main_dec)).detach().numpy()

    mlx(mx.array(dummy), 0, mx.array(main_prompt))  # prefill -> mlx.cache
    o = mlx(mx.array(x_dec), prompt_len, mx.array(main_dec))
    mx.eval(o)
    assert _maxdiff(o, r) <= 2e-3


def _copy_rmsnorm(ref_n, mlx_n, rng):
    import torch

    w = (rng.standard_normal(tuple(ref_n.weight.shape)) * 0.1 + 1.0).astype(np.float32)
    ref_n.weight.data = torch.tensor(w)
    mlx_n.weight = mx.array(w)


def _copy_moe(ref, mlx, args, rng):
    import torch

    dim, inter, e_count = args["dim"], args["moe_inter_dim"], args["n_routed_experts"]
    gw = (rng.standard_normal((e_count, dim)) * 0.1).astype(np.float32)
    gb = (rng.standard_normal((e_count,)) * 0.1).astype(np.float32)
    ref.gate.weight.data = torch.tensor(gw)
    ref.gate.bias.data = torch.tensor(gb)
    mlx.gate.weight = mx.array(gw)
    mlx.gate.bias = mx.array(gb)

    shapes = (("w1", (inter, dim)), ("w2", (dim, inter)), ("w3", (inter, dim)))

    def cp_expert(re, me):  # shared expert: me is an mlx Expert module
        for name, shape in shapes:
            w = (rng.standard_normal(shape) * 0.1).astype(np.float32)
            getattr(re, name).weight.data = torch.tensor(w)
            getattr(me, name).weight = mx.array(w)

    # routed experts: per-expert into the torch ref, stacked into the mlx MoE ([E, out, in])
    stacks = {"w1": [], "w2": [], "w3": []}
    for i in range(e_count):
        for name, shape in shapes:
            w = (rng.standard_normal(shape) * 0.1).astype(np.float32)
            getattr(ref.experts[i], name).weight.data = torch.tensor(w)
            stacks[name].append(w)
    for name in ("w1", "w2", "w3"):
        setattr(mlx, name, mx.array(np.stack(stacks[name])))
    cp_expert(ref.shared_experts, mlx.shared_experts)


def _copy_hc(ref, mlx, args, rng):
    import torch

    hc, dim = args["hc_mult"], args["dim"]
    mix_hc, hc_dim = (2 + hc) * hc, hc * dim
    for fn_n, sc_n, ba_n in (
        ("hc_attn_fn", "hc_attn_scale", "hc_attn_base"),
        ("hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"),
    ):
        fn = (rng.standard_normal((mix_hc, hc_dim)) * 0.1).astype(np.float32)
        sc = (rng.standard_normal((3,)) * 0.1).astype(np.float32)
        ba = (rng.standard_normal((mix_hc,)) * 0.1).astype(np.float32)
        getattr(ref, fn_n).data = torch.tensor(fn)
        getattr(ref, sc_n).data = torch.tensor(sc)
        getattr(ref, ba_n).data = torch.tensor(ba)
        setattr(mlx, fn_n, mx.array(fn))
        setattr(mlx, sc_n, mx.array(sc))
        setattr(mlx, ba_n, mx.array(ba))


@needs_ref
def test_dspark_block_forward_parity() -> None:
    import torch

    from dspark_mlx.model.block import DSparkBlock
    from dspark_mlx.model.config import DSparkArgs

    rng = np.random.default_rng(7)
    args = ATTN_ARGS
    ref = REF.DSparkBlock(args["n_layers"] + 0, REF.ModelArgs(**args))
    mlx = DSparkBlock(DSparkArgs.from_dict(args), 0, max_seq_len=args["max_seq_len"])
    _copy_attention(ref.attn, mlx.attn, args, rng)
    _copy_moe(ref.ffn, mlx.ffn, args, rng)
    _copy_rmsnorm(ref.attn_norm, mlx.attn_norm, rng)
    _copy_rmsnorm(ref.ffn_norm, mlx.ffn_norm, rng)
    _copy_hc(ref, mlx, args, rng)

    b, dim, bs, hc = 2, args["dim"], args["dspark_block_size"], args["hc_mult"]
    main_prompt = (rng.standard_normal((b, 5, dim)) * 0.5).astype(np.float32)
    dummy = (rng.standard_normal((b, bs, hc, dim)) * 0.5).astype(np.float32)
    x_dec = (rng.standard_normal((b, bs, hc, dim)) * 0.5).astype(np.float32)
    main_dec = (rng.standard_normal((b, 1, dim)) * 0.5).astype(np.float32)
    ids = rng.integers(0, args["vocab_size"], size=(b,)).astype(np.int64)

    with torch.no_grad():
        ref(torch.tensor(dummy), 0, torch.tensor(ids), torch.tensor(main_prompt))
        r = ref(torch.tensor(x_dec), 5, torch.tensor(ids), torch.tensor(main_dec)).detach().numpy()
    mlx(mx.array(dummy), 0, mx.array(ids.astype(np.int32)), mx.array(main_prompt))
    o = mlx(mx.array(x_dec), 5, mx.array(ids.astype(np.int32)), mx.array(main_dec))
    mx.eval(o)
    assert _maxdiff(o, r) <= 3e-3


@needs_ref
def test_forward_embed_parity() -> None:
    import mlx.nn as nn
    import torch

    from dspark_mlx.model.block import DSparkBlock
    from dspark_mlx.model.config import DSparkArgs

    rng = np.random.default_rng(8)
    args = ATTN_ARGS
    n_targets, dim, vocab = len(args["dspark_target_layer_ids"]), args["dim"], args["vocab_size"]
    ref = REF.DSparkBlock(args["n_layers"] + 0, REF.ModelArgs(**args))
    mlx = DSparkBlock(DSparkArgs.from_dict(args), 0, max_seq_len=args["max_seq_len"])

    _set_linear(ref.main_proj, mlx.main_proj, (dim, dim * n_targets), rng)
    _copy_rmsnorm(ref.main_norm, mlx.main_norm, rng)
    embed_w = (rng.standard_normal((vocab, dim)) * 0.1).astype(np.float32)
    ref.embed = REF.ParallelEmbedding(vocab, dim)
    ref.embed.weight.data = torch.tensor(embed_w)
    mlx_embed = nn.Embedding(vocab, dim)
    mlx_embed.weight = mx.array(embed_w)

    b = 2
    main_hidden = (rng.standard_normal((b, 1, dim * n_targets)) * 0.5).astype(np.float32)
    ids = rng.integers(0, vocab, size=(b,)).astype(np.int64)

    with torch.no_grad():
        rx, rmain = ref.forward_embed(torch.tensor(main_hidden), torch.tensor(ids))
    mx_x, mx_main = mlx.forward_embed(mx.array(main_hidden), mx.array(ids.astype(np.int32)), mlx_embed)
    mx.eval(mx_x, mx_main)
    assert _maxdiff(mx_x, rx.detach().numpy()) <= 1e-4
    assert _maxdiff(mx_main, rmain.detach().numpy()) <= 1e-4
