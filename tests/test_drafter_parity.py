# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
#
# Parity of the draft-token head (forward_head) and the full draft stack (forward_spec)
# against the real DeepSeek reference. Temperature 0 makes sampling deterministic so the
# emitted draft ids, biased logits, and confidence all match exactly. Skips without ref.
from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch

from test_block_parity import (
    ATTN_ARGS,
    REF,
    _copy_attention,
    _copy_hc,
    _copy_moe,
    _copy_rmsnorm,
    _maxdiff,
    _set_linear,
    needs_ref,
)


def _copy_head_modules(ref_blk, mlx_blk, args, rng):
    dim, vocab, rank, hc = args["dim"], args["vocab_size"], args["dspark_markov_rank"], args["hc_mult"]
    hc_dim = hc * dim
    _copy_rmsnorm(ref_blk.norm, mlx_blk.norm, rng)
    mw1 = (rng.standard_normal((vocab, rank)) * 0.1).astype(np.float32)
    mw2 = (rng.standard_normal((vocab, rank)) * 0.1).astype(np.float32)
    ref_blk.markov_head.markov_w1.weight.data = torch.tensor(mw1)
    ref_blk.markov_head.markov_w2.weight.data = torch.tensor(mw2)
    mlx_blk.markov_head.markov_w1.weight = mx.array(mw1)
    mlx_blk.markov_head.markov_w2.weight = mx.array(mw2)
    cw = (rng.standard_normal((1, dim + rank)) * 0.1).astype(np.float32)
    ref_blk.confidence_head.proj.weight.data = torch.tensor(cw)
    mlx_blk.confidence_head.proj.weight = mx.array(cw)
    hfn = (rng.standard_normal((hc, hc_dim)) * 0.1).astype(np.float32)
    hsc = (rng.standard_normal((1,)) * 0.1).astype(np.float32)
    hba = (rng.standard_normal((hc,)) * 0.1).astype(np.float32)
    ref_blk.hc_head_fn.data = torch.tensor(hfn)
    ref_blk.hc_head_scale.data = torch.tensor(hsc)
    ref_blk.hc_head_base.data = torch.tensor(hba)
    mlx_blk.hc_head_fn = mx.array(hfn)
    mlx_blk.hc_head_scale = mx.array(hsc)
    mlx_blk.hc_head_base = mx.array(hba)


def _copy_block(ref_blk, mlx_blk, args, rng):
    _copy_attention(ref_blk.attn, mlx_blk.attn, args, rng)
    _copy_moe(ref_blk.ffn, mlx_blk.ffn, args, rng)
    _copy_rmsnorm(ref_blk.attn_norm, mlx_blk.attn_norm, rng)
    _copy_rmsnorm(ref_blk.ffn_norm, mlx_blk.ffn_norm, rng)
    _copy_hc(ref_blk, mlx_blk, args, rng)
    if hasattr(mlx_blk, "main_proj"):
        n_targets, dim = len(args["dspark_target_layer_ids"]), args["dim"]
        _set_linear(ref_blk.main_proj, mlx_blk.main_proj, (dim, dim * n_targets), rng)
        _copy_rmsnorm(ref_blk.main_norm, mlx_blk.main_norm, rng)
    if getattr(mlx_blk, "is_last", False):
        _copy_head_modules(ref_blk, mlx_blk, args, rng)


@needs_ref
def test_forward_head_parity() -> None:
    from dspark_mlx.model.block import DSparkBlock
    from dspark_mlx.model.config import DSparkArgs

    rng = np.random.default_rng(10)
    args = {**ATTN_ARGS, "temperature": 0.0, "n_mtp_layers": 1}
    dim, vocab, bs, hc = args["dim"], args["vocab_size"], args["dspark_block_size"], args["hc_mult"]
    ref = REF.DSparkBlock(args["n_layers"] + 0, REF.ModelArgs(**args))
    mlx = DSparkBlock(DSparkArgs.from_dict(args), 0, max_seq_len=args["max_seq_len"])
    _copy_head_modules(ref, mlx, args, rng)

    hw = (rng.standard_normal((vocab, dim)) * 0.1).astype(np.float32)
    ref.head = REF.ParallelHead(vocab, dim)
    ref.head.weight.data = torch.tensor(hw)
    mlx_head = mx_linear(dim, vocab, hw)

    b = 2
    h = (rng.standard_normal((b, bs, hc, dim)) * 0.5).astype(np.float32)
    anchor = rng.integers(0, vocab, size=(b,)).astype(np.int64)
    with torch.no_grad():
        r_ids, r_logits, r_conf = ref.forward_head(torch.tensor(h), torch.tensor(anchor))
    m_ids, m_logits, m_conf = mlx.forward_head(
        mx.array(h), mx.array(anchor.astype(np.int32)), mlx_head
    )
    mx.eval(m_ids, m_logits, m_conf)
    assert np.array_equal(np.array(m_ids), r_ids.numpy())
    assert _maxdiff(m_logits, r_logits.numpy()) <= 2e-3
    assert _maxdiff(m_conf, r_conf.numpy()) <= 2e-3


@needs_ref
def test_forward_spec_parity() -> None:
    from dspark_mlx.model.config import DSparkArgs
    from dspark_mlx.model.drafter import DSparkDrafter

    rng = np.random.default_rng(11)
    args = {**ATTN_ARGS, "temperature": 0.0, "n_mtp_layers": 2, "compress_ratios": [0, 0, 0, 0]}
    n_mtp, dim, vocab = args["n_mtp_layers"], args["dim"], args["vocab_size"]
    n_targets = len(args["dspark_target_layer_ids"])

    ref_args = REF.ModelArgs(**args)
    ref_blocks = [REF.DSparkBlock(args["n_layers"] + i, ref_args) for i in range(n_mtp)]
    drafter = DSparkDrafter(DSparkArgs.from_dict(args), max_seq_len=args["max_seq_len"])

    embed_w = (rng.standard_normal((vocab, dim)) * 0.1).astype(np.float32)
    head_w = (rng.standard_normal((vocab, dim)) * 0.1).astype(np.float32)
    ref_embed = REF.ParallelEmbedding(vocab, dim)
    ref_embed.weight.data = torch.tensor(embed_w)
    ref_head = REF.ParallelHead(vocab, dim)
    ref_head.weight.data = torch.tensor(head_w)
    for blk in ref_blocks:
        blk.embed, blk.head = ref_embed, ref_head
    drafter.embed.weight = mx.array(embed_w)
    drafter.head.weight = mx.array(head_w)
    for i in range(n_mtp):
        _copy_block(ref_blocks[i], drafter.blocks[i], args, rng)

    b, start_pos = 2, 5
    main_prompt = (rng.standard_normal((b, 5, dim * n_targets)) * 0.3).astype(np.float32)
    main_dec = (rng.standard_normal((b, 1, dim * n_targets)) * 0.3).astype(np.float32)
    anchor = rng.integers(0, vocab, size=(b,)).astype(np.int64)

    def ref_fs(anchor_t, main_t, sp):
        h, main_x = ref_blocks[0].forward_embed(main_t, anchor_t)
        for blk in ref_blocks:
            h = blk(h, sp, anchor_t, main_x)
        return None if sp == 0 else ref_blocks[-1].forward_head(h, anchor_t)

    with torch.no_grad():
        ref_fs(torch.tensor(anchor), torch.tensor(main_prompt), 0)
        r_ids, r_logits, r_conf = ref_fs(torch.tensor(anchor), torch.tensor(main_dec), start_pos)

    a_mx = mx.array(anchor.astype(np.int32))
    drafter.forward_spec(a_mx, mx.array(main_prompt), 0)
    m_ids, m_logits, m_conf = drafter.forward_spec(a_mx, mx.array(main_dec), start_pos)
    mx.eval(m_ids, m_logits, m_conf)
    assert np.array_equal(np.array(m_ids), r_ids.numpy())
    assert _maxdiff(m_logits, r_logits.numpy()) <= 5e-3
    assert _maxdiff(m_conf, r_conf.numpy()) <= 5e-3


def mx_linear(in_dim, out_dim, weight):
    import mlx.nn as nn

    lin = nn.Linear(in_dim, out_dim, bias=False)
    lin.weight = mx.array(weight)
    return lin
