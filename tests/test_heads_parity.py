# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
#
# Numeric parity of the MLX DSpark heads against a faithful CPU PyTorch reference,
# on tiny random weights. The reference draft stack uses CUDA-only kernels for
# attention / sinkhorn, but the Markov + Confidence heads are pure embedding/linear
# and therefore run on CPU, which is what we pin here.
from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch

from dspark_mlx.model.config import DSparkArgs
from dspark_mlx.model.heads import DSparkConfidenceHead, DSparkMarkovHead
from torch_ref_heads import RefConfidenceHead, RefMarkovHead

VOCAB, DIM, RANK, B, K = 512, 64, 16, 2, 5
TOL = 1e-4


def test_markov_head_parity() -> None:
    rng = np.random.default_rng(0)
    w1 = (rng.standard_normal((VOCAB, RANK)) * 0.02).astype(np.float32)
    w2 = (rng.standard_normal((VOCAB, RANK)) * 0.02).astype(np.float32)
    tok = rng.integers(0, VOCAB, size=(B,)).astype(np.int64)

    ref = RefMarkovHead(VOCAB, RANK)
    ref.markov_w1.weight.data = torch.tensor(w1)
    ref.markov_w2.weight.data = torch.tensor(w2)
    with torch.no_grad():
        r_logits, r_embed = ref(torch.tensor(tok))

    head = DSparkMarkovHead(VOCAB, RANK)
    head.markov_w1.weight = mx.array(w1)
    head.markov_w2.weight = mx.array(w2)
    m_logits, m_embed = head(mx.array(tok.astype(np.int32)))
    mx.eval(m_logits, m_embed)

    assert np.max(np.abs(np.array(m_embed) - r_embed.numpy())) <= TOL
    assert np.max(np.abs(np.array(m_logits) - r_logits.numpy())) <= TOL


def test_confidence_head_parity() -> None:
    rng = np.random.default_rng(1)
    wc = (rng.standard_normal((1, DIM + RANK)) * 0.02).astype(np.float32)
    hidden = (rng.standard_normal((B, K, DIM)) * 0.5).astype(np.float32)
    membed = (rng.standard_normal((B, K, RANK)) * 0.5).astype(np.float32)

    ref = RefConfidenceHead(DIM + RANK)
    ref.proj.weight.data = torch.tensor(wc)
    with torch.no_grad():
        r_conf = ref(torch.tensor(hidden), torch.tensor(membed))

    head = DSparkConfidenceHead(DIM + RANK)
    head.proj.weight = mx.array(wc)
    m_conf = head(mx.array(hidden), mx.array(membed))
    mx.eval(m_conf)

    assert np.max(np.abs(np.array(m_conf) - r_conf.numpy())) <= TOL


def test_config_from_dict_parses_dspark_fields() -> None:
    cfg = {
        "vocab_size": 129280,
        "dim": 4096,
        "n_layers": 43,
        "n_mtp_layers": 3,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [40, 41, 42],
        "dspark_markov_rank": 256,
        "an_unknown_base_model_key": 123,
    }
    args = DSparkArgs.from_dict(cfg)
    assert args.dspark_block_size == 5
    assert args.dspark_target_layer_ids == (40, 41, 42)
    assert args.dspark_markov_rank == 256
    assert args.confidence_in == 4096 + 256
    assert args.main_proj_in == 4096 * 3
