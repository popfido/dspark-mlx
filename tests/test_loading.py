# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
#
# Weight loading: the mtp.* <-> blocks.N.* key mapping, DSpark detection, a full
# round-trip into a fresh drafter (no reference needed), and the dequant block math.
from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from dspark_mlx.loading import (
    dequant_fp4_groupwise,
    dequant_fp8_blockwise,
    drafter_path_to_checkpoint_key,
    is_dspark_checkpoint,
    load_drafter,
    map_checkpoint_key,
)
from dspark_mlx.model.config import DSparkArgs
from dspark_mlx.model.drafter import DSparkDrafter

ARGS = dict(
    vocab_size=32, dim=16, n_heads=4, head_dim=16, rope_head_dim=8, q_lora_rank=16,
    o_lora_rank=8, o_groups=2, window_size=8, norm_eps=1e-6, rope_theta=10000.0, n_layers=2,
    n_mtp_layers=2, n_routed_experts=4, moe_inter_dim=16, n_activated_experts=2,
    n_shared_experts=1, n_hash_layers=0, score_func="sqrtsoftplus", route_scale=1.5,
    swiglu_limit=10.0, hc_mult=4, hc_sinkhorn_iters=20, dspark_block_size=4,
    dspark_noise_token_id=31, dspark_target_layer_ids=[0, 1], dspark_markov_rank=8,
)


def test_map_checkpoint_key() -> None:
    assert map_checkpoint_key("mtp.0.attn.wq_a.weight") == "blocks.0.attn.wq_a.weight"
    assert map_checkpoint_key("mtp.2.markov_head.markov_w1.weight") == "blocks.2.markov_head.markov_w1.weight"
    assert map_checkpoint_key("embed.weight") == "embed.weight"
    assert map_checkpoint_key("head.weight") == "head.weight"
    assert map_checkpoint_key("layers.0.attn.wkv.weight") is None  # base-model key
    assert drafter_path_to_checkpoint_key("blocks.1.ffn.experts.3.w1.weight") == "mtp.1.ffn.experts.3.w1.weight"


def test_is_dspark_checkpoint() -> None:
    assert is_dspark_checkpoint(["mtp.0.main_proj.weight", "mtp.0.attn.wkv.weight"])
    assert is_dspark_checkpoint(["mtp.2.confidence_head.proj.weight"])
    assert is_dspark_checkpoint(["mtp.0.attn.wkv.weight"], config={"dspark_block_size": 5})
    # plain MTP (no DSpark markers, no config flag) -> not DSpark
    assert not is_dspark_checkpoint(["mtp.0.attn.wkv.weight", "mtp.0.ffn.gate.weight"])


def _drafter_state(drafter):
    return {
        p: v for p, v in tree_flatten(drafter.parameters())
        if "_cos" not in p and "_sin" not in p
    }


def test_load_drafter_roundtrip() -> None:
    args = DSparkArgs.from_dict(ARGS)
    src = DSparkDrafter(args, max_seq_len=64)
    dst = DSparkDrafter(args, max_seq_len=64)  # fresh, different random init

    # Serialize src as a checkpoint (mtp.* keys, no rope tables) and load into dst.
    weights = {drafter_path_to_checkpoint_key(p): v for p, v in _drafter_state(src).items()}
    weights["base.junk.weight"] = mx.zeros((2, 2))  # a base-model key the loader must skip
    skipped = load_drafter(dst, weights)
    assert skipped == ["base.junk.weight"]

    src_state, dst_state = _drafter_state(src), _drafter_state(dst)
    assert set(src_state) == set(dst_state)
    for path in src_state:
        assert np.array_equal(np.array(src_state[path]), np.array(dst_state[path])), path


def test_dequant_fp8_blockwise() -> None:
    w = np.arange(16, dtype=np.float32).reshape(4, 4)
    scale = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)  # 2x2 blocks of size 2
    out = dequant_fp8_blockwise(w, scale, block=2)
    expected = w.copy()
    expected[0:2, 0:2] *= 2
    expected[0:2, 2:4] *= 3
    expected[2:4, 0:2] *= 4
    expected[2:4, 2:4] *= 5
    assert np.array_equal(out, expected)


def test_dequant_fp4_groupwise() -> None:
    w = np.arange(16, dtype=np.float32).reshape(2, 8)
    scale = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)  # group=4 along K
    out = dequant_fp4_groupwise(w, scale, group=4)
    expected = w.copy()
    expected[0, 0:4] *= 2
    expected[0, 4:8] *= 3
    expected[1, 0:4] *= 4
    expected[1, 4:8] *= 5
    assert np.array_equal(out, expected)
