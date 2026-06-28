# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
#
# The architecture seam: resolve_arch dispatches by model_type, builds a conforming
# DraftBackbone, and exposes the per-arch checkpoint key map.
from __future__ import annotations

import pytest

from dspark_mlx.arch.backbone import DraftBackbone
from dspark_mlx.registry import ARCH_REGISTRY, resolve_arch

_V4_CFG = dict(
    model_type="deepseek_v4", vocab_size=32, dim=16, n_heads=4, head_dim=16,
    rope_head_dim=8, q_lora_rank=16, o_lora_rank=8, o_groups=2, window_size=8,
    n_layers=2, n_mtp_layers=1, n_routed_experts=4, moe_inter_dim=16,
    n_activated_experts=2, n_hash_layers=0, hc_mult=4, hc_sinkhorn_iters=20,
    dspark_block_size=4, dspark_noise_token_id=31, dspark_target_layer_ids=[0, 1],
    dspark_markov_rank=8,
)


def test_registry_lists_deepseek_v4() -> None:
    assert any(a.name == "deepseek_v4" for a in ARCH_REGISTRY)


def test_resolve_and_build_deepseek_v4() -> None:
    arch = resolve_arch({"model_type": "deepseek_v4"})
    assert arch.name == "deepseek_v4"
    drafter = arch.build(_V4_CFG, max_seq_len=64)
    assert hasattr(drafter, "forward_spec") and hasattr(drafter, "advance")
    assert isinstance(drafter, DraftBackbone)


def test_deepseek_v4_key_map() -> None:
    arch = resolve_arch({"model_type": "deepseek_v4"})
    assert arch.key_map("mtp.0.attn.wq_a.weight") == "blocks.0.attn.wq_a.weight"
    assert arch.key_map("layers.0.attn.wkv.weight") is None  # base-model key


def test_unknown_model_type_raises() -> None:
    with pytest.raises(ValueError, match="No DSpark backbone"):
        resolve_arch({"model_type": "llama_not_supported"})
