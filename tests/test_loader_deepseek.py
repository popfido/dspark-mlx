# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
"""Loader wiring for the native DeepSeek-V4-{Flash,Pro}-DSpark checkpoints: name presets,
shard-selective draft download, inference/config.json reading, and the host guard. No network."""

from __future__ import annotations

import json

import pytest

from dspark_mlx.loader import (
    KNOWN_MODELS,
    _deepseek_config,
    _draft_shards,
    _is_deepseek_checkpoint,
    load_host,
    resolve_model,
)


def test_deepseek_presets_present():
    for name in ("deepseek-v4-flash", "deepseek-v4-pro"):
        assert name in KNOWN_MODELS
        draft, base = resolve_model(name)
        # the draft (mtp.*) is bundled inside the base checkpoint -> same repo
        assert draft == base
        assert "DeepSeek-V4" in draft


def test_draft_shards_selects_only_mtp_and_shared():
    wm = {
        "embed.weight": "s1", "head.weight": "s1",
        "layers.0.attn.wq_a.weight": "s2",       # base layer -> excluded
        "layers.30.ffn.experts.3.w1.weight": "s3",  # base MoE -> excluded
        "mtp.0.attn.wq_a.weight": "s9",
        "mtp.0.ffn.experts.5.w1.weight": "s9",
        "mtp.2.markov_head.markov_w1.weight": "s10",
    }
    keep, shards = _draft_shards(wm)
    assert keep == {
        "embed.weight", "head.weight", "mtp.0.attn.wq_a.weight",
        "mtp.0.ffn.experts.5.w1.weight", "mtp.2.markov_head.markov_w1.weight",
    }
    assert shards == ["s1", "s10", "s9"]  # sorted unique; base shards s2/s3 dropped
    assert "s2" not in shards and "s3" not in shards


def test_deepseek_config_injects_model_type(tmp_path):
    inf = tmp_path / "inference"
    inf.mkdir()
    (inf / "config.json").write_text(json.dumps({"dim": 7168, "n_layers": 61, "n_routed_experts": 384}))
    assert _is_deepseek_checkpoint(str(tmp_path))
    cfg = _deepseek_config(str(tmp_path))
    assert cfg["model_type"] == "deepseek_v4"  # injected; inference/config.json omits it
    assert cfg["dim"] == 7168 and cfg["n_routed_experts"] == 384


def test_is_deepseek_false_for_standalone(tmp_path):
    (tmp_path / "config.json").write_text("{}")  # standalone qwen3/gemma4 draft repo
    assert not _is_deepseek_checkpoint(str(tmp_path))


def test_load_host_rejects_deepseek():
    with pytest.raises(NotImplementedError, match="DeepSeek-V4 host"):
        load_host("deepseek-ai/DeepSeek-V4-Pro-DSpark", {"model_type": "deepseek_v4"})
