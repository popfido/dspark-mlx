# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash/Pro-DSpark, deepseek-ai/DeepSpec)

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple


@dataclass
class DSparkArgs:
    """DSpark-relevant hyperparameters.

    Field names follow the reference ``inference/config.json`` (``dim``/``n_layers``
    style, not the HF root-config ``hidden_size``/``num_hidden_layers`` style).
    ``from_dict`` tolerates and drops unknown keys so the full base-model config can
    be passed in verbatim.
    """

    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 2048
    n_layers: int = 43
    n_mtp_layers: int = 3
    n_hash_layers: int = 3
    n_heads: int = 64
    # moe
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: str = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    # attention
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    norm_eps: float = 1e-6
    # hyper-connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # rope / yarn
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    original_seq_len: int = 65536
    beta_fast: int = 32
    beta_slow: int = 1
    compress_rope_theta: float = 160000.0
    # dspark
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: Tuple[int, ...] = (40, 41, 42)
    dspark_markov_rank: int = 256
    temperature: float = 1.0

    @property
    def main_proj_in(self) -> int:
        """Input width of the stage-0 ``main_proj`` (concatenated target hiddens)."""
        return self.dim * len(self.dspark_target_layer_ids)

    @property
    def confidence_in(self) -> int:
        """Input width of the confidence head: ``[hidden ‖ markov_embed]``."""
        return self.dim + self.dspark_markov_rank

    @classmethod
    def from_dict(cls, params: Mapping[str, Any]) -> "DSparkArgs":
        names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in params.items() if k in names}
        if "dspark_target_layer_ids" in kwargs:
            kwargs["dspark_target_layer_ids"] = tuple(kwargs["dspark_target_layer_ids"])
        return cls(**kwargs)
