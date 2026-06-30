# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""Mixture-of-Experts for the DSpark draft blocks (``inference/model.py`` Gate/Expert/MoE).

The DSpark mtp blocks sit at layer ids >= ``n_hash_layers``, so they use score routing
(``sqrtsoftplus`` gate + bias-shifted top-k), not hash routing — both modes are ported for
completeness. The expert combine here is dense (every expert evaluated, then masked by the
routing weight): numerically exact and simple, but O(num_experts).

A sparse ``gather_mm`` dispatch (only the top-k routed experts/token) is validated in
``_repro/moe_gather_mm_sim.py`` at the real DeepSeek-V4 draft dims — **parity-exact** and
**6.5–82× faster** (30× at the M=5 draft block; the win shrinks with M as the dense matmuls
get GPU-efficient). Wiring it in needs stacked ``[E, …]`` expert weights (the checkpoint stores
per-expert ``ffn.experts.{e}.*`` keys), so it is a storage + weight-loader change, not a local
edit here; the DeepSeek-V4 draft can't be run locally, so it is parity-tested, not e2e-tested.
"""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import DSparkArgs


def _sqrtsoftplus(x: mx.array) -> mx.array:
    # sqrt(softplus(x)); stable softplus = max(x,0) + log1p(exp(-|x|))
    softplus = mx.maximum(x, 0.0) + mx.log1p(mx.exp(-mx.abs(x)))
    return mx.sqrt(softplus)


class Gate(nn.Module):
    """Expert routing. ``layer_id < n_hash_layers`` -> hash routing, else score routing."""

    def __init__(self, args: DSparkArgs, layer_id: int):
        super().__init__()
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = mx.zeros((args.n_routed_experts, args.dim), dtype=mx.float32)
        if self.hash:
            self.tid2eid = mx.zeros((args.vocab_size, args.n_activated_experts), dtype=mx.int32)
        else:
            self.bias = mx.zeros((args.n_routed_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        scores = mx.matmul(x.astype(mx.float32), self.weight.T)
        if self.score_func == "softmax":
            scores = mx.softmax(scores, axis=-1)
        elif self.score_func == "sigmoid":
            scores = mx.sigmoid(scores)
        else:
            scores = _sqrtsoftplus(scores)
        original = scores
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            # bias shifts selection only; weights come from the unbiased scores.
            indices = mx.argsort(scores + self.bias, axis=-1)[..., -self.topk :]
        weights = mx.take_along_axis(original, indices, axis=-1)
        if self.score_func != "softmax":
            weights = weights / mx.sum(weights, axis=-1, keepdims=True)
        return weights * self.route_scale, indices


class Expert(nn.Module):
    """SwiGLU FFN with the reference's asymmetric clamp (gate: max only; up: both sides)."""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.w1(x).astype(mx.float32)
        up = self.w3(x).astype(mx.float32)
        if self.swiglu_limit > 0:
            up = mx.clip(up, -self.swiglu_limit, self.swiglu_limit)
            gate = mx.minimum(gate, self.swiglu_limit)
        h = nn.silu(gate) * up
        return self.w2(h.astype(x.dtype))


class MoE(nn.Module):
    """Top-k routed experts + an always-on shared expert."""

    def __init__(self, args: DSparkArgs, layer_id: int):
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.gate = Gate(args, layer_id)
        self.experts = [
            Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)
            for _ in range(args.n_routed_experts)
        ]
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)

    def __call__(self, x: mx.array, input_ids: Optional[mx.array]) -> mx.array:
        shape = x.shape
        in_dtype = x.dtype
        x = x.reshape(-1, self.dim)
        ids = input_ids.reshape(-1) if input_ids is not None else None
        weights, indices = self.gate(x, ids)  # [N, topk], [N, topk]

        # w2 is bias-free linear, so weight * expert(x) == applying the weight inside the
        # expert; accumulate each expert masked by its per-token routing weight.
        y = mx.zeros(x.shape, dtype=mx.float32)
        for e in range(self.n_routed_experts):
            w_e = mx.sum(mx.where(indices == e, weights, 0.0), axis=-1, keepdims=True)
            y = y + self.experts[e](x).astype(mx.float32) * w_e
        y = y + self.shared_experts(x).astype(mx.float32)
        return y.astype(in_dtype).reshape(shape)
