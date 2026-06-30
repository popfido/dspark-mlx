# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""Mixture-of-Experts for the DSpark draft blocks (``inference/model.py`` Gate/Expert/MoE).

The DSpark mtp blocks sit at layer ids >= ``n_hash_layers``, so they use score routing
(``sqrtsoftplus`` gate + bias-shifted top-k), not hash routing — both modes are ported for
completeness.

The routed experts are stored as **stacked** ``[E, out, in]`` weights and dispatched **sparsely**
with ``mx.gather_mm`` over only the top-k routed experts/token (``mx.gather_qmm`` once
:meth:`MoE.quantize` is called — the real DeepSeek-V4 draft ships fp8/fp4 experts). This is the
``O(topk)`` path validated in ``_repro/moe_gather_mm_sim.py`` at the real draft dims: parity-exact
vs the naive ``O(num_experts)`` dense combine, and 6.5–82× faster (30× at the M=5 draft block;
q8 is faster still + 1.9× smaller, q4 3.6× smaller). The DeepSeek-V4 base can't run locally, so
the path is parity-tested (``tests/test_moe_gather_mm.py``), not e2e-tested. The shared expert is
always-on and runs dense.
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


def _swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    """DSpark asymmetric clamp (gate: max only; up: both sides) then silu(gate) * up, in fp32."""
    gate = gate.astype(mx.float32)
    up = up.astype(mx.float32)
    if limit > 0:
        up = mx.clip(up, -limit, limit)
        gate = mx.minimum(gate, limit)
    return nn.silu(gate) * up


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
    """SwiGLU FFN with the reference's asymmetric clamp (gate: max only; up: both sides).

    Used for the always-on shared expert; the routed experts live stacked inside :class:`MoE`.
    """

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        h = _swiglu(self.w1(x), self.w3(x), self.swiglu_limit)
        return self.w2(h.astype(x.dtype))


class MoE(nn.Module):
    """Top-k routed experts (stacked, gather_mm/gather_qmm sparse dispatch) + a shared expert.

    Routed-expert weights are stacked ``[E, out, in]`` (nn.Linear layout): ``w1``/``w3`` are
    ``[E, inter, dim]`` (gate/up), ``w2`` is ``[E, dim, inter]`` (down). The loader stacks the
    checkpoint's per-expert ``ffn.experts.{e}.{w1,w2,w3}.weight`` keys into these. After
    :meth:`quantize`, each ``wN`` is replaced by ``wN_q``/``wN_scales``/``wN_biases`` and the
    forward switches to ``gather_qmm``.
    """

    def __init__(self, args: DSparkArgs, layer_id: int):
        super().__init__()
        self.dim = args.dim
        self.inter_dim = args.moe_inter_dim
        self.n_routed_experts = args.n_routed_experts
        self.swiglu_limit = args.swiglu_limit
        self.gate = Gate(args, layer_id)
        E, dim, inter = args.n_routed_experts, args.dim, args.moe_inter_dim
        self.w1 = mx.zeros((E, inter, dim), dtype=mx.float32)  # gate_proj
        self.w3 = mx.zeros((E, inter, dim), dtype=mx.float32)  # up_proj
        self.w2 = mx.zeros((E, dim, inter), dtype=mx.float32)  # down_proj
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)
        self.quant: Optional[Tuple[int, int]] = None  # (bits, group_size) once quantized

    def quantize(self, bits: int = 8, group_size: int = 64) -> None:
        """Quantize the stacked routed experts in place (gather_qmm path)."""
        for n in ("w1", "w3", "w2"):
            wq, scales, biases = mx.quantize(self[n], group_size=group_size, bits=bits)
            self[f"{n}_q"], self[f"{n}_scales"], self[f"{n}_biases"] = wq, scales, biases
            del self[n]
        self.quant = (bits, group_size)

    def _route_experts(self, xe: mx.array, indices: mx.array) -> mx.array:
        """Run the top-k routed experts for each token via (quantized) gather matmul.

        ``xe`` is ``[N, 1, 1, dim]``; ``indices`` is ``[N, topk]``. Returns ``[N, topk, dim]``.
        """
        if self.quant is not None:
            bits, gs = self.quant
            qkw = dict(rhs_indices=indices, transpose=True, group_size=gs, bits=bits)
            g = mx.gather_qmm(xe, self.w1_q, self.w1_scales, self.w1_biases, **qkw)
            u = mx.gather_qmm(xe, self.w3_q, self.w3_scales, self.w3_biases, **qkw)
            h = _swiglu(g, u, self.swiglu_limit).astype(xe.dtype)
            o = mx.gather_qmm(h, self.w2_q, self.w2_scales, self.w2_biases, **qkw)
        else:
            g = mx.gather_mm(xe, self.w1.swapaxes(-1, -2), rhs_indices=indices)
            u = mx.gather_mm(xe, self.w3.swapaxes(-1, -2), rhs_indices=indices)
            h = _swiglu(g, u, self.swiglu_limit).astype(xe.dtype)
            o = mx.gather_mm(h, self.w2.swapaxes(-1, -2), rhs_indices=indices)
        return o.squeeze(-2)  # [N, topk, dim]

    def __call__(self, x: mx.array, input_ids: Optional[mx.array]) -> mx.array:
        shape = x.shape
        in_dtype = x.dtype
        x = x.reshape(-1, self.dim)
        ids = input_ids.reshape(-1) if input_ids is not None else None
        weights, indices = self.gate(x, ids)  # [N, topk], [N, topk]

        o = self._route_experts(mx.expand_dims(x, (-2, -3)), indices)  # [N, topk, dim]
        y = mx.sum(o.astype(mx.float32) * mx.expand_dims(weights, -1), axis=-2)
        y = y + self.shared_experts(x).astype(mx.float32)
        return y.astype(in_dtype).reshape(shape)
