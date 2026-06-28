# Faithful CPU torch reference for the Gemma4 DSpark backbone, transcribed from
# deepseek-ai/DeepSpec dspark/gemma4/modeling.py using the real transformers Gemma4
# primitives (Gemma4RMSNorm, rotate_half). Attribute names match the MLX port.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm, rotate_half


def _rope(x, cos, sin):  # x: [b, heads, seq, hd]; cos/sin: [seq, hd]
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


class RefAttn(nn.Module):
    def __init__(self, h, nh, nkv, hd, eps, k_eq_v):
        super().__init__()
        self.nh, self.nkv, self.hd, self.k_eq_v = nh, nkv, hd, k_eq_v
        self.q_proj = nn.Linear(h, nh * hd, bias=False)
        self.k_proj = nn.Linear(h, nkv * hd, bias=False)
        self.v_proj = None if k_eq_v else nn.Linear(h, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, h, bias=False)
        self.q_norm = Gemma4RMSNorm(hd, eps)
        self.k_norm = Gemma4RMSNorm(hd, eps)
        self.v_norm = Gemma4RMSNorm(hd, eps, with_scale=False)

    def forward(self, hidden, target_ctx, cos, sin):
        b, q = hidden.shape[:2]
        ctx = target_ctx.shape[1]
        qh = self.q_norm(self.q_proj(hidden).view(b, q, self.nh, self.hd)).transpose(1, 2)
        k_ctx, k_noise = self.k_proj(target_ctx), self.k_proj(hidden)
        v_ctx, v_noise = (k_ctx, k_noise) if self.k_eq_v else (self.v_proj(target_ctx), self.v_proj(hidden))
        k = self.k_norm(torch.cat([k_ctx, k_noise], 1).view(b, ctx + q, self.nkv, self.hd)).transpose(1, 2)
        v = self.v_norm(torch.cat([v_ctx, v_noise], 1).view(b, ctx + q, self.nkv, self.hd)).transpose(1, 2)
        qh = qh * cos[None, None, -q:] + rotate_half(qh) * sin[None, None, -q:]
        k = _rope(k, cos, sin)
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, 1)
        v = v.repeat_interleave(rep, 1)
        attn = F.scaled_dot_product_attention(qh, k, v, scale=1.0)
        return self.o_proj(attn.transpose(1, 2).reshape(b, q, self.nh * self.hd))


class RefMLP(nn.Module):
    def __init__(self, h, inter):
        super().__init__()
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class RefLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        h, eps = args.hidden_size, args.rms_norm_eps
        self.self_attn = RefAttn(h, args.num_attention_heads, args.num_key_value_heads, args.head_dim, eps, args.attention_k_eq_v)
        self.mlp = RefMLP(h, args.intermediate_size)
        self.input_layernorm = Gemma4RMSNorm(h, eps)
        self.post_attention_layernorm = Gemma4RMSNorm(h, eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(h, eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(h, eps)
        self.layer_scalar = nn.Parameter(torch.ones(1))

    def forward(self, hidden, target_ctx, cos, sin):
        h = self.post_attention_layernorm(self.self_attn(self.input_layernorm(hidden), target_ctx, cos, sin))
        hidden = hidden + h
        h = self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(hidden)))
        hidden = hidden + h
        return hidden * self.layer_scalar


class RefGemma4Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc = nn.Linear(args.fc_in, args.hidden_size, bias=False)
        self.hidden_norm = Gemma4RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.layers = nn.ModuleList([RefLayer(args) for _ in range(args.num_hidden_layers)])
        self.norm = Gemma4RMSNorm(args.hidden_size, args.rms_norm_eps)

    def project_context(self, target_hidden):
        return self.hidden_norm(self.fc(target_hidden))

    def forward(self, noise, target_ctx, cos, sin):
        h = noise
        for layer in self.layers:
            h = layer(h, target_ctx, cos, sin)
        return self.norm(h)
