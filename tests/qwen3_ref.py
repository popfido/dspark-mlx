# Faithful CPU torch reference for the Qwen3 DSpark backbone, transcribed from
# deepseek-ai/DeepSpec deepspec/modeling/dspark/qwen3/modeling.py using the real
# transformers Qwen3 primitives (Qwen3RMSNorm, rotate_half). Module attribute names match
# the MLX port so a single weight dict loads into both.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, rotate_half


def _apply_rope(x, cos, sin):  # x: [b, heads, seq, hd]; cos/sin: [seq, hd]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class RefAttn(nn.Module):
    def __init__(self, h, nh, nkv, hd, eps):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.scale = hd ** -0.5
        self.q_proj = nn.Linear(h, nh * hd, bias=False)
        self.k_proj = nn.Linear(h, nkv * hd, bias=False)
        self.v_proj = nn.Linear(h, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, h, bias=False)
        self.q_norm = Qwen3RMSNorm(hd, eps)
        self.k_norm = Qwen3RMSNorm(hd, eps)

    def forward(self, hidden, target_ctx, cos, sin):
        b, q = hidden.shape[:2]
        ctx = target_ctx.shape[1]
        qh = self.q_norm(self.q_proj(hidden).view(b, q, self.nh, self.hd)).transpose(1, 2)
        k = torch.cat([self.k_proj(target_ctx), self.k_proj(hidden)], dim=1).view(b, ctx + q, self.nkv, self.hd)
        v = torch.cat([self.v_proj(target_ctx), self.v_proj(hidden)], dim=1).view(b, ctx + q, self.nkv, self.hd)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        qh = qh * cos[None, None, -q:, :] + rotate_half(qh) * sin[None, None, -q:, :]
        k = _apply_rope(k, cos, sin)
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        attn = F.scaled_dot_product_attention(qh, k, v, scale=self.scale)
        attn = attn.transpose(1, 2).reshape(b, q, self.nh * self.hd)
        return self.o_proj(attn)


class RefMLP(nn.Module):
    def __init__(self, h, inter):
        super().__init__()
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RefLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.self_attn = RefAttn(
            args.hidden_size, args.num_attention_heads, args.num_key_value_heads,
            args.head_dim, args.rms_norm_eps,
        )
        self.mlp = RefMLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = Qwen3RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(args.hidden_size, args.rms_norm_eps)

    def forward(self, hidden, target_ctx, cos, sin):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), target_ctx, cos, sin)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class RefQwen3Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc = nn.Linear(args.fc_in, args.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.layers = nn.ModuleList([RefLayer(args) for _ in range(args.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(args.hidden_size, args.rms_norm_eps)

    def project_context(self, target_hidden):
        return self.hidden_norm(self.fc(target_hidden))

    def forward(self, noise, target_ctx, cos, sin):
        h = noise
        for layer in self.layers:
            h = layer(h, target_ctx, cos, sin)
        return self.norm(h)
