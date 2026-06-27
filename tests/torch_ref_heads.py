# Faithful CPU transcription of the DSpark head modules from
# deepseek-ai/DeepSeek-V4-Flash-DSpark inference/model.py:
#   DSparkMarkovHead     (L795-804)
#   DSparkConfidenceHead (L807-815)
#   ParallelEmbedding    (L89-111, world_size==1 path)
#   ParallelHead         (L719-740, world_size==1 path)
# Dependency-free (no kernel.py / CUDA) so it runs on CPU as a parity reference.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RefParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)


class _RefParallelHead(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, full_logits: bool = False) -> torch.Tensor:
        if not full_logits:
            x = x[:, -1]
        return F.linear(x.float(), self.weight)


class RefMarkovHead(nn.Module):
    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = _RefParallelEmbedding(vocab_size, rank)
        self.markov_w2 = _RefParallelHead(vocab_size, rank)

    def forward(self, token_ids: torch.Tensor):
        embed = self.markov_w1(token_ids)
        logits = self.markov_w2(embed, full_logits=True)
        return logits, embed


class RefConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # Reference uses Linear(input_dim, 1, dtype=fp32), bias=False.
        self.proj = nn.Linear(input_dim, 1, bias=False).to(torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(hidden.float()).squeeze(-1)
