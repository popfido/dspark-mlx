# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""Stream events emitted by the generate loop (mirrors dflash-mlx's event surface)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenEvent:
    token: int
    n_accepted: int  # draft tokens accepted in the block that produced this token


@dataclass
class SummaryEvent:
    n_emitted: int
    n_drafted: int
    n_accepted: int

    @property
    def acceptance_rate(self) -> float:
        return self.n_accepted / self.n_drafted if self.n_drafted else 0.0
