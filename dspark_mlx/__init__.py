# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

__version__ = "0.0.1"

from .adapter import BaseModelAdapter, BlockOut, StepOut
from .events import SummaryEvent, TokenEvent
from .generate import generate
from .loading import is_dspark_checkpoint, load_drafter, map_checkpoint_key
from .model.config import DSparkArgs
from .model.drafter import DSparkDrafter
from .verify import AcceptResult, greedy_accept, speculative_sample_accept

__all__ = [
    "BaseModelAdapter",
    "BlockOut",
    "StepOut",
    "DSparkArgs",
    "DSparkDrafter",
    "generate",
    "greedy_accept",
    "speculative_sample_accept",
    "AcceptResult",
    "TokenEvent",
    "SummaryEvent",
    "load_drafter",
    "map_checkpoint_key",
    "is_dspark_checkpoint",
]
