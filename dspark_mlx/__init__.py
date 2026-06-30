# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash/Pro-DSpark, deepseek-ai/DeepSpec)

__version__ = "0.1.0"

from .adapter import BaseModelAdapter, BlockOut, StepOut
from .arch.backbone import DraftArch, DraftBackbone
from .events import SummaryEvent, TokenEvent
from .generate import generate
from .loader import KNOWN_MODELS, load_draft, load_host, resolve_model
from .loading import is_dspark_checkpoint, load_drafter, map_checkpoint_key
from .loop import generate_eager
from .model.config import DSparkArgs
from .model.drafter import DSparkDrafter
from .quant import quantize_drafter
from .registry import ARCH_REGISTRY, resolve_arch
from .verify import AcceptResult, greedy_accept, speculative_sample_accept

__all__ = [
    "BaseModelAdapter",
    "BlockOut",
    "StepOut",
    "DSparkArgs",
    "DSparkDrafter",
    "DraftArch",
    "DraftBackbone",
    "resolve_arch",
    "ARCH_REGISTRY",
    "generate",
    "generate_eager",
    "load_draft",
    "load_host",
    "resolve_model",
    "KNOWN_MODELS",
    "greedy_accept",
    "speculative_sample_accept",
    "AcceptResult",
    "TokenEvent",
    "SummaryEvent",
    "load_drafter",
    "map_checkpoint_key",
    "is_dspark_checkpoint",
    "quantize_drafter",
]
