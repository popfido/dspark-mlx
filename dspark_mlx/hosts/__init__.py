# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash/Pro-DSpark, deepseek-ai/DeepSpec)

"""Reference host integrations: concrete :class:`~dspark_mlx.adapter.BaseModelAdapter`s.

dspark-mlx is target-agnostic — the host owns the base model. These are the reference
hosts used for testing and benchmarking (and as a template for downstream hosts such as
the omlx ``DSparkEngine``).
"""

from .mlx_lm import HostHooks, MlxLmHostAdapter, build_mlx_lm_hooks

__all__ = ["MlxLmHostAdapter", "HostHooks", "build_mlx_lm_hooks"]
