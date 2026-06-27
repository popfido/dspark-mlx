# Loads the real DeepSeek reference inference/model.py for parity testing, with the
# CUDA/tilelang kernels replaced by the CPU stubs in cpu_kernels.py and weights forced to
# fp32. Returns None (tests skip) when the reference checkpoint code isn't present locally.
#
# Point it at a checkout via DSPARK_REF_DIR; otherwise it falls back to the known dev path.
from __future__ import annotations

import importlib.util
import os
import sys

_CANDIDATES = [
    os.environ.get("DSPARK_REF_DIR"),
    os.path.join(os.path.dirname(__file__), "_dspark_ref"),
    "/Users/Fido/workspace/omlx/_research/dspark_ref",
]

_cached = "unset"


def load_reference_model():
    global _cached
    if _cached != "unset":
        return _cached
    _cached = _load()
    return _cached


def _load():
    try:
        import torch
    except Exception:
        return None

    ref_dir = next(
        (c for c in _CANDIDATES if c and os.path.isfile(os.path.join(c, "model.py"))),
        None,
    )
    if ref_dir is None:
        return None

    import cpu_kernels  # the kernel stub model.py imports from

    sys.modules["kernel"] = cpu_kernels
    if ref_dir not in sys.path:
        sys.path.insert(0, ref_dir)

    spec = importlib.util.spec_from_file_location(
        "dspark_ref_model", os.path.join(ref_dir, "model.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Force the single-rank fp32 regime (drops the fp8/fp4 QAT round-trip).
    mod.world_size = 1
    mod.rank = 0
    mod.default_dtype = torch.float32
    mod.scale_fmt = None
    mod.scale_dtype = torch.float32
    return mod
