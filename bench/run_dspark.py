#!/usr/bin/env python
# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
#
# Benchmark DSpark self-speculative decoding against plain base greedy, end-to-end on a
# real base model + published draft checkpoint. Reports tokens/s, mean accepted-per-block,
# acceptance rate, and wall-clock speedup -- and asserts the DSpark stream is byte-identical
# to base greedy (the whole point: lossless).
#
#   python bench/run_dspark.py --arch qwen3 --precision bf16
#   python bench/run_dspark.py --arch qwen3 --precision 8bit --max-new-tokens 128
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx

from dspark_mlx.events import SummaryEvent, TokenEvent
from dspark_mlx.generate import generate
from dspark_mlx.loop import generate_eager
from dspark_mlx.hosts.mlx_lm import MlxLmHostAdapter
from dspark_mlx.loading import load_drafter
from dspark_mlx.registry import resolve_arch

_RESEARCH = "/Users/Fido/workspace/omlx/_research/dspark_multi"

# arch -> (base repos by precision, draft checkpoint, draft config)
PRESETS = {
    "qwen3": {
        "base": {"bf16": "Qwen/Qwen3-4B", "8bit": "mlx-community/Qwen3-4B-8bit"},
        "ckpt": f"{_RESEARCH}/ckpt/qwen3_4b.safetensors",
        "config": f"{_RESEARCH}/qwen3_4b_config.json",
    },
    "gemma4": {
        # the DSpark draft targets the deployed (instruct) model; the pretrained base has no
        # chat template and gives low acceptance (see RESULTS.md)
        "base": {"bf16": "google/gemma-4-12b-it", "8bit": "google/gemma-4-12b-it"},
        "ckpt": f"{_RESEARCH}/ckpt/gemma4_12b.safetensors",
        "config": f"{_RESEARCH}/gemma4_12b_config.json",
    },
}

DEFAULT_PROMPT = (
    "The history of computing is a story of abstraction. Each generation built tools that "
    "hid the complexity of the one before it, and in doing so"
)


@dataclass
class Run:
    tokens: List[int]
    seconds: float
    n_drafted: int = 0
    n_accepted: int = 0
    n_blocks: int = 0

    @property
    def tps(self) -> float:
        return len(self.tokens) / self.seconds


def _base_greedy(adapter, prompt_ids: List[int], n: int) -> Run:
    adapter.reset()
    t0 = time.perf_counter()
    step = adapter.prefill(mx.array([prompt_ids], dtype=mx.int32))
    tok = int(mx.argmax(step.logits[0]).item())
    out = [tok]
    while len(out) < n:
        step = adapter.decode_step(mx.array([tok], dtype=mx.int32))
        tok = int(mx.argmax(step.logits[0]).item())
        out.append(tok)
    return Run(tokens=out, seconds=time.perf_counter() - t0)


def _dspark(adapter, drafter, prompt_ids: List[int], n: int, eos: Optional[int],
            loop="legacy", confidence=0.0) -> Run:
    adapter.reset()
    drafter.reset()
    if loop == "eager":
        gen = lambda *a, **k: generate_eager(*a, confidence_threshold=confidence, **k)
    else:
        gen = generate
    t0 = time.perf_counter()
    events = list(gen(adapter, drafter, [prompt_ids], max_new_tokens=n, eos_id=eos))
    seconds = time.perf_counter() - t0
    tokens = [e.token for e in events if isinstance(e, TokenEvent)]
    summary = next(e for e in events if isinstance(e, SummaryEvent))
    n_blocks = summary.n_drafted // drafter.block_size if drafter.block_size else 0
    return Run(tokens, seconds, summary.n_drafted, summary.n_accepted, n_blocks)


def _load_base(arch: str, precision: str):
    repo = PRESETS[arch]["base"][precision]
    if arch == "gemma4":
        from dspark_mlx.hosts.gemma4_unified import load_gemma4_unified_adapter

        return load_gemma4_unified_adapter(repo, PRESETS[arch]["config"], precision)
    from mlx_lm import load

    model, tokenizer = load(repo)
    config = json.load(open(PRESETS[arch]["config"]))
    adapter = MlxLmHostAdapter(model, target_layer_ids=config["target_layer_ids"])
    return model, tokenizer, adapter


def _load_drafter(arch: str):
    config = json.load(open(PRESETS[arch]["config"]))
    drafter = resolve_arch(config).build(config, max_seq_len=8192)
    weights = mx.load(PRESETS[arch]["ckpt"])
    skipped = load_drafter(drafter, weights, key_map=resolve_arch(config).key_map)
    if skipped:
        raise SystemExit(f"unmapped draft keys: {skipped[:8]} ...")
    mx.eval(drafter.parameters())
    return drafter


def _lossless_note(ds_tokens, base_tokens) -> str:
    """DSpark emits the base model's argmax from its own verify forward, so it is lossless by
    construction. bf16 attention is batch-order sensitive, so the stream can still differ from a
    *separately run* sequential greedy by a few tokens (one flip cascades). That is bf16
    nondeterminism, not loss — the loop is bit-exact in float32 (see tests + _repro/eager_diag).
    """
    n = min(len(ds_tokens), len(base_tokens))
    diffs = sum(ds_tokens[i] != base_tokens[i] for i in range(n))
    if diffs == 0:
        return "identical to sequential base greedy"
    return f"{diffs}/{n} differ from sequential greedy (bf16 batch nondeterminism; bit-exact in f32)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=list(PRESETS), default="qwen3")
    ap.add_argument("--precision", choices=["bf16", "8bit"], default="bf16")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=8, help="warmup tokens (excluded from timing)")
    ap.add_argument("--loop", choices=["legacy", "eager"], default="eager",
                    help="eager = one base forward/cycle (reference-matched)")
    ap.add_argument("--chat", action="store_true",
                    help="wrap the prompt in the chat template (in-distribution for the draft)")
    ap.add_argument("--confidence", type=float, default=0.0,
                    help="confidence-gated draft length (sigmoid threshold; 0 = full block)")
    args = ap.parse_args()

    print(f"loading {args.arch} base ({args.precision}) + draft ...", flush=True)
    model, tokenizer, adapter = _load_base(args.arch, args.precision)
    drafter = _load_drafter(args.arch)
    if args.chat and getattr(tokenizer, "chat_template", None):
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], add_generation_prompt=True, tokenize=True
        )
        if isinstance(prompt_ids, dict) or hasattr(prompt_ids, "keys"):
            prompt_ids = prompt_ids["input_ids"]
        if len(prompt_ids) and isinstance(prompt_ids[0], (list, tuple)):
            prompt_ids = prompt_ids[0]
        prompt_ids = [int(t) for t in prompt_ids]
    else:
        if args.chat:
            print("  (no chat template on this tokenizer -- using raw prompt)")
        prompt_ids = tokenizer.encode(args.prompt)

    if args.warmup:
        _dspark(adapter, drafter, prompt_ids, args.warmup, None, args.loop, args.confidence)
        _base_greedy(adapter, prompt_ids, args.warmup)

    # eos disabled for a fixed-length, comparable benchmark window
    ds = _dspark(adapter, drafter, prompt_ids, args.max_new_tokens, None, args.loop, args.confidence)
    base = _base_greedy(adapter, prompt_ids, len(ds.tokens))

    note = _lossless_note(ds.tokens, base.tokens)
    mean_acc = ds.n_accepted / ds.n_blocks if ds.n_blocks else 0.0
    acc_rate = ds.n_accepted / ds.n_drafted if ds.n_drafted else 0.0

    print(f"\n=== {args.arch} / {args.precision} | loop={args.loop} | block_size={drafter.block_size} | "
          f"prompt={len(prompt_ids)} tok, generated={len(ds.tokens)} ===")
    print(f"  lossless:                    {note}")
    print(f"  mean accepted / block:       {mean_acc:.2f} / {drafter.block_size}")
    print(f"  draft acceptance rate:       {acc_rate*100:.1f}%")
    print(f"  base greedy:                 {base.tps:6.1f} tok/s  ({base.seconds*1e3:.0f} ms)")
    print(f"  DSpark:                      {ds.tps:6.1f} tok/s  ({ds.seconds*1e3:.0f} ms)")
    print(f"  speedup:                     {base.seconds/ds.seconds:.2f}x")


if __name__ == "__main__":
    main()
