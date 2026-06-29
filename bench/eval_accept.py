#!/usr/bin/env python
# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
#
# Acceptance-length eval: the correctness/quality check that maps onto the DSpark paper.
# Speedup is hardware-bound (paper = datacenter GPU, us = Metal), but *average accepted
# length* (tokens per verify step) is hardware-independent and is what the paper reports.
# Runs DSpark greedy over a slice of an eval dataset and reports macro/micro accepted length.
#
#   python bench/eval_accept.py --arch qwen3 --dataset gsm8k --chat --n 20
#   python bench/eval_accept.py --arch qwen3 --dataset mbpp  --chat --n 20
from __future__ import annotations

import argparse
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_dspark import PRESETS, _load_base, _load_drafter  # noqa: E402

from dspark_mlx.events import SummaryEvent  # noqa: E402
from dspark_mlx.loop import generate_eager  # noqa: E402


def _gsm8k(n):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [ex["question"] for ex in ds.select(range(n))]


def _math500(n):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [ex["problem"] for ex in ds.select(range(n))]


def _mbpp(n):
    from datasets import load_dataset

    ds = load_dataset("mbpp", split="test")
    return [f"{ex['text']}\nYour code should pass:\n{chr(10).join(ex['test_list'])}"
            for ex in ds.select(range(n))]


def _humaneval(n):
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")
    return ["Complete the following Python function:\n\n" + ex["prompt"] for ex in ds.select(range(n))]


def _mtbench(n):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")  # 80 multi-turn prompts
    return [ex["prompt"][0] for ex in ds.select(range(min(n, len(ds))))]  # first turn only


def _alpaca(n):
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rows = []
    for ex in ds.select(range(n)):
        instr = ex["instruction"]
        rows.append(f"{instr}\n\n{ex['input']}" if ex["input"] else instr)
    return rows


# math: gsm8k (grade-school), math500 (competition) · code: humaneval, mbpp · chat: mtbench, alpaca
DATASETS = {"gsm8k": _gsm8k, "math500": _math500, "humaneval": _humaneval, "mbpp": _mbpp,
            "mtbench": _mtbench, "alpaca": _alpaca}


def _ids_from(out):
    """apply_chat_template may return a flat list, a [[...]] batch, or a dict/BatchEncoding."""
    if isinstance(out, dict) or hasattr(out, "keys"):
        out = out["input_ids"]
    if len(out) and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(t) for t in out]


def _encode(tokenizer, text, chat, think=True):
    if chat and getattr(tokenizer, "chat_template", None):
        kw = {} if think else {"enable_thinking": False}
        return _ids_from(tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True, **kw
        ))
    return tokenizer.encode(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=list(PRESETS), default="qwen3")
    ap.add_argument("--precision", choices=["bf16", "8bit"], default="bf16")
    ap.add_argument("--dataset", choices=list(DATASETS), default="gsm8k")
    ap.add_argument("--n", type=int, default=20, help="number of samples")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--no-think", action="store_true", help="disable Qwen3 thinking mode")
    ap.add_argument("--quant-draft", type=int, default=0, choices=[0, 8, 4],
                    help="quantize the draft to N bits (8 = acceptance-lossless)")
    args = ap.parse_args()

    print(f"loading {args.arch}/{args.precision} + draft ...", flush=True)
    model, tokenizer, adapter = _load_base(args.arch, args.precision)
    drafter = _load_drafter(args.arch, args.quant_draft)
    eos = getattr(tokenizer, "eos_token_id", None)
    block = drafter.block_size
    prompts = DATASETS[args.dataset](args.n)
    chat = args.chat and bool(getattr(tokenizer, "chat_template", None))
    if args.chat and not chat:
        print("  (tokenizer has no chat template -- evaluating raw prompts)")

    taus, prompt_lens = [], []
    tot_acc = tot_blocks = tot_emit = 0
    t0 = time.perf_counter()
    for i, q in enumerate(prompts):
        ids = _encode(tokenizer, q, chat, think=not args.no_think)
        adapter.reset()
        drafter.reset()
        events = list(generate_eager(adapter, drafter, [ids], max_new_tokens=args.max_new_tokens, eos_id=eos))
        s = next(e for e in events if isinstance(e, SummaryEvent))
        blocks = s.n_drafted // block
        if blocks:
            taus.append(s.n_accepted / blocks + 1.0)  # tokens emitted per verify step
        tot_acc += s.n_accepted
        tot_blocks += blocks
        tot_emit += s.n_emitted
        prompt_lens.append(len(ids))
        print(f"  [{i + 1}/{len(prompts)}] plen={len(ids):4d} gen={s.n_emitted:3d} "
              f"accept_len={(s.n_accepted / blocks + 1) if blocks else 0:.2f}", flush=True)

    sec = time.perf_counter() - t0
    macro = float(np.mean(taus)) if taus else 0.0
    micro = (tot_acc / tot_blocks + 1.0) if tot_blocks else 0.0
    acc_rate = 100 * tot_acc / (tot_blocks * block) if tot_blocks else 0.0
    print(f"\n=== {args.arch}/{args.precision} | {args.dataset} | N={len(prompts)} | "
          f"block={block} | chat={chat} ===")
    print(f"  avg accepted length (macro):  {macro:.2f}   tokens / verify step")
    print(f"  avg accepted length (micro):  {micro:.2f}")
    print(f"  draft acceptance rate:        {acc_rate:.1f}%")
    print(f"  mean prompt / gen tokens:     {np.mean(prompt_lens):.0f} / {tot_emit / len(prompts):.0f}")
    print(f"  throughput:                   {tot_emit / sec:.1f} tok/s  ({sec:.0f}s total)")


if __name__ == "__main__":
    main()
