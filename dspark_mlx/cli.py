# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash-DSpark, deepseek-ai/DeepSpec)

"""``dspark`` command line: run / benchmark / evaluate DSpark speculative decoding.

    dspark models
    dspark generate --model qwen3-4b "Explain speculative decoding." --quant-draft 8 --no-think
    dspark bench    --model qwen3-4b --quant-draft 8 --no-think
    dspark eval     --model qwen3-4b --dataset gsm8k --n 20 --no-think
"""

from __future__ import annotations

import argparse
import time
from typing import List, Optional, Sequence

import mlx.core as mx
import numpy as np

from .events import SummaryEvent, TokenEvent
from .loader import KNOWN_MODELS, load_draft, load_host, resolve_model
from .loop import generate_eager


def _encode(tokenizer, text: str, chat: bool, think: bool) -> List[int]:
    if chat and getattr(tokenizer, "chat_template", None):
        kw = {} if think else {"enable_thinking": False}
        out = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True, **kw
        )
        if hasattr(out, "keys"):
            out = out["input_ids"]
        if len(out) and isinstance(out[0], (list, tuple)):
            out = out[0]
        return [int(t) for t in out]
    return tokenizer.encode(text)


def _load(args):
    draft_ref, base_ref = resolve_model(args.model, args.base)
    print(f"loading draft {draft_ref} (q{args.quant_draft or 'bf16'}) + base {base_ref} ({args.precision}) ...",
          flush=True)
    drafter, cfg = load_draft(draft_ref, quant_bits=args.quant_draft)
    model, tok, adapter = load_host(base_ref, cfg, args.precision)
    return drafter, model, tok, adapter


def _base_greedy(adapter, ids: List[int], n: int):
    adapter.reset()
    t0 = time.perf_counter()
    s = adapter.prefill(mx.array([ids], dtype=mx.int32))
    t = int(mx.argmax(s.logits[0].astype(mx.float32)).item())
    out = [t]
    while len(out) < n:
        s = adapter.decode_step(mx.array([t], dtype=mx.int32))
        t = int(mx.argmax(s.logits[0].astype(mx.float32)).item())
        out.append(t)
    return out, time.perf_counter() - t0


def _run_dspark(adapter, drafter, ids: List[int], n: int, eos: Optional[int]):
    adapter.reset()
    drafter.reset()
    t0 = time.perf_counter()
    events = list(generate_eager(adapter, drafter, [ids], max_new_tokens=n, eos_id=eos))
    sec = time.perf_counter() - t0
    toks = [e.token for e in events if isinstance(e, TokenEvent)]
    summ = next(e for e in events if isinstance(e, SummaryEvent))
    blocks = summ.n_drafted // drafter.block_size if drafter.block_size else 0
    accepted_len = summ.n_accepted / blocks + 1 if blocks else 0.0
    return toks, sec, accepted_len


def cmd_models(_args) -> None:
    print(f"{'name':12s}  draft  +  base (deployed instruct)")
    for name, (draft, base) in KNOWN_MODELS.items():
        print(f"{name:12s}  {draft}  +  {base}")
    print("\nGemma bases are gemma4_unified (multimodal) — install the 'gemma' extra for mlx-vlm.")


def cmd_generate(args) -> None:
    drafter, model, tok, adapter = _load(args)
    ids = _encode(tok, args.prompt, chat=not args.no_chat, think=not args.no_think)
    eos = None if args.no_eos else getattr(tok, "eos_token_id", None)
    toks, sec, acc = _run_dspark(adapter, drafter, ids, args.max_tokens, eos)
    print("\n" + tok.decode(toks))
    print(f"\n[{len(toks)} tok · {len(toks) / sec:.1f} tok/s · accepted-len {acc:.2f} / {drafter.block_size}]")


def cmd_bench(args) -> None:
    drafter, model, tok, adapter = _load(args)
    ids = _encode(tok, args.prompt, chat=not args.no_chat, think=not args.no_think)
    _run_dspark(adapter, drafter, ids, 8, None)
    _base_greedy(adapter, ids, 8)  # warmup
    toks, ds_sec, acc = _run_dspark(adapter, drafter, ids, args.max_tokens, None)
    _, base_sec = _base_greedy(adapter, ids, len(toks))
    print(f"\n=== {args.model} | base {args.precision} | draft q{args.quant_draft or 'bf16'} | "
          f"{'no-think' if args.no_think else 'chat'} ===")
    print(f"  accepted length : {acc:.2f} / {drafter.block_size}")
    print(f"  base greedy     : {len(toks) / base_sec:6.1f} tok/s")
    print(f"  DSpark          : {len(toks) / ds_sec:6.1f} tok/s")
    print(f"  speedup         : {base_sec / ds_sec:.2f}x")


def cmd_eval(args) -> None:
    from datasets import load_dataset

    drafter, model, tok, adapter = _load(args)
    if args.dataset == "gsm8k":
        rows = [ex["question"] for ex in load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))]
    else:
        ds = load_dataset("mbpp", split="test").select(range(args.n))
        rows = [f"{ex['text']}\n{chr(10).join(ex['test_list'])}" for ex in ds]
    eos = getattr(tok, "eos_token_id", None)
    taus = []
    for q in rows:
        ids = _encode(tok, q, chat=not args.no_chat, think=not args.no_think)
        _, _, acc = _run_dspark(adapter, drafter, ids, args.max_tokens, eos)
        taus.append(acc)
    print(f"\n=== {args.model} | {args.dataset} | N={len(rows)} ===")
    print(f"  avg accepted length : {np.mean(taus):.2f}  (tokens / verify step)")


def _add_model_args(sp, *, prompt: bool = False) -> None:
    sp.add_argument("--model", default="qwen3-4b",
                    help=f"known name {sorted(KNOWN_MODELS)} or a draft HF ref (then pass --base)")
    sp.add_argument("--base", default=None, help="base model HF ref (overrides the default for known names)")
    sp.add_argument("--precision", choices=["bf16", "8bit"], default="bf16")
    sp.add_argument("--quant-draft", type=int, choices=[0, 8, 4], default=0,
                    help="quantize the draft (8 = acceptance-lossless, half size)")
    sp.add_argument("--no-think", action="store_true", help="disable Qwen3 thinking mode (higher acceptance)")
    sp.add_argument("--no-chat", action="store_true", help="raw prompt, skip the chat template")
    sp.add_argument("--max-tokens", type=int, default=256)
    if prompt:
        sp.add_argument("prompt", help="the prompt text")


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="dspark", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("models", help="list known model name -> (draft, base) mappings")
    gen = sub.add_parser("generate", help="generate from a prompt")
    _add_model_args(gen, prompt=True)
    gen.add_argument("--no-eos", action="store_true")
    g = sub.add_parser("bench", help="DSpark vs plain base greedy speedup")
    _add_model_args(g)
    g.add_argument("--prompt", default="Explain why speculative decoding speeds up LLM inference.")
    g.add_argument("--no-eos", action="store_true")
    e = sub.add_parser("eval", help="average accepted length over GSM8K / MBPP")
    _add_model_args(e)
    e.add_argument("--dataset", choices=["gsm8k", "mbpp"], default="gsm8k")
    e.add_argument("--n", type=int, default=20)

    args = p.parse_args(argv)
    if args.command == "generate" and not args.no_eos:
        args.no_eos = False
    {"models": cmd_models, "generate": cmd_generate, "bench": cmd_bench, "eval": cmd_eval}[args.command](args)


if __name__ == "__main__":
    main()
