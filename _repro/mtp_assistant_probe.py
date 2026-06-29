"""Benchmark Google's Gemma-4 MTP assistant drafter as a speculative drafter
under MLX (mlx-vlm's built-in speculative subsystem) against gemma-4-12b-it.

Reuses the highest-level single-stream entry, ``mlx_vlm.generate.generate_step``,
which already wires the speculative prefill (``return_hidden`` /
``return_shared_kv``) and ``run_speculative_rounds`` internally. We only drive the
public API -- nothing under site-packages is modified.

Reported per prompt: tokens generated, acceptance length tau (mean accepted
TOKENS per round), accept_rate %, effective draft_block_size, speculative tok/s,
plain-greedy baseline tok/s, and speedup = base_time / spec_time.
"""

import time

import mlx.core as mx

from mlx_vlm import load
from mlx_vlm.generate import generate_step
from mlx_vlm.speculative import load_drafter
from mlx_vlm.speculative.utils import format_speculative_stats

TARGET_REPO = "google/gemma-4-12b-it"
DRAFT_REPO = "mlx-community/gemma-4-12B-it-assistant-bf16"

PROMPTS = [
    (
        "gsm8k",
        "Natalia sold clips to 48 of her friends in April, and then she sold "
        "half as many clips in May. How many clips did she sell altogether in "
        "April and May?",
    ),
    (
        "chat",
        "Compose a short, engaging travel blog post about a recent trip to "
        "Hawaii, highlighting cultural experiences.",
    ),
]


def _load():
    print(f"Loading target {TARGET_REPO} (cached) ...", flush=True)
    model, processor = load(TARGET_REPO)
    print(f"Loading drafter {DRAFT_REPO} ...", flush=True)
    draft_model, draft_kind = load_drafter(DRAFT_REPO)
    print(f"  drafter resolved kind = {draft_kind!r}", flush=True)
    return model, processor, draft_model, draft_kind


def _build_input_ids(processor, prompt_text):
    """Apply the gemma chat template and unwrap to an [1, L] int32 mx.array.

    mlx-vlm's processor ``apply_chat_template`` hands back an ``mx.array`` of
    shape [1, L] here, but other tokenizers may return a bare list, a numpy
    array, or a dict / BatchEncoding ({"input_ids": ...}). Normalise all of
    them through numpy and flatten to a single [1, L] row.
    """
    import numpy as np

    apply = getattr(processor, "apply_chat_template", None)
    if apply is None:
        apply = processor.tokenizer.apply_chat_template
    out = apply(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True,
        tokenize=True,
    )
    # Unwrap dict / BatchEncoding -> input_ids
    if hasattr(out, "keys"):
        out = out["input_ids"]
    elif hasattr(out, "input_ids"):
        out = out.input_ids
    ids = np.array(out).reshape(-1).astype("int32")
    return mx.array(ids[None, :])


def _consume(gen):
    """Drive a generate_step generator to exhaustion, returning the token list."""
    toks = []
    for token, _ in gen:
        toks.append(int(token))
    return toks


def _spec_step(model, draft_model, draft_kind, input_ids, max_tokens, draft_block_size=None):
    return generate_step(
        input_ids,
        model,
        None,  # pixel_values
        None,  # mask
        max_tokens=max_tokens,
        temperature=0.0,  # greedy: sampler is None + temp==0 -> argmax
        draft_model=draft_model,
        draft_kind=draft_kind,
        draft_block_size=draft_block_size,  # None -> drafter's configured block_size
    )


def _base_step(model, input_ids, max_tokens):
    return generate_step(
        input_ids,
        model,
        None,
        None,
        max_tokens=max_tokens,
        temperature=0.0,
    )


def run(model, processor, draft_model, draft_kind, prompt_text, max_new_tokens=128,
        draft_block_size=None):
    input_ids = _build_input_ids(processor, prompt_text)

    # Per-prompt warmup (compiles Metal kernels for this prompt length so the
    # timed runs measure steady-state throughput, not first-call compilation).
    _consume(_spec_step(model, draft_model, draft_kind, input_ids, 8, draft_block_size))
    _consume(_base_step(model, input_ids, 8))
    mx.clear_cache()

    # --- Speculative (greedy) ---
    t0 = time.perf_counter()
    spec_tokens = _consume(
        _spec_step(model, draft_model, draft_kind, input_ids, max_new_tokens, draft_block_size)
    )
    spec_time = time.perf_counter() - t0
    n_tokens = len(spec_tokens)

    # Acceptance stats are reset at the start of each round-loop (drafter.reset),
    # so accept_lens / draft_lens now hold exactly this run's data.
    accept_lens = list(getattr(draft_model, "accept_lens", []) or [])
    draft_lens = list(getattr(draft_model, "draft_lens", []) or [])
    rounds = len(accept_lens)
    accepted_drafts = sum(accept_lens)
    total_drafted = sum(draft_lens)
    tau = (accepted_drafts + rounds) / rounds if rounds else 1.0
    accept_rate = 100.0 * accepted_drafts / total_drafted if total_drafted else 0.0
    block_size = int(getattr(draft_model.config, "block_size", 0))
    stats_str = format_speculative_stats(draft_model)

    mx.clear_cache()

    # --- Plain greedy baseline, same token budget ---
    t0 = time.perf_counter()
    base_tokens = _consume(_base_step(model, input_ids, n_tokens))
    base_time = time.perf_counter() - t0
    mx.clear_cache()

    spec_tps = n_tokens / spec_time if spec_time > 0 else 0.0
    base_tps = len(base_tokens) / base_time if base_time > 0 else 0.0
    speedup = base_time / spec_time if spec_time > 0 else 0.0

    result = {
        "tokens": n_tokens,
        "tau": round(tau, 3),
        "accept_rate": round(accept_rate, 1),
        "block_size": block_size,
        "drafts": round(total_drafted / rounds, 2) if rounds else 0.0,  # mean drafted/round
        "spec_tps": round(spec_tps, 2),
        "base_tps": round(base_tps, 2),
        "speedup": round(speedup, 3),
    }
    return result, stats_str


def main():
    model, processor, draft_model, draft_kind = _load()
    for name, prompt in PROMPTS:
        print(f"\n=== prompt: {name} ===", flush=True)
        result, stats_str = run(model, processor, draft_model, draft_kind, prompt, 128)
        print(f"result: {result}", flush=True)
        print(f"stats : {stats_str}", flush=True)


if __name__ == "__main__":
    main()
