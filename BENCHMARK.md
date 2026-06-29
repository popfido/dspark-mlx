# dspark-mlx benchmark results

All runs on Apple Silicon (Metal), single sequence, greedy (temperature 0), **lossless**
(every emitted token is the base model's argmax from the verify forward; bit-exact vs base
greedy in float32 — bf16 divergence from a sequential reference is batch-order
nondeterminism, not loss). Decode loop = `generate_eager` (one base forward per cycle).

## Speedup (DSpark vs plain base greedy)

> Speedup is **hardware-bound** — the DSpark paper reports datacenter-GPU serving numbers
> (51–400% throughput, 60–85% per-user). These Metal single-stream numbers are not directly
> comparable; the comparable metric is acceptance length (below).

| model | precision | base greedy | DSpark | speedup | accepted/block (prompt) |
|---|---|---|---|---|---|
| Qwen3-4B | bf16 | 37 tok/s | 42 tok/s | **1.17×** | 2.49 (chat) |
| Qwen3-4B | 8-bit | 69 tok/s | 52 tok/s | 0.75× | 2.24 |
| **Gemma4-12B-it** | bf16 | 13.0 tok/s | 26.8 tok/s | **2.06×** | **5.24** (chat) |
| Gemma4-12B (base/pretrained) | bf16 | 13.5 tok/s | 11.7 tok/s | 0.87× | 1.46 (raw) |

**Speedup is driven by acceptance length.** High acceptance amortizes the base forward over
more tokens, so an *expensive* base (12B) gives the *bigger* win when acceptance is high
(Gemma4-12B-it: 5.24 accepted/block → 2.06×). The Gemma `(base/pretrained)` row is the wrong
target — the DSpark draft is trained for the deployed **instruct** model, and a pretrained base
(no chat template) gives ~3× lower acceptance. **8-bit bases lose** (Qwen3 0.75×) because
quantization both cheapens the base *and* lowers acceptance (quantized hiddens diverge from the
draft's bf16 training).

## Acceptance length vs the DSpark paper

τ = average accepted length = tokens emitted per verify step (`bench/eval_accept.py`, bf16,
greedy). The paper reports DSpark = **Eagle3 + ~30%**; Eagle3 on GSM8K is ~3–3.5, so DSpark's
expected band is ~4–4.5 (math), lower for code. GSM8K test = 1,319 examples; numbers below are
from a 50-sample slice (stable vs a 12-sample slice — thinking-on identical, thinking-off ±0.2).

| model | dataset | τ (micro) | acceptance rate | notes |
|---|---|---|---|---|
| Qwen3-4B (thinking on) | GSM8K (math, chat) | **3.85** | 41% | default Qwen3 chat template |
| Qwen3-4B (**thinking off**) | GSM8K (math, chat) | **6.27** | **75%** | `--no-think` — see below |
| Qwen3-4B | MBPP (code, chat) | **3.20** | 31% | math > code, as expected ✓ |
| **Gemma4-12B-it** | GSM8K (math, chat) | **5.84** | **69%** | matches/exceeds the paper ✓ |
| Gemma4-12B (base) | GSM8K (raw) | ~2.5 | 21% | wrong target — pretrained, no chat template |

**Both arches match the paper.** DSpark's expected band is Eagle3 (~3–3.5 on GSM8K) +30%.
Qwen3-4B lands at 3.85 (thinking on) / 6.27 (thinking off); Gemma4-12B-it at 5.84. The
loop/recipe/verify/accept are shared across arches, so this validates the whole port.

**Thinking mode roughly halves acceptance.** Qwen3's chat template defaults to thinking on, so
generation starts with a free-form `<think>…</think>` trace — creative text that is hard to
draft. Disabling it (`--no-think`) nearly doubles acceptance (41% → 75%, τ 3.85 → 6.27), at
which point Qwen3-4B *exceeds* Gemma4-12B-it on the same (non-thinking) footing. Use `--no-think`
for an apples-to-apples comparison with the paper / non-reasoning models.

The Gemma `(base)` row was the wrong target: it went 1% → 21% once the embed-scaling bug was
fixed (structurally correct), but a *pretrained* base (no chat template) gives ~3× lower
acceptance than the deployed instruct model the draft is trained for. Switching to
`gemma-4-12b-it` with its chat template took it to 5.84 / 69%.

## Key findings

1. **Eager loop** (`loop.py`) — one base forward/cycle instead of two; conditions the draft
   exactly as the reference does. Qwen3-4B 0.59× → 1.17×. Both perf and a correctness fix
   (legacy loop put the anchor hidden in context + shifted the block RoPE by one).
2. **Confidence-gated draft length hurts on Metal** — verify is memory-bound, so drafting
   fewer tokens doesn't save verify time but lowers accepted-per-cycle. Default off; kept for
   compute-bound / batched backends (where the reference uses it).
3. **Draft KV cache** — marginal at short context (the lm_head + Markov block loop dominate
   the draft, not the context projection), O(ctx)→O(1) so it matters at long context.
4. **Gemma embed-scaling bug** — the draft missed Gemma's √hidden embedding scale (~62×);
   1% → 21% acceptance. Qwen3 doesn't scale embeds, so it was unaffected.
5. **Right target matters as much as the model** — the draft is trained for the *deployed
   instruct* model. The pretrained Gemma base (no chat template) gave ~3× lower acceptance;
   `gemma-4-12b-it` fixed it. (Qwen3-4B already *is* the instruct model.)
6. **Thinking mode roughly halves Qwen3 acceptance** — `<think>` traces are hard to draft;
   `--no-think` takes Qwen3-4B GSM8K (n=50) from τ 3.85 / 41% to **6.27 / 75%**.
7. **DFlash invested ~2,000 LOC of Metal kernels** — most relevantly `verify_qmm.py`
   (small-M quantized GEMM for verify). That's the lever for the quantized-base cases where
   DSpark currently loses; not the bf16 (memory-bound) case.

## Reproduce

```bash
# speedup
python bench/run_dspark.py  --arch qwen3  --precision bf16 --loop eager --chat \
    --prompt "Explain why speculative decoding speeds up LLM inference."
# acceptance length
python bench/eval_accept.py --arch qwen3  --dataset gsm8k --chat --n 20
python bench/eval_accept.py --arch qwen3  --dataset mbpp  --chat --n 20
```
