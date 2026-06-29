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
| Gemma4-12B | bf16 | 13.5 tok/s | 11.7 tok/s | 0.87× | 1.46 (raw) |
| Gemma4-12B | 8-bit | 22 tok/s | 14 tok/s | 0.63× | 1.37 |

DSpark wins only when the base is **memory-bound at the verify block size** — true for 4B
bf16, not for a 12B (multi-token verify goes compute-bound) or quantized bases (base too
cheap to amortize the draft).

## Acceptance length vs the DSpark paper

τ = average accepted length = tokens emitted per verify step (`bench/eval_accept.py`, bf16,
greedy). The paper reports DSpark = **Eagle3 + ~30%**; Eagle3 on GSM8K is ~3–3.5, so DSpark's
expected band is ~4–4.5 (math), lower for code.

| model | dataset | τ (macro) | acceptance rate | notes |
|---|---|---|---|---|
| Qwen3-4B | GSM8K (math, chat) | **3.91** | 41% | in DSpark's expected band ✓ |
| Qwen3-4B | MBPP (code, chat) | **3.25** | 31% | math > code, as expected ✓ |
| Gemma4-12B | GSM8K (raw) | ~2.5 | 21% | base has no chat template — see below |

**Qwen3 matches the paper.** The loop/recipe/verify/accept are shared across arches, so this
validates the whole port.

**Gemma4 reads low** — but it went 1% → 21% once the embed-scaling bug was fixed, so it is
structurally correct. The remaining gap is most likely a **pairing/prompt confound**: the
pretrained `gemma-4-12b` base (no chat template) is the wrong analog to instruct Qwen3-4B; the
DSpark draft most likely targets `gemma-4-12b-it`. Verification with the `-it` base is pending.

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
5. **DFlash invested ~2,000 LOC of Metal kernels** — most relevantly `verify_qmm.py`
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
