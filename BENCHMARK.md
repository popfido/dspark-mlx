# dspark-mlx benchmark results

All runs on Apple Silicon (Metal), single sequence, greedy (temperature 0), **lossless**
(every emitted token is the base model's argmax from the verify forward; bit-exact vs base
greedy in float32 — bf16 divergence from a sequential reference is batch-order
nondeterminism, not loss). Decode loop = `generate_eager` (one base forward per cycle).

## Speedup (DSpark vs plain base greedy)

> Speedup is **hardware-bound** — the DSpark paper reports datacenter-GPU serving numbers
> (51–400% throughput, 60–85% per-user). These Metal single-stream numbers are not directly
> comparable; the comparable metric is acceptance length (below).

One consistent protocol: GSM8K-style chat prompt, greedy, `--no-think` (Qwen3), eager loop,
lossless. **Base precision** and **draft precision** are independent (`--quant-draft`).

| model | base prec | draft prec | base greedy | DSpark | speedup | acc/blk |
|---|---|---|---|---|---|---|
| Qwen3-4B | bf16 | bf16 | 37 tok/s | 47 tok/s | 1.26× | 2.9 |
| Qwen3-4B | bf16 | **q8** | 37 tok/s | 62 tok/s | **1.67×** | 2.9 |
| Qwen3-4B | 8-bit | bf16 | 62 tok/s | 83 tok/s | 1.34× | 4.3 |
| Qwen3-4B | 8-bit | **q8** | 62 tok/s | 101 tok/s | **1.62×** | 4.3 |
| **Qwen3-14B** | bf16 | bf16 | 12.3 tok/s | 25.4 tok/s | 2.07× | 4.4 |
| **Qwen3-14B** | bf16 | **q8** | 12.4 tok/s | 29.1 tok/s | **2.36×** | 4.4 |
| **Gemma4-12B-it** | bf16 | bf16 | 13.8 tok/s | 23.6 tok/s | 1.71× | 4.0 |
| **Gemma4-12B-it** | bf16 | **q8** | 13.9 tok/s | 26.0 tok/s | **1.87×** | 4.0 |

(q4 draft is ~similar to q8 but ~3–7% lower acceptance — see *Quantizing the drafter*. Speedup
is prompt-dependent via acceptance; higher-acceptance prompts go higher, e.g. Gemma4-12B-it
reaches 2.06× on a longer reasoning prompt.)

**Three knobs.** (1) *Acceptance length* — set by the **draft + task**, roughly **size-independent**
within a family (Qwen3-4B ≈ 14B). (2) *Base precision / size* — a costlier base amortizes the
fixed draft + verify overhead better, so at similar acceptance the bigger/heavier base wins
(Qwen3-4B 1.26× → 14B 2.07×); a *cheaper* (8-bit) base wins less. (3) *Draft precision* — an 8-bit
draft is acceptance-lossless and ~2× cheaper, adding ~10–20% on top of any base (q8 column).

The "8-bit base loses 0.75×" headline from earlier was a **thinking-on artifact** (low acceptance)
plus the bf16-draft cost — with `--no-think` and a **q8 draft** the 8-bit base reaches **1.62×**.
(Separately, the *pretrained* Gemma base is the wrong target — see the acceptance section.)

## Quantizing the drafter

DeepSeek-V4 ships its draft as **fp8** (the DSpark layers live in the fp8 base checkpoint), so a
low-precision drafter is the intended design. `--quant-draft {8,4}` (`dspark_mlx.quantize_drafter`)
quantizes the **whole draft** — compute Linears, `lm_head`, the token embedding, and the Markov
head's rank→vocab projection. 8-bit is **acceptance-lossless**, and it's both smaller and faster
(the Markov projection is a real per-block matmul, so quantizing it helps speed too, not just
size):

| Qwen3-4B draft | size | accepted length | bf16-base speedup |
|---|---|---|---|
| bf16 | 2786 MB (1.00×) | 6.12 | 1.26× |
| **q8** | **1480 MB (0.53×)** | **6.11** | **1.67×** |
| q4 | 784 MB (0.28×) | 5.88 | 1.57× |

Gemma4-12B draft: 6861 → 3645 MB (q8, accepted 5.47) → 1930 MB (q4, 5.14). 8-bit is free; 4-bit
costs ~4% (Qwen) / ~7% (Gemma) accepted length. q8 also makes the 8-bit *base* win (1.27× → 1.51×).

This is a far better lever for the quantized case than a custom small-M verify GEMM kernel — the
verify is already memory-bound-efficient at M=8 (verify(8) ≈ 2.35× a single decode), so the bf16
*draft*, not the verify, was the bottleneck. (Quantizing only the Linears and not the embedding +
Markov head — a tempting "be safe" choice — is *worse* on all three axes: 0.69× size, 1.51× speed.)

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
| Qwen3-14B (think on / off) | GSM8K (math, chat) | 3.79 / **6.29** | 40% / **76%** | ≈ 4B — acceptance is size-independent |
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

## vs DFlash (and Eagle3)

DSpark, DFlash, and Eagle3 are the three drafters in DeepSeek's DeepSpec. Fair head-to-head on
**Qwen3-4B / GSM8K / same Metal hardware / 10-prompt median speedup / same greedy baseline**,
each method tuned to its best (DSpark eager loop; DFlash `--verify-mode ddtree --copyspec-mode
auto` + quantized draft — its default config was only 0.89×):

| | DSpark bf16 | **DSpark q8** | DFlash bf16 | DFlash q4 |
|---|---|---|---|---|
| thinking-on  | 1.25× | **1.48×** | 0.98× | 1.07× |
| thinking-off | 1.99× | **2.55×** | 1.33× | 1.57× |

Tokens per target-forward (the hardware-independent efficiency metric): DSpark **3.89** (on) /
**6.19** (off) vs DFlash **2.85** (on). **DSpark wins in every quadrant**; at each method's best
(thinking-off + quantized draft) DSpark **2.55×** vs DFlash **1.57×** (~1.6×). Consistent with
the paper ordering (DSpark > DFlash > Eagle3), though our gap exceeds the paper's +16–18%
accepted-length margin.

On **Gemma4-12B-it**, DSpark reaches **2.35× (q8 draft) / 1.93× (bf16)** (tokens/cycle 5.97), but
the head-to-head can't be run: **dflash-mlx doesn't support the `gemma4_unified` multimodal base**
(its target backends are `gemma4_text` + `qwen_gdn`), whereas DSpark's mlx-vlm host loads the
text tower of the unified checkpoint. So DSpark has broader base coverage here. Caveats:

- **Hardware: pre-M5 Apple GPU** (NAX matrix kernels unavailable → steel fallback). DFlash's
  custom `verify_qmm` Metal kernels target newer chips, so **DFlash would likely close the gap on
  M5+**. DSpark uses MLX's *built-in* quantized matmul everywhere (verify + quantized draft), so
  it inherits NAX automatically on M5+ without custom kernels — see finding 8.
- DFlash is tuned to its best here; DSpark-mlx is *also* untuned (no custom kernels).
- **Eagle3 has no MLX implementation** (no package / checkpoints) — paper-cited only: DSpark =
  Eagle3 +30%, DFlash = Eagle3 +16–18% accepted length.
- DFlash uses tree / block-16 speculation, DSpark block-7; `tokens_per_cycle` normalizes this.

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
7. **Quantize the draft, not the verify** (`--quant-draft`, `dspark_mlx.quantize_drafter`) —
   quantizing the *whole* draft (incl. embedding + Markov head) is acceptance-lossless at 8-bit,
   0.53× the size, and lifts speedup everywhere (Qwen3-4B bf16 base 1.26× → **1.67×** q8; 8-bit
   base 1.27× → 1.51×). A custom small-M verify GEMM kernel (DFlash's `verify_qmm.py`) would
   *not* help on this hardware — the verify is already memory-bound-efficient at M=8; the bf16
   draft was the bottleneck. (DeepSeek-V4 already ships its draft as fp8.)
8. **DSpark inherits new-hardware (M5+/NAX) speedups for free; DFlash needs custom kernels.**
   DFlash hand-wrote ~2,000 LOC of Metal (`verify_qmm.py`) because its *tree* verify has
   irregular small-M shapes (M = number of branches: m4/m16 k-split, mma2big) that MLX's stock
   ops don't optimize — and those kernels carry NAX (M5+) paths. DSpark verifies a single dense
   block of K+1 tokens and runs its quantized draft through **MLX's built-in `quantized_matmul`**
   everywhere, so it picks up NAX automatically when MLX adds it — no custom kernel to write or
   maintain. A bespoke verify kernel could only help DSpark on M5+ if MLX's built-in turned out
   suboptimal for the M=8 shape; that's testable when M5 hardware is available, and the verify is
   memory-bound at M=8 regardless, so the upside is bounded. Net: the `verify_qmm` lever is
   DFlash-specific; DSpark gets the equivalent through MLX. *(This is why the comparison above
   notes DFlash would close the gap on M5+ — it ships the kernels; DSpark waits on MLX, which is
   the right place for that work.)*

## Reproduce

```bash
# speedup (add --quant-draft 8 for a free ~25% boost, --no-think for high acceptance)
python bench/run_dspark.py  --arch qwen3 --precision bf16 --loop eager --chat --quant-draft 8 \
    --prompt "Explain why speculative decoding speeds up LLM inference."
# acceptance length
python bench/eval_accept.py --arch qwen3 --dataset gsm8k --chat --n 20 [--no-think] [--quant-draft 8]
python bench/eval_accept.py --arch qwen3 --dataset mbpp  --chat --n 20
```
