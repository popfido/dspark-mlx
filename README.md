# dspark-mlx

Target-agnostic MLX implementation of DeepSeek **DSpark** self-speculative decoding.

DSpark drafts a block of tokens from a small EAGLE-style draft model (projected target
hidden states + a low-rank Markov logit bias + a per-token confidence head), which the
host base model then verifies **losslessly**. This package owns the DSpark draft stack and
the verify/accept policy; the base model is supplied by the host through a small adapter
(`dspark_mlx/adapter.py`). The emitted stream is identical to greedy decoding from the base
model alone.

## Install

```bash
pip install dspark-mlx          # Qwen3 (mlx-lm) and Gemma (mlx-vlm) both supported out of the box
```

```bash
dspark generate --model qwen3-4b "Explain speculative decoding." --quant-draft 8 --no-think
dspark bench    --model qwen3-4b --quant-draft 8 --no-think
dspark eval     --model qwen3-4b --dataset gsm8k --n 20
```

The package itself is tiny; `dspark` downloads the DSpark draft + the deployed **instruct** base
on first use (some bases — e.g. `google/gemma-4-12b-it` — are gated and large, ~24 GB).

## Architectures

One DSpark recipe, three base-model backbones — selected by `model_type` via the registry
(`dspark_mlx/registry.py`), mirroring `dflash-mlx`'s `TARGET_BACKENDS`:

| Backbone | Checkpoint | Draft layer body |
|---|---|---|
| `deepseek_v4` | `DeepSeek-V4-Flash-DSpark` (bundled fp8/fp4, `mtp.*`) | MLA + hash-MoE + Hyper-Connections + windowed sparse attn |
| `qwen3` | `dspark_qwen3_{4b,8b,14b}_block7` (standalone bf16, `layers.*`) | Qwen3 GQA + QK-norm + SwiGLU |
| `gemma4` | `dspark_gemma4_12b_block7` (standalone bf16, `layers.*`) | Gemma4 GQA (K=V) + sandwich norms + GeGLU + partial RoPE + softcap |

Add an architecture: implement a `DraftArch` (build + key_map) in `dspark_mlx/arch/<name>.py`
and append it to `ARCH_REGISTRY` — `generate`/`verify`/`adapter` are unchanged.

```python
from dspark_mlx import resolve_arch, load_drafter, generate

arch = resolve_arch(config)                      # by config["model_type"]
drafter = arch.build(config, max_seq_len=...)
load_drafter(drafter, weights, key_map=arch.key_map)
for event in generate(adapter, drafter, prompt_tokens, max_new_tokens):
    ...
```

## Benchmarks

Lossless speedup on Apple Silicon (eager loop, greedy) and average **accepted length**
(τ = tokens per verify step) vs the DSpark paper — the hardware-independent metric the paper
reports. Both backbones land in the paper's acceptance band. Full methodology, all precisions,
and findings in **[BENCHMARK.md](https://github.com/popfido/dspark-mlx/blob/main/BENCHMARK.md)**.

| model | accepted length (GSM8K) | acceptance rate | speedup |
|---|---|---|---|
| Qwen3-4B (bf16) | 3.86 / 6.27† | 41% / 75%† | 1.17× |
| Qwen3-14B (bf16) | 3.79 / 6.29† | 40% / 76%† | **1.98×**† |
| Gemma4-12B-it (bf16) | **5.84** | **69%** | **2.06×** |

†thinking off (`--no-think`). Two knobs: **acceptance length** is set by the draft + task
(≈ size-independent — 4B ≈ 14B), while **speedup** is set by how expensive the base is (a
costlier base amortizes the draft better, so Qwen3-4B 1.17× → 14B 1.98× → Gemma-12B 2.06×).
**Quantize the draft** (`--quant-draft 8`, acceptance-lossless, ½ the draft size) for a free
~10–20% on top of any base — Qwen3-4B bf16 1.26× → **1.67×**, 8-bit base 1.34× → **1.62×**,
Qwen3-14B 2.07× → **2.36×** (see BENCHMARK.md for the base×draft-precision table). The draft must
target the deployed **instruct** model (the pretrained Gemma base gives ~3× lower acceptance);
Qwen3's `<think>` traces roughly halve acceptance vs `--no-think`.

Based on `deepseek-ai/DeepSeek-V4-Flash-DSpark` and the DeepSpec codebase (`dspark/*`).
Repo structure mirrors `dflash-mlx`.

Status: the full draft→verify→accept pipeline is verified against the reference for all
three backbones (parity on tiny weights; lossless end-to-end against toy bases; real Qwen3
/ Gemma4 checkpoints load and run). See the test suite.
