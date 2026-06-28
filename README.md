# dspark-mlx

Target-agnostic MLX implementation of DeepSeek **DSpark** self-speculative decoding.

DSpark drafts a block of tokens from a small EAGLE-style draft model (projected target
hidden states + a low-rank Markov logit bias + a per-token confidence head), which the
host base model then verifies **losslessly**. This package owns the DSpark draft stack and
the verify/accept policy; the base model is supplied by the host through a small adapter
(`dspark_mlx/adapter.py`). The emitted stream is identical to greedy decoding from the base
model alone.

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

Based on `deepseek-ai/DeepSeek-V4-Flash-DSpark` and the DeepSpec codebase (`dspark/*`).
Repo structure mirrors `dflash-mlx`.

Status: the full draft→verify→accept pipeline is verified against the reference for all
three backbones (parity on tiny weights; lossless end-to-end against toy bases; real Qwen3
/ Gemma4 checkpoints load and run). See the test suite.
