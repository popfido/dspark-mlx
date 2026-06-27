# dspark-mlx

Target-agnostic MLX implementation of DeepSeek **DSpark** self-speculative decoding.

DSpark augments DeepSeek-V4 native multi-token prediction with a low-rank Markov
logit bias and a per-draft-token confidence head, drafting a block of tokens that a
host base model verifies losslessly. This package owns the DSpark draft stack and the
verify/accept policy; the base model is supplied by the host through a small adapter
(see `dspark_mlx/adapter.py`).

Based on `deepseek-ai/DeepSeek-V4-Flash-DSpark` (`inference/model.py`) and the DeepSpec
codebase. Repo structure mirrors `dflash-mlx`.

Status: early. See the test suite for what is verified.
