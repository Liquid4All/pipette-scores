# GGUF quant reference

GGUF repo + filename picks for each entry in `pipette_calibration/slurm/models.py`,
covering quants `Q4_0`, `Q4_K_M`, `Q5_K_M`, `Q8_0`.

Verified against the HuggingFace API on 2026-04-17.

## Pick rule

Org priority: **original** (model's HF org) → **unsloth** → **ggml-org** → **bartowski**.

For each model, the first org (in priority order) that publishes **all four** target
quants wins. If no org covers all four, the highest-priority org with the most
coverage wins and the gap is called out below.

Repos outside this priority list (QuantFactory, andrijdavid, TheBloke, lmstudio-community,
etc.) are intentionally excluded, even when they carry the missing quant.

## Gaps

Skipped — no GGUF found in any priority org:

- `stabilityai/stablelm-zephyr-3b`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (only `andrijdavid/*-GGUF` has it)

Partial coverage — no priority org publishes one of the four:

| Model | Winner | Missing |
|---|---|---|
| `Qwen/Qwen3-8B` | `Qwen/Qwen3-8B-GGUF` | `Q4_0` |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | `unsloth/Llama-3.1-8B-Instruct-GGUF` | `Q8_0` |
| `microsoft/Phi-3-mini-4k-instruct` | `bartowski/Phi-3-mini-4k-instruct-GGUF` | `Q4_0` |
| `CohereForAI/aya-23-8B` | `bartowski/aya-23-8B-GGUF` | `Q4_0` |

## Naming notes

- **Qwen/Qwen2.5-\*-Instruct-GGUF** — lowercase filenames (`qwen2.5-0.5b-instruct-q4_0.gguf`).
- **Qwen/Qwen2.5-7B-Instruct-GGUF** — each quant is split across 2–3 shards (`-00001-of-00002.gguf` etc.).
- **tiiuae/Falcon3-\*-GGUF** — lowercase quant suffix (`-q4_0.gguf`).
- **NousResearch/Hermes-2-Pro-Mistral-7B-GGUF** — dot separator (`Hermes-2-Pro-Mistral-7B.Q4_0.gguf`).
- **bartowski/microsoft_Phi-4-mini-instruct-GGUF** — repo and filenames are prefixed with `microsoft_`.
- **bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF** — repo and filenames prefixed with `nvidia_`.
- **unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF** — repo name drops the `-BF16` suffix present in the base model id.

## List

```python
GGUFS = [
    # LiquidAI LFM2 / LFM2.5 — original
    "LiquidAI/LFM2-350M-GGUF:LFM2-350M-Q4_0.gguf",
    "LiquidAI/LFM2-350M-GGUF:LFM2-350M-Q4_K_M.gguf",
    "LiquidAI/LFM2-350M-GGUF:LFM2-350M-Q5_K_M.gguf",
    "LiquidAI/LFM2-350M-GGUF:LFM2-350M-Q8_0.gguf",
    "LiquidAI/LFM2-700M-GGUF:LFM2-700M-Q4_0.gguf",
    "LiquidAI/LFM2-700M-GGUF:LFM2-700M-Q4_K_M.gguf",
    "LiquidAI/LFM2-700M-GGUF:LFM2-700M-Q5_K_M.gguf",
    "LiquidAI/LFM2-700M-GGUF:LFM2-700M-Q8_0.gguf",
    "LiquidAI/LFM2-1.2B-GGUF:LFM2-1.2B-Q4_0.gguf",
    "LiquidAI/LFM2-1.2B-GGUF:LFM2-1.2B-Q4_K_M.gguf",
    "LiquidAI/LFM2-1.2B-GGUF:LFM2-1.2B-Q5_K_M.gguf",
    "LiquidAI/LFM2-1.2B-GGUF:LFM2-1.2B-Q8_0.gguf",
    "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_0.gguf",
    "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf",
    "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q5_K_M.gguf",
    "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q8_0.gguf",
    "LiquidAI/LFM2-2.6B-Exp-GGUF:LFM2-2.6B-Exp-Q4_0.gguf",
    "LiquidAI/LFM2-2.6B-Exp-GGUF:LFM2-2.6B-Exp-Q4_K_M.gguf",
    "LiquidAI/LFM2-2.6B-Exp-GGUF:LFM2-2.6B-Exp-Q5_K_M.gguf",
    "LiquidAI/LFM2-2.6B-Exp-GGUF:LFM2-2.6B-Exp-Q8_0.gguf",
    "LiquidAI/LFM2-8B-A1B-GGUF:LFM2-8B-A1B-Q4_0.gguf",
    "LiquidAI/LFM2-8B-A1B-GGUF:LFM2-8B-A1B-Q4_K_M.gguf",
    "LiquidAI/LFM2-8B-A1B-GGUF:LFM2-8B-A1B-Q5_K_M.gguf",
    "LiquidAI/LFM2-8B-A1B-GGUF:LFM2-8B-A1B-Q8_0.gguf",
    "LiquidAI/LFM2.5-350M-GGUF:LFM2.5-350M-Q4_0.gguf",
    "LiquidAI/LFM2.5-350M-GGUF:LFM2.5-350M-Q4_K_M.gguf",
    "LiquidAI/LFM2.5-350M-GGUF:LFM2.5-350M-Q5_K_M.gguf",
    "LiquidAI/LFM2.5-350M-GGUF:LFM2.5-350M-Q8_0.gguf",
    "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:LFM2.5-1.2B-Instruct-Q4_0.gguf",
    "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
    "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:LFM2.5-1.2B-Instruct-Q5_K_M.gguf",
    "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:LFM2.5-1.2B-Instruct-Q8_0.gguf",

    # Gemma 4 — original (unsloth is the model org for these entries)
    "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_0.gguf",
    "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_K_M.gguf",
    "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q5_K_M.gguf",
    "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q8_0.gguf",
    "unsloth/gemma-4-E4B-it-GGUF:gemma-4-E4B-it-Q4_0.gguf",
    "unsloth/gemma-4-E4B-it-GGUF:gemma-4-E4B-it-Q4_K_M.gguf",
    "unsloth/gemma-4-E4B-it-GGUF:gemma-4-E4B-it-Q5_K_M.gguf",
    "unsloth/gemma-4-E4B-it-GGUF:gemma-4-E4B-it-Q8_0.gguf",

    # Qwen3 — unsloth
    "unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_0.gguf",
    "unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf",
    "unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q5_K_M.gguf",
    "unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf",
    "unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q4_0.gguf",
    "unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q4_K_M.gguf",
    "unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q5_K_M.gguf",
    "unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q8_0.gguf",
    "unsloth/Qwen3-4B-GGUF:Qwen3-4B-Q4_0.gguf",
    "unsloth/Qwen3-4B-GGUF:Qwen3-4B-Q4_K_M.gguf",
    "unsloth/Qwen3-4B-GGUF:Qwen3-4B-Q5_K_M.gguf",
    "unsloth/Qwen3-4B-GGUF:Qwen3-4B-Q8_0.gguf",
    # Qwen3-8B: Q4_0 not published in any priority org
    "Qwen/Qwen3-8B-GGUF:Qwen3-8B-Q4_K_M.gguf",
    "Qwen/Qwen3-8B-GGUF:Qwen3-8B-Q5_K_M.gguf",
    "Qwen/Qwen3-8B-GGUF:Qwen3-8B-Q8_0.gguf",

    # Qwen3.5 — unsloth
    "unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q4_0.gguf",
    "unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q4_K_M.gguf",
    "unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q5_K_M.gguf",
    "unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q8_0.gguf",
    "unsloth/Qwen3.5-2B-GGUF:Qwen3.5-2B-Q4_0.gguf",
    "unsloth/Qwen3.5-2B-GGUF:Qwen3.5-2B-Q4_K_M.gguf",
    "unsloth/Qwen3.5-2B-GGUF:Qwen3.5-2B-Q5_K_M.gguf",
    "unsloth/Qwen3.5-2B-GGUF:Qwen3.5-2B-Q8_0.gguf",
    "unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-Q4_0.gguf",
    "unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-Q4_K_M.gguf",
    "unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-Q5_K_M.gguf",
    "unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-Q8_0.gguf",
    "unsloth/Qwen3.5-9B-GGUF:Qwen3.5-9B-Q4_0.gguf",
    "unsloth/Qwen3.5-9B-GGUF:Qwen3.5-9B-Q4_K_M.gguf",
    "unsloth/Qwen3.5-9B-GGUF:Qwen3.5-9B-Q5_K_M.gguf",
    "unsloth/Qwen3.5-9B-GGUF:Qwen3.5-9B-Q8_0.gguf",

    # Qwen2.5 — original; 7B is sharded, filenames are lowercase
    "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_0.gguf",
    "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q5_k_m.gguf",
    "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q8_0.gguf",
    "Qwen/Qwen2.5-1.5B-Instruct-GGUF:qwen2.5-1.5b-instruct-q4_0.gguf",
    "Qwen/Qwen2.5-1.5B-Instruct-GGUF:qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "Qwen/Qwen2.5-1.5B-Instruct-GGUF:qwen2.5-1.5b-instruct-q5_k_m.gguf",
    "Qwen/Qwen2.5-1.5B-Instruct-GGUF:qwen2.5-1.5b-instruct-q8_0.gguf",
    "Qwen/Qwen2.5-3B-Instruct-GGUF:qwen2.5-3b-instruct-q4_0.gguf",
    "Qwen/Qwen2.5-3B-Instruct-GGUF:qwen2.5-3b-instruct-q4_k_m.gguf",
    "Qwen/Qwen2.5-3B-Instruct-GGUF:qwen2.5-3b-instruct-q5_k_m.gguf",
    "Qwen/Qwen2.5-3B-Instruct-GGUF:qwen2.5-3b-instruct-q8_0.gguf",
    "Qwen/Qwen2.5-7B-Instruct-GGUF:qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf",
    "Qwen/Qwen2.5-7B-Instruct-GGUF:qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    "Qwen/Qwen2.5-7B-Instruct-GGUF:qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
    "Qwen/Qwen2.5-7B-Instruct-GGUF:qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf",

    # IBM Granite 4.0 — original
    "ibm-granite/granite-4.0-h-350m-GGUF:granite-4.0-h-350m-Q4_0.gguf",
    "ibm-granite/granite-4.0-h-350m-GGUF:granite-4.0-h-350m-Q4_K_M.gguf",
    "ibm-granite/granite-4.0-h-350m-GGUF:granite-4.0-h-350m-Q5_K_M.gguf",
    "ibm-granite/granite-4.0-h-350m-GGUF:granite-4.0-h-350m-Q8_0.gguf",
    "ibm-granite/granite-4.0-h-1b-GGUF:granite-4.0-h-1b-Q4_0.gguf",
    "ibm-granite/granite-4.0-h-1b-GGUF:granite-4.0-h-1b-Q4_K_M.gguf",
    "ibm-granite/granite-4.0-h-1b-GGUF:granite-4.0-h-1b-Q5_K_M.gguf",
    "ibm-granite/granite-4.0-h-1b-GGUF:granite-4.0-h-1b-Q8_0.gguf",
    "ibm-granite/granite-4.0-h-micro-GGUF:granite-4.0-h-micro-Q4_0.gguf",
    "ibm-granite/granite-4.0-h-micro-GGUF:granite-4.0-h-micro-Q4_K_M.gguf",
    "ibm-granite/granite-4.0-h-micro-GGUF:granite-4.0-h-micro-Q5_K_M.gguf",
    "ibm-granite/granite-4.0-h-micro-GGUF:granite-4.0-h-micro-Q8_0.gguf",
    "ibm-granite/granite-4.0-h-tiny-GGUF:granite-4.0-h-tiny-Q4_0.gguf",
    "ibm-granite/granite-4.0-h-tiny-GGUF:granite-4.0-h-tiny-Q4_K_M.gguf",
    "ibm-granite/granite-4.0-h-tiny-GGUF:granite-4.0-h-tiny-Q5_K_M.gguf",
    "ibm-granite/granite-4.0-h-tiny-GGUF:granite-4.0-h-tiny-Q8_0.gguf",
    "ibm-granite/granite-4.0-h-small-GGUF:granite-4.0-h-small-Q4_0.gguf",
    "ibm-granite/granite-4.0-h-small-GGUF:granite-4.0-h-small-Q4_K_M.gguf",
    "ibm-granite/granite-4.0-h-small-GGUF:granite-4.0-h-small-Q5_K_M.gguf",
    "ibm-granite/granite-4.0-h-small-GGUF:granite-4.0-h-small-Q8_0.gguf",

    # Gemma 3 — unsloth
    "unsloth/gemma-3-270m-it-GGUF:gemma-3-270m-it-Q4_0.gguf",
    "unsloth/gemma-3-270m-it-GGUF:gemma-3-270m-it-Q4_K_M.gguf",
    "unsloth/gemma-3-270m-it-GGUF:gemma-3-270m-it-Q5_K_M.gguf",
    "unsloth/gemma-3-270m-it-GGUF:gemma-3-270m-it-Q8_0.gguf",
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_0.gguf",
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q5_K_M.gguf",
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q8_0.gguf",
    "unsloth/gemma-3-4b-it-GGUF:gemma-3-4b-it-Q4_0.gguf",
    "unsloth/gemma-3-4b-it-GGUF:gemma-3-4b-it-Q4_K_M.gguf",
    "unsloth/gemma-3-4b-it-GGUF:gemma-3-4b-it-Q5_K_M.gguf",
    "unsloth/gemma-3-4b-it-GGUF:gemma-3-4b-it-Q8_0.gguf",
    "unsloth/gemma-3n-E4B-it-GGUF:gemma-3n-E4B-it-Q4_0.gguf",
    "unsloth/gemma-3n-E4B-it-GGUF:gemma-3n-E4B-it-Q4_K_M.gguf",
    "unsloth/gemma-3n-E4B-it-GGUF:gemma-3n-E4B-it-Q5_K_M.gguf",
    "unsloth/gemma-3n-E4B-it-GGUF:gemma-3n-E4B-it-Q8_0.gguf",
    "unsloth/gemma-3-12b-it-GGUF:gemma-3-12b-it-Q4_0.gguf",
    "unsloth/gemma-3-12b-it-GGUF:gemma-3-12b-it-Q4_K_M.gguf",
    "unsloth/gemma-3-12b-it-GGUF:gemma-3-12b-it-Q5_K_M.gguf",
    "unsloth/gemma-3-12b-it-GGUF:gemma-3-12b-it-Q8_0.gguf",

    # Llama 3.x — unsloth
    "unsloth/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q4_0.gguf",
    "unsloth/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "unsloth/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q5_K_M.gguf",
    "unsloth/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q8_0.gguf",
    "unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q4_0.gguf",
    "unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q5_K_M.gguf",
    "unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q8_0.gguf",
    # Meta-Llama-3.1-8B: Q8_0 not published in any priority org
    "unsloth/Llama-3.1-8B-Instruct-GGUF:Llama-3.1-8B-Instruct-Q4_0.gguf",
    "unsloth/Llama-3.1-8B-Instruct-GGUF:Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "unsloth/Llama-3.1-8B-Instruct-GGUF:Llama-3.1-8B-Instruct-Q5_K_M.gguf",

    # SmolLM — bartowski for SmolLM2, unsloth for SmolLM3
    "bartowski/SmolLM2-360M-Instruct-GGUF:SmolLM2-360M-Instruct-Q4_0.gguf",
    "bartowski/SmolLM2-360M-Instruct-GGUF:SmolLM2-360M-Instruct-Q4_K_M.gguf",
    "bartowski/SmolLM2-360M-Instruct-GGUF:SmolLM2-360M-Instruct-Q5_K_M.gguf",
    "bartowski/SmolLM2-360M-Instruct-GGUF:SmolLM2-360M-Instruct-Q8_0.gguf",
    "bartowski/SmolLM2-1.7B-Instruct-GGUF:SmolLM2-1.7B-Instruct-Q4_0.gguf",
    "bartowski/SmolLM2-1.7B-Instruct-GGUF:SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
    "bartowski/SmolLM2-1.7B-Instruct-GGUF:SmolLM2-1.7B-Instruct-Q5_K_M.gguf",
    "bartowski/SmolLM2-1.7B-Instruct-GGUF:SmolLM2-1.7B-Instruct-Q8_0.gguf",
    "unsloth/SmolLM3-3B-GGUF:SmolLM3-3B-Q4_0.gguf",
    "unsloth/SmolLM3-3B-GGUF:SmolLM3-3B-Q4_K_M.gguf",
    "unsloth/SmolLM3-3B-GGUF:SmolLM3-3B-Q5_K_M.gguf",
    "unsloth/SmolLM3-3B-GGUF:SmolLM3-3B-Q8_0.gguf",

    # Phi — bartowski (filename prefix varies)
    "bartowski/microsoft_Phi-4-mini-instruct-GGUF:microsoft_Phi-4-mini-instruct-Q4_0.gguf",
    "bartowski/microsoft_Phi-4-mini-instruct-GGUF:microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
    "bartowski/microsoft_Phi-4-mini-instruct-GGUF:microsoft_Phi-4-mini-instruct-Q5_K_M.gguf",
    "bartowski/microsoft_Phi-4-mini-instruct-GGUF:microsoft_Phi-4-mini-instruct-Q8_0.gguf",
    # Phi-3-mini-4k: Q4_0 not published in any priority org
    "bartowski/Phi-3-mini-4k-instruct-GGUF:Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    "bartowski/Phi-3-mini-4k-instruct-GGUF:Phi-3-mini-4k-instruct-Q5_K_M.gguf",
    "bartowski/Phi-3-mini-4k-instruct-GGUF:Phi-3-mini-4k-instruct-Q8_0.gguf",

    # Ministral 3 — unsloth (mistralai official repos miss Q4_0)
    "unsloth/Ministral-3-3B-Instruct-2512-GGUF:Ministral-3-3B-Instruct-2512-Q4_0.gguf",
    "unsloth/Ministral-3-3B-Instruct-2512-GGUF:Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
    "unsloth/Ministral-3-3B-Instruct-2512-GGUF:Ministral-3-3B-Instruct-2512-Q5_K_M.gguf",
    "unsloth/Ministral-3-3B-Instruct-2512-GGUF:Ministral-3-3B-Instruct-2512-Q8_0.gguf",
    "unsloth/Ministral-3-3B-Reasoning-2512-GGUF:Ministral-3-3B-Reasoning-2512-Q4_0.gguf",
    "unsloth/Ministral-3-3B-Reasoning-2512-GGUF:Ministral-3-3B-Reasoning-2512-Q4_K_M.gguf",
    "unsloth/Ministral-3-3B-Reasoning-2512-GGUF:Ministral-3-3B-Reasoning-2512-Q5_K_M.gguf",
    "unsloth/Ministral-3-3B-Reasoning-2512-GGUF:Ministral-3-3B-Reasoning-2512-Q8_0.gguf",

    # DeepSeek R1 Distill — bartowski
    "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:DeepSeek-R1-Distill-Qwen-7B-Q4_0.gguf",
    "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
    "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:DeepSeek-R1-Distill-Qwen-7B-Q5_K_M.gguf",
    "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf",

    # Hermes 2 Pro Mistral 7B — original (dot separator)
    "NousResearch/Hermes-2-Pro-Mistral-7B-GGUF:Hermes-2-Pro-Mistral-7B.Q4_0.gguf",
    "NousResearch/Hermes-2-Pro-Mistral-7B-GGUF:Hermes-2-Pro-Mistral-7B.Q4_K_M.gguf",
    "NousResearch/Hermes-2-Pro-Mistral-7B-GGUF:Hermes-2-Pro-Mistral-7B.Q5_K_M.gguf",
    "NousResearch/Hermes-2-Pro-Mistral-7B-GGUF:Hermes-2-Pro-Mistral-7B.Q8_0.gguf",

    # Cohere aya
    # aya-23-8B: Q4_0 not published in any priority org
    "bartowski/aya-23-8B-GGUF:aya-23-8B-Q4_K_M.gguf",
    "bartowski/aya-23-8B-GGUF:aya-23-8B-Q5_K_M.gguf",
    "bartowski/aya-23-8B-GGUF:aya-23-8B-Q8_0.gguf",
    "bartowski/aya-expanse-8b-GGUF:aya-expanse-8b-Q4_0.gguf",
    "bartowski/aya-expanse-8b-GGUF:aya-expanse-8b-Q4_K_M.gguf",
    "bartowski/aya-expanse-8b-GGUF:aya-expanse-8b-Q5_K_M.gguf",
    "bartowski/aya-expanse-8b-GGUF:aya-expanse-8b-Q8_0.gguf",

    # Falcon3 — original (lowercase quants)
    "tiiuae/Falcon3-1B-Instruct-GGUF:Falcon3-1B-Instruct-q4_0.gguf",
    "tiiuae/Falcon3-1B-Instruct-GGUF:Falcon3-1B-Instruct-q4_k_m.gguf",
    "tiiuae/Falcon3-1B-Instruct-GGUF:Falcon3-1B-Instruct-q5_k_m.gguf",
    "tiiuae/Falcon3-1B-Instruct-GGUF:Falcon3-1B-Instruct-q8_0.gguf",
    "tiiuae/Falcon3-3B-Instruct-GGUF:Falcon3-3B-Instruct-q4_0.gguf",
    "tiiuae/Falcon3-3B-Instruct-GGUF:Falcon3-3B-Instruct-q4_k_m.gguf",
    "tiiuae/Falcon3-3B-Instruct-GGUF:Falcon3-3B-Instruct-q5_k_m.gguf",
    "tiiuae/Falcon3-3B-Instruct-GGUF:Falcon3-3B-Instruct-q8_0.gguf",
    "tiiuae/Falcon3-7B-Instruct-GGUF:Falcon3-7B-Instruct-q4_0.gguf",
    "tiiuae/Falcon3-7B-Instruct-GGUF:Falcon3-7B-Instruct-q4_k_m.gguf",
    "tiiuae/Falcon3-7B-Instruct-GGUF:Falcon3-7B-Instruct-q5_k_m.gguf",
    "tiiuae/Falcon3-7B-Instruct-GGUF:Falcon3-7B-Instruct-q8_0.gguf",
    "tiiuae/Falcon3-Mamba-7B-Instruct-GGUF:Falcon3-Mamba-7B-Instruct-q4_0.gguf",
    "tiiuae/Falcon3-Mamba-7B-Instruct-GGUF:Falcon3-Mamba-7B-Instruct-q4_k_m.gguf",
    "tiiuae/Falcon3-Mamba-7B-Instruct-GGUF:Falcon3-Mamba-7B-Instruct-q5_k_m.gguf",
    "tiiuae/Falcon3-Mamba-7B-Instruct-GGUF:Falcon3-Mamba-7B-Instruct-q8_0.gguf",

    # AllenAI OLMo
    "allenai/OLMo-2-0425-1B-Instruct-GGUF:OLMo-2-0425-1B-Instruct-Q4_0.gguf",
    "allenai/OLMo-2-0425-1B-Instruct-GGUF:OLMo-2-0425-1B-Instruct-Q4_K_M.gguf",
    "allenai/OLMo-2-0425-1B-Instruct-GGUF:OLMo-2-0425-1B-Instruct-Q5_K_M.gguf",
    "allenai/OLMo-2-0425-1B-Instruct-GGUF:OLMo-2-0425-1B-Instruct-Q8_0.gguf",
    "unsloth/Olmo-3-7B-Instruct-GGUF:Olmo-3-7B-Instruct-Q4_0.gguf",
    "unsloth/Olmo-3-7B-Instruct-GGUF:Olmo-3-7B-Instruct-Q4_K_M.gguf",
    "unsloth/Olmo-3-7B-Instruct-GGUF:Olmo-3-7B-Instruct-Q5_K_M.gguf",
    "unsloth/Olmo-3-7B-Instruct-GGUF:Olmo-3-7B-Instruct-Q8_0.gguf",

    # Nvidia Nemotron
    "bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF:nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_0.gguf",
    "bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF:nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf",
    "bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF:nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q5_K_M.gguf",
    "bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF:nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf",
    "unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron-3-Nano-4B-Q4_0.gguf",
    "unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf",
    "unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron-3-Nano-4B-Q5_K_M.gguf",
    "unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf",
]
```

## Regenerating

The picks were produced by probing `https://huggingface.co/api/models/<repo>` for
each candidate repo per base model, collecting the `.gguf` siblings, and applying
the pick rule above. Re-run when adding to or removing from
`pipette_calibration/slurm/models.py`.
