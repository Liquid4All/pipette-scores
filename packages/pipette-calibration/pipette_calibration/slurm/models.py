"""Frozen model registry for slurm calibration runs.

Adding a model is a code change: append a ModelSpec and open a PR. The TSV
models.conf is gone — everything lives here so tests can import it and there
is no parse step to drift.
"""

from dataclasses import dataclass

from pipette_scores.types import EvalId


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str  # HuggingFace id (org/repo)
    mem: str  # sbatch --mem (host RAM), e.g. "16G"
    base_shards: int  # shards, multiplied per-eval via EVAL_SHARD_MULTIPLIER


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("LiquidAI/LFM2-350M", "16G", 3),
    ModelSpec("LiquidAI/LFM2-700M", "16G", 4),
    ModelSpec("LiquidAI/LFM2-1.2B", "16G", 6),
    ModelSpec("LiquidAI/LFM2-2.6B", "24G", 8),
    ModelSpec("LiquidAI/LFM2-2.6B-Exp", "24G", 8),
    ModelSpec("LiquidAI/LFM2-8B-A1B", "32G", 16),
    ModelSpec("LiquidAI/LFM2.5-350M", "16G", 3),
    ModelSpec("LiquidAI/LFM2.5-1.2B-Instruct", "16G", 6),
    ModelSpec("unsloth/gemma-4-E2B-it", "16G", 4),
    ModelSpec("unsloth/gemma-4-E4B-it", "24G", 8),
    ModelSpec("Qwen/Qwen3-0.6B", "16G", 3),
    ModelSpec("Qwen/Qwen3-1.7B", "16G", 4),
    ModelSpec("Qwen/Qwen3-4B", "24G", 6),
    ModelSpec("Qwen/Qwen3-8B", "32G", 8),
    ModelSpec("Qwen/Qwen3.5-0.8B", "16G", 3),
    ModelSpec("Qwen/Qwen3.5-2B", "16G", 8),
    ModelSpec("Qwen/Qwen3.5-4B", "24G", 12),
    ModelSpec("Qwen/Qwen3.5-9B", "32G", 16),
    ModelSpec("Qwen/Qwen2.5-0.5B-Instruct", "16G", 3),
    ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", "16G", 4),
    ModelSpec("Qwen/Qwen2.5-3B-Instruct", "16G", 6),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct", "32G", 8),
    ModelSpec("ibm-granite/granite-4.0-h-350m", "16G", 3),
    ModelSpec("ibm-granite/granite-4.0-h-1b", "16G", 4),
    ModelSpec("ibm-granite/granite-4.0-h-micro", "16G", 6),
    ModelSpec("ibm-granite/granite-4.0-h-tiny", "24G", 8),
    ModelSpec("ibm-granite/granite-4.0-h-small", "32G", 16),
    ModelSpec("google/gemma-3-270m-it", "16G", 3),
    ModelSpec("google/gemma-3-1b-it", "16G", 4),
    ModelSpec("google/gemma-3-4b-it", "24G", 6),
    ModelSpec("google/gemma-3n-E4B-it", "32G", 8),
    ModelSpec("google/gemma-3-12b-it", "32G", 16),
    ModelSpec("meta-llama/Llama-3.2-1B-Instruct", "16G", 4),
    ModelSpec("meta-llama/Llama-3.2-3B-Instruct", "16G", 6),
    ModelSpec("meta-llama/Meta-Llama-3.1-8B-Instruct", "32G", 8),
    ModelSpec("HuggingFaceTB/SmolLM2-360M-Instruct", "16G", 3),
    ModelSpec("HuggingFaceTB/SmolLM2-1.7B-Instruct", "16G", 4),
    ModelSpec("HuggingFaceTB/SmolLM3-3B", "16G", 6),
    ModelSpec("microsoft/Phi-4-mini-instruct", "16G", 6),
    ModelSpec("microsoft/Phi-3-mini-4k-instruct", "24G", 6),
    ModelSpec("mistralai/Ministral-3-3B-Instruct-2512", "16G", 6),
    ModelSpec("mistralai/Ministral-3-3B-Reasoning-2512", "16G", 6),
    ModelSpec("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "32G", 8),
    ModelSpec("stabilityai/stablelm-zephyr-3b", "16G", 6),
    ModelSpec("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "16G", 4),
    ModelSpec("NousResearch/Hermes-2-Pro-Mistral-7B", "32G", 8),
    ModelSpec("CohereForAI/aya-23-8B", "32G", 8),
    ModelSpec("CohereForAI/aya-expanse-8b", "32G", 8),
    ModelSpec("tiiuae/Falcon3-1B-Instruct", "16G", 4),
    ModelSpec("tiiuae/Falcon3-3B-Instruct", "16G", 6),
    ModelSpec("tiiuae/Falcon3-7B-Instruct", "32G", 8),
    ModelSpec("tiiuae/Falcon3-Mamba-7B-Instruct", "32G", 8),
    ModelSpec("allenai/OLMo-2-0425-1B-Instruct", "16G", 4),
    ModelSpec("allenai/Olmo-3-7B-Instruct", "32G", 8),
    ModelSpec("nvidia/NVIDIA-Nemotron-Nano-9B-v2", "32G", 16),
    ModelSpec("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "24G", 6),
)


def effective_shards(spec: ModelSpec, eval_id: EvalId, dataset: str) -> int:
    """Shard count for a (model, eval, dataset).

    Non-default datasets are representative subsets (~200 samples) — no need
    to shard; collapse to 1 so sbatch doesn't fan out uselessly.
    """
    if dataset != "default":
        return 1
    return spec.base_shards
