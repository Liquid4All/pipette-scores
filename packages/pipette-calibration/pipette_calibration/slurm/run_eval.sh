#!/bin/bash
#SBATCH --output=/home/%u/lambdafs/logs/%x-%A_%a.out
#SBATCH --error=/home/%u/lambdafs/logs/%x-%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#
# sbatch body for one (eval, model, dataset) calibration job. Invoked by
# `pipette-calibration slurm-completions submit`. Dynamic flags (--job-name, --mem,
# --array, --exclude) come from the sbatch CLI call.
#
# Positional args: <eval> <model> <dataset>

set -euo pipefail

EVAL="$1"
MODEL="$2"
DATASET="${3:-default}"

# This script lives at:
#   <repo>/packages/pipette-calibration/pipette_calibration/slurm/run_eval.sh
# Climb four levels to reach the workspace root so `uv run` finds pyproject.toml.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PROJECT_ROOT"

module load cuda13.0/toolkit/13.0.2
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"

# Model is pre-downloaded by `slurm-completions submit`; run offline so array tasks don't
# all hit the HF API at once.
export HF_HOME="$HOME/lambdafs/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Redirect torch-inductor / triton JIT caches off the shared /tmp, which fills
# up fast on the head node and causes ENOSPC mid-compile. lambdafs has room.
export TORCHINDUCTOR_CACHE_DIR="$HOME/lambdafs/.cache/torchinductor"
export TRITON_CACHE_DIR="$HOME/lambdafs/.cache/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

SHARD_ARGS=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SHARD_ARGS=(--shard "$SLURM_ARRAY_TASK_ID" --num-shards "$SLURM_ARRAY_TASK_COUNT")
fi

exec uv run --package pipette-calibration --group gpu pipette-calibration generate-completions \
    --eval "$EVAL" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output-dir "$HOME/lambdafs/calibration/results" \
    --hf-cache "$HF_HOME" \
    "${SHARD_ARGS[@]}"
