#!/bin/bash
#SBATCH --job-name=build-vlm-datasets
#SBATCH --account=liquidai
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pipette/OUT_%x.%j
#
# Build all 7 VLM benchmark datasets (full sets) from VLMEvalKit's pinned
# upstream TSVs. CPU-only, I/O bound — one job loops the 7 builder scripts.
#
# From the pipette-scores repo root:
#   mkdir -p logs/pipette
#   sbatch packages/pipette-scores/scripts/build_vlm_datasets.sbatch
#
# Extra flags are forwarded to every builder, e.g.:
#   sbatch packages/pipette-scores/scripts/build_vlm_datasets.sbatch --insecure --max-image-px 512
#
# (--insecure skips TLS verification for opencompass.openxlab.space, whose
# certificate has been observed expired; integrity is still enforced via the
# MD5s pinned from VLMEvalKit.)
#
# Output: datasets/<eval>/2026.10.1/{test.parquet,metadata.json}. LICENSING: the
# built datasets must never be committed or redistributed — see
# docs/dataset-licenses.md.

set -euo pipefail

# Resolve the repo root. Under slurm the batch script executes from a spooled
# COPY (e.g. /cm/local/apps/slurm/...), so BASH_SOURCE cannot locate the repo —
# use $SLURM_SUBMIT_DIR (the sbatch invocation dir, i.e. the repo root per the
# usage above). Fall back to BASH_SOURCE for direct local execution.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "$PROJECT_ROOT"

SCRIPTS_DIR="packages/pipette-scores/scripts"

for eval in mme mmstar ocrbench mmmu_val realworldqa mmbench_en_dev pope; do
    echo "=== building ${eval} ==="
    uv run --no-sync --project packages/pipette-scores \
        python "${SCRIPTS_DIR}/build_${eval}_dataset.py" "$@"
done

echo "=== all 7 VLM datasets built ==="
