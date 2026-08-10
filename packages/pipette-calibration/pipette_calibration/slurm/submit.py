"""Submit calibration sbatch jobs for one eval across the frozen model set."""

import logging
import os
import subprocess
from pathlib import Path

from pipette_scores.types import EvalId

from pipette_calibration.slurm.models import MODELS, ModelSpec, effective_shards

logger = logging.getLogger(__name__)

RUN_EVAL_SH = Path(__file__).parent / "run_eval.sh"

# Must match run_eval.sh's HF_HOME; compute nodes start with HF_HUB_OFFLINE=1
# so the cache the head node populates has to be the one the compute node reads.
HF_HOME = str(Path.home() / "lambdafs" / ".cache" / "huggingface")


def _prefetch(hf_id: str) -> None:
    """Pre-download model weights on the head node so array tasks can run offline."""
    logger.info("Pre-downloading %s into %s", hf_id, HF_HOME)
    env = {**os.environ, "HF_HOME": HF_HOME}
    subprocess.run(
        [
            "python3",
            "-c",
            "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])",
            hf_id,
        ],
        check=True,
        env=env,
    )


def _resolve_models(names: list[str] | None) -> list[ModelSpec]:
    if not names:
        return list(MODELS)
    by_id = {m.hf_id: m for m in MODELS}
    unknown = [n for n in names if n not in by_id]
    if unknown:
        raise SystemExit(f"unknown model(s): {', '.join(unknown)}. Add to MODELS in pipette_calibration.slurm.models.")
    return [by_id[n] for n in names]


def _slug(hf_id: str) -> str:
    return hf_id.replace("/", "_")


def submit(
    *,
    eval_id: EvalId,
    models: list[str] | None = None,
    dataset: str = "default",
    exclude_nodes: str | None = None,
    dry_run: bool = False,
) -> None:
    specs = _resolve_models(models)

    if not dry_run:
        for spec in specs:
            _prefetch(spec.hf_id)

    for spec in specs:
        shards = effective_shards(spec, eval_id, dataset)
        args = [
            "sbatch",
            f"--job-name={eval_id}_{_slug(spec.hf_id)}",
            f"--mem={spec.mem}",
        ]
        if exclude_nodes:
            args.append(f"--exclude={exclude_nodes}")
        if shards > 1:
            args.append(f"--array=0-{shards - 1}")
        args += [str(RUN_EVAL_SH), eval_id, spec.hf_id, dataset]

        if dry_run:
            print(" ".join(args))
        else:
            subprocess.run(args, check=True)
