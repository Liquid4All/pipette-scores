"""Eval definition types and public dataset catalog interface."""

import functools
import json
import logging
import os
import pathlib
from collections.abc import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from pipette_scores.dataset_catalog import (
    _asr_common,
    _vlm_common,
    asr_prompt_profiles,
    gpqa_diamond,
    ifbench,
    ifstruct,
    math_500,
    prompt_profiles,
)
from pipette_scores.types import ASR_EVAL_IDS, DatasetSample, EvalId, EvalSample
from pipette_scores.parquet_io import read_parquet

# All VLM evals share one sample shape (VlmSample) and one loader (_vlm_common).
_VLM_EVALS = {
    EvalId.MME,
    EvalId.MMSTAR,
    EvalId.OCRBENCH,
    EvalId.MMMU_VAL,
    EvalId.REALWORLDQA,
    EvalId.MMBENCH_EN_DEV,
    EvalId.POPE,
}


def is_vlm_eval(eval_id: str) -> bool:
    """Whether ``eval_id`` names one of the VLM evals."""
    return eval_id in _VLM_EVALS


# ifstruct stores raw rows as JSONL (schema has nested dicts and a Union[int, list[int]] field
# that does not round-trip cleanly through parquet); everything else uses parquet.
_JSONL_EVALS = {EvalId.IFSTRUCT}

logger = logging.getLogger(__name__)

# When unpacked in the workspace layout, the file is at
# `<repo>/packages/pipette-scores/pipette_scores/dataset_catalog/__init__.py`, and
# `datasets/` lives at `<repo>/datasets/` (shared across packages). Climb five
# parents: __init__.py → dataset_catalog → pipette_scores → scores → packages → <repo>.
_DEFAULT_ROOT = pathlib.Path(
    os.environ.get(
        "PIPETTE_SCORES_DATA_DIR",
        pathlib.Path(__file__).resolve().parents[4],
    )
)


def _resolve_root(root_dir: pathlib.Path | str | None) -> pathlib.Path:
    if root_dir is not None:
        return pathlib.Path(root_dir)
    return _DEFAULT_ROOT


def _dataset_dir(eval_id: str, dataset_name: str, root_dir: pathlib.Path | str | None = None) -> pathlib.Path:
    return _resolve_root(root_dir) / "datasets" / eval_id / dataset_name


def list_dataset_names(
    eval_id: EvalId,
    root_dir: pathlib.Path | str | None = None,
) -> list[str]:
    eval_dir = _resolve_root(root_dir) / "datasets" / eval_id
    if not eval_dir.is_dir():
        return []
    return sorted(d.name for d in eval_dir.iterdir() if d.is_dir())


# Datasets are immutable build artifacts and sample IDs are recomputed (sha256) on every
# load, so loaded samples are memoized per (eval_id, dataset_name, root_dir). Calibration
# tools that write datasets always target a *new* dataset name, never the source they load,
# so the cache never serves stale data. Returned lists are treated as read-only by callers.
@functools.lru_cache(maxsize=128)
def load_eval_samples(
    eval_id: EvalId,
    dataset_name: str,
    root_dir: pathlib.Path | str | None = None,
) -> list[DatasetSample]:
    d = _dataset_dir(eval_id, dataset_name, root_dir)
    if not d.exists():
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found for {eval_id}: {d}")
    if eval_id in _VLM_EVALS:
        return _vlm_common.load_eval_samples(d)
    if eval_id in ASR_EVAL_IDS:
        return _asr_common.load_eval_samples(d)
    return {
        EvalId.IFBENCH: ifbench.load_eval_samples,
        EvalId.IFSTRUCT: ifstruct.load_eval_samples,
        EvalId.GPQA_DIAMOND: gpqa_diamond.load_eval_samples,
        EvalId.MATH_500: math_500.load_eval_samples,
    }[eval_id](d)


@functools.lru_cache(maxsize=128)
def load_prompt_samples(
    eval_id: EvalId,
    dataset_name: str,
    root_dir: pathlib.Path | str | None = None,
    prompt_profile: str | None = None,
) -> list[EvalSample]:
    """``prompt_profile`` names a model-specific prompt (prompt_profiles.PROFILES);
    only the VLM evals have them. Unknown names raise ValueError. An ASR eval's
    prompts here carry no audio, each audio part a placeholder with its format
    and no data (see iter_asr_prompt_samples for the requests)."""
    if eval_id in ASR_EVAL_IDS:
        if prompt_profile is None:
            raise ValueError(f"{eval_id} needs an ASR prompt profile ({', '.join(asr_prompt_profiles.PROFILES)})")
        return _asr_common.load_prompt_text_samples(
            EvalId(eval_id), _dataset_dir(eval_id, dataset_name, root_dir), prompt_profile
        )
    if prompt_profile is not None and eval_id not in prompt_profiles.BENCHMARKS:
        raise ValueError(f"{eval_id} has no prompt profiles; they apply to the VLM evals only")
    build = prompt_profiles.profile(prompt_profile) if prompt_profile is not None else None
    d = _dataset_dir(eval_id, dataset_name, root_dir)
    if not d.exists():
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found for {eval_id}: {d}")
    if eval_id in _VLM_EVALS:
        return _vlm_common.load_prompt_samples(
            d,
            interleave_images=eval_id == EvalId.MMMU_VAL,
            prompt=functools.partial(build, EvalId(eval_id)) if build else None,
            system=prompt_profiles.SYSTEM_PROMPTS.get(prompt_profile) if prompt_profile else None,
            prefill=prompt_profiles.PREFILLS.get(prompt_profile) if prompt_profile else None,
        )
    return {
        EvalId.IFBENCH: ifbench.load_prompt_samples,
        EvalId.IFSTRUCT: ifstruct.load_prompt_samples,
        EvalId.GPQA_DIAMOND: gpqa_diamond.load_prompt_samples,
        EvalId.MATH_500: math_500.load_prompt_samples,
    }[eval_id](d)


def iter_asr_prompt_samples(
    eval_id: EvalId,
    dataset_name: str,
    prompt_profile: str,
    root_dir: pathlib.Path | str | None = None,
) -> Iterator[EvalSample]:
    """An ASR eval's requests, with their audio, one at a time in served order
    (_asr_common.iter_prompt_samples). Not cached: a set's audio is gigabytes.
    The dataset is found and verified before this returns."""
    d = _dataset_dir(eval_id, dataset_name, root_dir)
    if not d.exists():
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found for {eval_id}: {d}")
    return _asr_common.iter_prompt_samples(EvalId(eval_id), d, prompt_profile)


def asr_prompt_digests(
    eval_id: EvalId, dataset_name: str, root_dir: pathlib.Path | str | None = None
) -> dict[str, tuple[str, int]]:
    """Every ASR profile's samples digest and count for one dataset, in one
    pass (_asr_common.prompt_digests)."""
    d = _dataset_dir(eval_id, dataset_name, root_dir)
    if not d.exists():
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found for {eval_id}: {d}")
    return _asr_common.prompt_digests(EvalId(eval_id), d)


def load_raw_rows(eval_id: EvalId, dataset_name: str, split: str) -> list[dict]:
    """Load raw rows for a named dataset (parquet, or jsonl for ifstruct)."""
    ext = "jsonl" if eval_id in _JSONL_EVALS else "parquet"
    path = _dataset_dir(eval_id, dataset_name) / f"{split}.{ext}"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if ext == "jsonl":
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
    return read_parquet(path)


def save_raw_split(eval_id: EvalId, dataset_name: str, split: str, rows: list[dict]):
    """Save raw dict rows (parquet, or jsonl for ifstruct)."""
    ext = "jsonl" if eval_id in _JSONL_EVALS else "parquet"
    path = _dataset_dir(eval_id, dataset_name) / f"{split}.{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if ext == "jsonl":
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    else:
        pq.write_table(pa.Table.from_pylist(rows), path)
    logger.info("Wrote %d rows to %s", len(rows), path)
