"""Eval definition types and public dataset catalog interface."""

import functools
import json
import logging
import os
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq

from pipette_scores.dataset_catalog import gpqa_diamond, ifbench, ifstruct, math_500
from pipette_scores.types import DatasetSample, EvalId, EvalSample
from pipette_scores.parquet_io import read_parquet

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
) -> list[EvalSample]:
    d = _dataset_dir(eval_id, dataset_name, root_dir)
    if not d.exists():
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found for {eval_id}: {d}")
    return {
        EvalId.IFBENCH: ifbench.load_prompt_samples,
        EvalId.IFSTRUCT: ifstruct.load_prompt_samples,
        EvalId.GPQA_DIAMOND: gpqa_diamond.load_prompt_samples,
        EvalId.MATH_500: math_500.load_prompt_samples,
    }[eval_id](d)


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
