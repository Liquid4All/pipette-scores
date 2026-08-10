"""Verify completions against the dataset: check count, sample IDs, duplicates, and empty responses."""

import json
import logging
import pathlib
from dataclasses import dataclass

from pipette_scores import dataset_catalog

logger = logging.getLogger(__name__)


@dataclass
class ModelVerification:
    """Verification result for one model on one eval."""

    model: str
    eval_id: str
    total: int  # expected sample count
    have: int  # completion IDs matching expected
    missing: int  # expected IDs not in completions
    extra: int  # completion IDs not in expected
    duplicates: int  # duplicate IDs across shards
    empty: int  # completions with blank text

    @property
    def ok(self) -> bool:
        return self.missing == 0 and self.extra == 0 and self.duplicates == 0

    @property
    def status(self) -> str:
        if self.have == 0 and self.missing > 0:
            return "missing"
        if self.missing > 0:
            return "incomplete"
        if self.duplicates > 0:
            return "duplicates"
        if self.empty == self.total:
            return "zero-score"
        if self.empty > 0:
            return "partial-empty"
        return "ok"


def _load_completions(dataset_dir: pathlib.Path) -> list[dict] | None:
    """Load all completions from result files in a dataset directory.

    Returns None if no completion files exist.
    """
    merged = dataset_dir / "completions.jsonl"
    shards = sorted(dataset_dir.glob("completions.shard*.jsonl"))

    if not merged.exists() and not shards:
        return None

    paths = shards if shards else [merged]
    completions = []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                completions.append(json.loads(line))
    return completions


def _count_duplicates(completions: list[dict]) -> int:
    """Count duplicate IDs in a completion list."""
    seen: set[str] = set()
    dups = 0
    for c in completions:
        sid = c["id"]
        if sid in seen:
            dups += 1
        else:
            seen.add(sid)
    return dups


def _count_empty(completions: list[dict]) -> int:
    """Count completions with blank text."""
    return sum(1 for c in completions if not c.get("completion", "").strip())


def verify_model(
    eval_id: str,
    dataset_dir: pathlib.Path,
    expected_ids: set[str],
) -> ModelVerification:
    """Verify one model's completions against expected sample IDs."""
    model = dataset_dir.parent.name
    total = len(expected_ids)

    completions = _load_completions(dataset_dir)
    if completions is None:
        return ModelVerification(
            model=model,
            eval_id=eval_id,
            total=total,
            have=0,
            missing=total,
            extra=0,
            duplicates=0,
            empty=0,
        )

    completion_ids = {c["id"] for c in completions}
    missing_ids = expected_ids - completion_ids
    extra_ids = completion_ids - expected_ids
    have = len(completion_ids & expected_ids)
    duplicates = _count_duplicates(completions)
    empty = _count_empty(completions)

    return ModelVerification(
        model=model,
        eval_id=eval_id,
        total=total,
        have=have,
        missing=len(missing_ids),
        extra=len(extra_ids),
        duplicates=duplicates,
        empty=empty,
    )


def verify_eval(
    eval_id: str,
    dataset: str,
    results_path: pathlib.Path,
) -> list[ModelVerification]:
    """Verify all models found on disk for one eval.

    Returns a list of ModelVerification results. Returns an empty list
    if the eval directory does not exist.
    """
    eval_dir = results_path / eval_id
    if not eval_dir.is_dir():
        return []

    model_dirs = sorted(d for d in eval_dir.iterdir() if d.is_dir() and (d / dataset).is_dir())
    if not model_dirs:
        return []

    expected_samples = dataset_catalog.load_eval_samples(eval_id, dataset)
    expected_ids = {s.id for s in expected_samples}

    results = []
    for model_dir in model_dirs:
        dataset_dir = model_dir / dataset
        results.append(verify_model(eval_id, dataset_dir, expected_ids))

    return results


def verify_completions(
    *,
    eval_id: str,
    dataset: str,
    results_path: pathlib.Path,
) -> bool:
    """Check all models for one eval. Returns True when all pass.

    Thin wrapper around verify_eval() for backward compatibility with the CLI.
    """
    results = verify_eval(eval_id, dataset, results_path)
    if not results:
        logger.error("No model results found for %s/%s", eval_id, dataset)
        return False

    logger.info(
        "Verifying %s/%s (%d expected samples, %d models)",
        eval_id,
        dataset,
        results[0].total,
        len(results),
    )

    for v in results:
        if v.ok:
            logger.info("  OK    %-45s  %d/%d", v.model, v.have, v.total)
        else:
            parts = []
            if v.missing:
                parts.append(f"{v.missing} missing")
            if v.extra:
                parts.append(f"{v.extra} extra")
            if v.duplicates:
                parts.append(f"{v.duplicates} duplicates")
            if v.empty:
                parts.append(f"{v.empty} empty")
            logger.error(
                "  FAIL  %-45s  %d/%d  (%s)",
                v.model,
                v.have,
                v.total,
                ", ".join(parts),
            )

    return all(v.ok for v in results)
