"""Orchestrator: score → filter chain → save."""

import json
import logging
import pathlib

from pipette_scores.dataset_catalog import load_eval_samples, load_raw_rows, save_raw_split
from pipette_scores.types import EvalId

from pipette_calibration.calibration import GROUP_OF, PRIMARY_SPLIT, SPLITS
from pipette_calibration.calibration.representative import (
    held_out_cv_metrics,
    select_representative_samples,
)
from pipette_calibration.compact.filters import dedup_behavior, unanimous
from pipette_calibration.compact.scoring import build_pass_vectors

logger = logging.getLogger(__name__)

FILTERS = [unanimous, dedup_behavior]


def _write_metadata(
    *,
    eval_id: str,
    dataset: str,
    source_dataset: str,
    num_samples: int,
    model_names: list[str],
    skipped: list[dict],
    stages: list[dict],
    max_deviation: float,
    held_out_cv: dict | None = None,
) -> None:
    from pipette_scores.dataset_catalog import _dataset_dir

    from pipette_calibration.inference import spec_for_metadata

    compaction: dict = {"max_deviation_in_sample": max_deviation, "stages": stages}
    if held_out_cv is not None:
        compaction["held_out_cv"] = held_out_cv
    meta = {
        "eval_id": eval_id,
        "dataset": dataset,
        "source_dataset": source_dataset,
        "num_samples": num_samples,
        "models": model_names,
        "skipped": skipped,
        "generation_params": spec_for_metadata(eval_id),
        "compaction": compaction,
    }
    path = _dataset_dir(eval_id, dataset) / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n")


def compact_dataset(
    *,
    eval_id: EvalId,
    source_dataset: str,
    results_path: pathlib.Path,
    new_dataset: str,
    target_size: int | None = None,
    cv_folds: int = 5,
    cv_seed: int = 0,
) -> None:
    """Score completions, apply filter chain, save the filtered pool.

    When ``target_size`` is set and the filter chain leaves more samples than
    requested, ``select_representative_samples`` is run as a final stage to
    shrink to exactly ``target_size`` items. The resulting dataset's
    ``metadata.json`` records in-sample drift and — when ``cv_folds > 0`` —
    held-out CV drift with ``cv_seed`` controlling the fold permutation.
    """
    logger.info(
        "compacting %s/%s → %s (target_size=%s, cv_folds=%d, cv_seed=%d)",
        eval_id,
        source_dataset,
        new_dataset,
        target_size if target_size is not None else "none",
        cv_folds,
        cv_seed,
    )
    pass_vectors, model_names, skipped = build_pass_vectors(eval_id, source_dataset, results_path)
    samples = load_eval_samples(eval_id, source_dataset)
    n_models = len(model_names)
    logger.info("loaded %d samples × %d models (%d models skipped)", len(pass_vectors), n_models, len(skipped))

    stages: list[dict] = []
    current = pass_vectors
    for fn in FILTERS:
        name = fn.__name__.replace("_", "-")
        before = len(current)
        current = fn(current)
        after = len(current)
        logger.info("stage %s: %d → %d (−%d)", name, before, after, before - after)
        stages.append({"stage": name, "before": before, "after": after})

    # Optional third stage: representative subset selection.
    held_out_cv: dict | None = None
    if target_size is not None and len(current) > target_size:
        before = len(current)
        group_of = GROUP_OF[eval_id]
        groups = {s.id: group_of(s) for s in samples if s.id in current}
        # Honest held-out CV against the pool the selector sees. Run this
        # before selection so the selector re-fits per fold on the training
        # model panel — a subset picked against the full panel would degenerate
        # the CV estimate (zero drift because the target equals the input).
        # The CV runs the selector ``cv_folds`` additional times (each ~one
        # selector invocation). Pass ``cv_folds=0`` to skip.
        if cv_folds > 0:
            held_out_cv = held_out_cv_metrics(current, groups, target_size, n_folds=cv_folds, seed=cv_seed)
            logger.info(
                "held-out CV (k=%d folds, seed=%d): mean drift %.2f%%, max %.2f%%, τ=%.3f, top5=%.2f",
                cv_folds,
                cv_seed,
                held_out_cv["mean_deviation"] * 100,
                held_out_cv["max_deviation"] * 100,
                held_out_cv["kendall_tau"],
                held_out_cv["top5_agreement"],
            )
        selected_ids = select_representative_samples(current, groups, target_size)
        current = {sid: current[sid] for sid in selected_ids}
        after = len(current)
        logger.info("stage representative: %d → %d (−%d)", before, after, before - after)
        stages.append({"stage": "representative", "before": before, "after": after})

    kept_ids = set(current)
    full_acc = [sum(p[m] for p in pass_vectors.values()) / len(pass_vectors) for m in range(n_models)]
    sub_acc = [sum(pass_vectors[sid][m] for sid in kept_ids) / len(kept_ids) for m in range(n_models)]
    max_dev = max(abs(f - s) for f, s in zip(full_acc, sub_acc))
    logger.info("output: %d samples, in-sample max per-model drift %.2f%%", len(kept_ids), max_dev * 100)

    primary = PRIMARY_SPLIT[eval_id]
    raw = load_raw_rows(eval_id, source_dataset, primary)
    kept_rows = [row for sample, row in zip(samples, raw) if sample.id in kept_ids]
    save_raw_split(eval_id, new_dataset, primary, kept_rows)
    for split in SPLITS[eval_id]:
        if split != primary:
            save_raw_split(eval_id, new_dataset, split, load_raw_rows(eval_id, source_dataset, split))

    _write_metadata(
        eval_id=eval_id,
        dataset=new_dataset,
        source_dataset=source_dataset,
        num_samples=len(kept_rows),
        model_names=model_names,
        skipped=skipped,
        stages=stages,
        max_deviation=max_dev,
        held_out_cv=held_out_cv,
    )
    logger.info("saved datasets/%s/%s/ (%d samples)", eval_id, new_dataset, len(kept_rows))
