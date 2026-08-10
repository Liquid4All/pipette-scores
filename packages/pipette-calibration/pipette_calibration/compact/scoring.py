"""Self-contained scoring: load completions per model, build pass-vector matrix.

Deliberately duplicated from calibration.representative so we can iterate on
the compaction pipeline without touching the production path. Difference:
this one correctly unpacks `scoring.score()`'s `(scores, context)` tuple.
"""

import concurrent.futures
import functools
import json
import logging
import multiprocessing
import operator
import os
import pathlib
import time

# Use fork so subprocesses inherit the parent instead of re-executing the entry
# script (lets callers omit the `if __name__ == "__main__"` guard).
_MP_CTX = multiprocessing.get_context("fork")

logger = logging.getLogger(__name__)


def _load_completion_lines(results_dir: pathlib.Path) -> list[str]:
    merged = results_dir / "completions.jsonl"
    shards = sorted(results_dir.glob("completions.shard*.jsonl"))
    if merged.exists() and shards:
        raise RuntimeError(f"Both {merged.name} and {len(shards)} shard file(s) in {results_dir}; remove one")
    if shards:
        return functools.reduce(operator.add, (s.read_text().splitlines() for s in shards))
    return merged.read_text().splitlines()


def _score_one(
    eval_id: str,
    source_dataset: str,
    results_dir: pathlib.Path,
) -> dict[str, bool]:
    """Score one model's completions. Runs in a ProcessPoolExecutor subprocess."""
    import traceback

    from pipette_scores import scoring
    from pipette_scores.dataset_catalog import load_eval_samples
    from pipette_scores.types import SampleCompletion

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(process)d:%(name)s:%(message)s")
    logging.getLogger("pylatexenc").setLevel(logging.ERROR)

    model_name = results_dir.parent.name
    try:
        samples = load_eval_samples(eval_id, source_dataset)
        lines = _load_completion_lines(results_dir)
        completions = [SampleCompletion(**json.loads(line)) for line in lines]

        scores, _context = scoring.score(eval_id, completions, samples, label=model_name)
        return {s.id: s.is_correct for s in scores}
    except Exception:
        logging.getLogger(__name__).error(
            "Scoring failed for %s in pid %d:\n%s", model_name, os.getpid(), traceback.format_exc()
        )
        raise


def _discover(results_path: pathlib.Path, eval_id: str, source_dataset: str) -> list[pathlib.Path]:
    eval_dir = results_path / eval_id
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"no results dir: {eval_dir}")
    return sorted(d / source_dataset for d in eval_dir.iterdir() if d.is_dir() and (d / source_dataset).is_dir())


def build_pass_vectors(
    eval_id: str,
    source_dataset: str,
    results_path: pathlib.Path,
) -> tuple[dict[str, tuple[bool, ...]], list[str], list[dict]]:
    """Score all model dirs in parallel. Returns (pass_vectors, kept_model_names, skipped)."""
    from pipette_scores.dataset_catalog import load_eval_samples

    dirs = _discover(results_path, eval_id, source_dataset)
    names = [d.parent.name for d in dirs]
    logger.info("Scoring %d model(s) for %s/%s", len(dirs), eval_id, source_dataset)

    samples = load_eval_samples(eval_id, source_dataset)
    # Sorted so iteration order (and therefore pass_vectors key order, the
    # selector's cell construction, and everything downstream) is deterministic
    # across Python processes — otherwise set iteration depends on
    # PYTHONHASHSEED and two runs with identical args produce different
    # subsets.
    all_ids = sorted(s.id for s in samples)
    all_ids_set = set(all_ids)

    max_workers = min(len(dirs), (os.cpu_count() or 1))
    t0 = time.monotonic()
    args = [(eval_id, source_dataset, d) for d in dirs]
    results_map: dict[str, dict[str, bool] | None] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=_MP_CTX) as pool:
        futures = {pool.submit(_score_one, *a): name for a, name in zip(args, names)}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            name = futures[fut]
            try:
                results_map[name] = fut.result()
                logger.info("  %d/%d %s", i, len(futures), name)
            except Exception:
                results_map[name] = None
                logger.exception("  %d/%d %s FAILED", i, len(futures), name)

    per_model: list[dict[str, bool]] = []
    kept: list[str] = []
    skipped: list[dict] = []
    for name in names:
        r = results_map.get(name)
        if r is None:
            skipped.append({"model": name, "reason": "scoring failed"})
            continue
        missing = all_ids_set - r.keys()
        if missing:
            skipped.append({"model": name, "reason": "incomplete", "missing": len(missing)})
            continue
        if sum(1 for sid in all_ids if r.get(sid, False)) == 0:
            skipped.append({"model": name, "reason": "zero accuracy"})
            continue
        per_model.append(r)
        kept.append(name)

    logger.info("Scored in %.1fs — %d kept, %d skipped", time.monotonic() - t0, len(kept), len(skipped))
    if not per_model:
        raise RuntimeError("no models with complete results")

    pass_vectors = {sid: tuple(m[sid] for m in per_model) for sid in all_ids}
    return pass_vectors, kept, skipped
