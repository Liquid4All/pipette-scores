"""Select a representative subset where each model's subset accuracy closely matches its full-set accuracy."""

import concurrent.futures
import functools
import heapq
import json
import logging
import operator
import os
import pathlib
import random
import statistics
import time
from collections import defaultdict

from pipette_scores.dataset_catalog import load_eval_samples, load_raw_rows, save_raw_split
from pipette_scores.types import EvalId

from pipette_calibration.calibration import GROUP_OF, PRIMARY_SPLIT, SPLITS

logger = logging.getLogger(__name__)


def load_completions(results_dir: pathlib.Path) -> list[str]:
    merged_path = results_dir / "completions.jsonl"
    shards = sorted(results_dir.glob("completions.shard*.jsonl"))

    if merged_path.exists() and shards:
        raise RuntimeError(
            f"Both {merged_path.name} and {len(shards)} shard file(s) exist in {results_dir}; "
            "remove one before continuing"
        )

    if shards:
        lines = functools.reduce(operator.add, (shard.read_text().splitlines() for shard in shards))
        logger.info("Loaded %d shards (%d lines) from %s", len(shards), len(lines), results_dir)
        return lines

    return merged_path.read_text().splitlines()


def score_completions(eval_id: str, results_dir: pathlib.Path, samples: list) -> dict[str, bool]:
    from pipette_scores import scoring
    from pipette_scores.types import SampleCompletion

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(process)d:%(name)s:%(message)s")
    logging.getLogger("pylatexenc").setLevel(logging.ERROR)
    model_name = results_dir.parent.name

    t0 = time.monotonic()
    lines = load_completions(results_dir)
    logger.info("[%s] Loaded %d lines (%.1fs)", model_name, len(lines), time.monotonic() - t0)

    t1 = time.monotonic()
    completions = [SampleCompletion(**json.loads(line)) for line in lines]
    logger.info("[%s] Parsed %d completions (%.1fs)", model_name, len(completions), time.monotonic() - t1)

    t2 = time.monotonic()
    scores, _context = scoring.score(eval_id, completions, samples, label=model_name)
    logger.info("[%s] Scored %d samples (%.1fs)", model_name, len(scores), time.monotonic() - t2)

    return {s.id: s.is_correct for s in scores}


def _score_one_model(
    eval_id: str,
    source_dataset: str,
    results_dir: pathlib.Path,
) -> dict[str, bool]:
    """Picklable wrapper for score_completions, used by ProcessPoolExecutor.

    Loads samples inside the subprocess to avoid pickling the full sample list
    across the process boundary for every model.
    """
    import traceback

    from pipette_scores.dataset_catalog import load_eval_samples

    model_name = results_dir.parent.name
    try:
        samples = load_eval_samples(eval_id, source_dataset)
        return score_completions(eval_id, results_dir, samples)
    except Exception:
        # Log inside the subprocess so the traceback is visible even if the
        # process is killed shortly after (e.g. by OOM killer).
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).error(
            "Scoring failed for %s in pid %d:\n%s",
            model_name,
            os.getpid(),
            traceback.format_exc(),
        )
        raise


def discover_model_results(
    results_path: pathlib.Path,
    eval_id: str,
    source_dataset: str,
) -> list[pathlib.Path]:
    """Return model result dirs under ``results_path/eval_id/*/`` that contain a ``source_dataset/`` subdirectory."""
    eval_dir = results_path / eval_id
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Eval directory not found: {eval_dir}")

    dirs: list[pathlib.Path] = []
    for model_dir in sorted(eval_dir.iterdir()):
        dataset_dir = model_dir / source_dataset
        if dataset_dir.is_dir():
            dirs.append(dataset_dir)

    if not dirs:
        raise FileNotFoundError(f"No model results found under {eval_dir} with source_dataset={source_dataset!r}")
    return dirs


def _max_deviation(
    correct: list[int],
    total: int,
    full_acc: list[float],
    n_models: int,
) -> float:
    """Return max |subset_acc - full_acc| across all models."""
    if total == 0:
        return float("inf")
    return max(abs(correct[m] / total - full_acc[m]) for m in range(n_models))


# ---------------------------------------------------------------------------
# Representative subset selector
# ---------------------------------------------------------------------------
#
# Stratifies samples on two axes — entity × difficulty-bin (by per-entity
# pass-count quantile) — and minimises max |subset_acc − full_acc| across all
# panel models via greedy + within-cell local search with seeded random
# tie-break. Measured on ifstruct/release_v1_0 at num_samples=288: held-out mean
# drift 1.49%, max 4.91%, Kendall τ 0.949, top-5 preserved (5-fold CV across
# 52 models).


def _full_accuracy(pass_vectors: dict[str, tuple[bool, ...]], n_models: int) -> list[float]:
    n = len(pass_vectors)
    total = [0] * n_models
    for pv in pass_vectors.values():
        for m in range(n_models):
            total[m] += int(pv[m])
    return [t / n for t in total]


def _build_cells(
    pass_vectors: dict[str, tuple[bool, ...]],
    groups: dict[str, str],
    n_difficulty_bins: int,
) -> dict[tuple[str, int], list[str]]:
    """Return {(entity, difficulty_bin): [sid, ...]} via per-entity rank split."""
    by_entity: dict[str, list[str]] = defaultdict(list)
    pass_counts = {sid: sum(pv) for sid, pv in pass_vectors.items()}
    for sid in pass_vectors:
        by_entity[groups[sid]].append(sid)

    cells: dict[tuple[str, int], list[str]] = {}
    for e, sids in by_entity.items():
        sids_sorted = sorted(sids, key=lambda s: (pass_counts[s], s))
        n = len(sids_sorted)
        if n == 0:
            continue
        bin_sizes = [n // n_difficulty_bins] * n_difficulty_bins
        for i in range(n % n_difficulty_bins):
            bin_sizes[-1 - i] += 1
        idx = 0
        for b, sz in enumerate(bin_sizes):
            cell = sids_sorted[idx : idx + sz]
            idx += sz
            if cell:
                cells[(e, b)] = cell
    return cells


def _allocate_quotas(cells: dict[tuple[str, int], list[str]], num_samples: int) -> dict[tuple[str, int], int]:
    """Proportional quota per cell with floor of 1 and Hamilton remainder."""
    cell_sizes = {k: len(v) for k, v in cells.items()}
    total = sum(cell_sizes.values())
    ideal = {k: sz / total * num_samples for k, sz in cell_sizes.items()}
    quota = {k: max(1, int(ideal[k])) for k in cells}
    for k in quota:
        quota[k] = min(quota[k], cell_sizes[k])
    allocated = sum(quota.values())
    while allocated < num_samples:
        candidates = [(ideal[k] - int(ideal[k]), k) for k in cells if quota[k] < cell_sizes[k]]
        if not candidates:
            break
        candidates.sort(reverse=True)
        for _, k in candidates:
            if allocated >= num_samples:
                break
            quota[k] += 1
            allocated += 1
    while allocated > num_samples:
        candidates = [(ideal[k] - int(ideal[k]), k) for k in cells if quota[k] > 1]
        if not candidates:
            break
        candidates.sort()
        for _, k in candidates:
            if allocated <= num_samples:
                break
            quota[k] -= 1
            allocated -= 1
    return quota


def _greedy_within_cells(
    cells: dict[tuple[str, int], list[str]],
    quotas: dict[tuple[str, int], int],
    pass_vectors: dict[str, tuple[bool, ...]],
    n_models: int,
    full_acc: list[float],
    rng: random.Random,
) -> list[str]:
    selected: list[str] = []
    correct = [0] * n_models
    cell_keys = list(cells.keys())
    rng.shuffle(cell_keys)
    for cell_key in cell_keys:
        quota = quotas[cell_key]
        if quota == 0:
            continue
        pool = list(cells[cell_key])
        rng.shuffle(pool)
        for _ in range(quota):
            total = len(selected) + 1
            best = float("inf")
            ties: list[str] = []
            for sid in pool:
                pv = pass_vectors[sid]
                dev = 0.0
                for m in range(n_models):
                    d = abs((correct[m] + int(pv[m])) / total - full_acc[m])
                    if d > dev:
                        dev = d
                if dev < best:
                    best = dev
                    ties = [sid]
                elif dev == best:
                    ties.append(sid)
            chosen = rng.choice(ties)
            selected.append(chosen)
            pool.remove(chosen)
            pv = pass_vectors[chosen]
            for m in range(n_models):
                correct[m] += int(pv[m])
    return selected


def _local_search_within_cells(
    selected: list[str],
    cells: dict[tuple[str, int], list[str]],
    pass_vectors: dict[str, tuple[bool, ...]],
    n_models: int,
    full_acc: list[float],
    rng: random.Random,
    max_rounds: int = 20,
) -> list[str]:
    sid_to_cell: dict[str, tuple[str, int]] = {}
    for k, sids in cells.items():
        for sid in sids:
            sid_to_cell[sid] = k

    selected = list(selected)
    total = len(selected)
    correct = [0] * n_models
    for sid in selected:
        pv = pass_vectors[sid]
        for m in range(n_models):
            correct[m] += int(pv[m])

    for _rnd in range(max_rounds):
        improved = False
        order = list(range(len(selected)))
        rng.shuffle(order)
        for i in order:
            old_id = selected[i]
            cell = sid_to_cell[old_id]
            old_pv = pass_vectors[old_id]
            trial = [correct[m] - int(old_pv[m]) for m in range(n_models)]
            cur_dev = _max_deviation([trial[m] + int(old_pv[m]) for m in range(n_models)], total, full_acc, n_models)
            selected_set = set(selected)
            cell_items = [s for s in cells[cell] if s not in selected_set]
            best_new: str | None = None
            best_dev = cur_dev
            ties: list[str] = []
            for new_id in cell_items:
                new_pv = pass_vectors[new_id]
                sw = [trial[m] + int(new_pv[m]) for m in range(n_models)]
                sd = _max_deviation(sw, total, full_acc, n_models)
                if sd < best_dev:
                    best_dev = sd
                    best_new = new_id
                    ties = [new_id]
                elif sd == best_dev and best_new is not None:
                    ties.append(new_id)
            if best_new is not None and best_dev < cur_dev - 1e-12:
                chosen = rng.choice(ties) if ties else best_new
                new_pv = pass_vectors[chosen]
                selected[i] = chosen
                for m in range(n_models):
                    correct[m] = trial[m] + int(new_pv[m])
                improved = True
        if not improved:
            break
    return selected


def select_representative_samples(
    pass_vectors: dict[str, tuple[bool, ...]],
    groups: dict[str, str],
    num_samples: int,
    *,
    n_difficulty_bins: int = 3,
    seed: int = 0,
) -> list[str]:
    """Pick a representative subset via (entity × difficulty-bin) stratification.

    Minimises max |subset_acc − full_acc| across models by greedy + within-cell
    local search with seeded random tie-break. Group cap is derived from
    ``groups`` (e.g. ``entity_type``); difficulty binning is per-entity rank
    split into ``n_difficulty_bins`` bins by pass count.
    """
    n_models = len(next(iter(pass_vectors.values())))
    rng = random.Random(seed)
    cells = _build_cells(pass_vectors, groups, n_difficulty_bins)
    quotas = _allocate_quotas(cells, num_samples)
    full_acc = _full_accuracy(pass_vectors, n_models)
    selected = _greedy_within_cells(cells, quotas, pass_vectors, n_models, full_acc, rng)
    selected = _local_search_within_cells(selected, cells, pass_vectors, n_models, full_acc, rng)
    return selected


def held_out_cv_metrics(
    pass_vectors: dict[str, tuple[bool, ...]],
    groups: dict[str, str],
    num_samples: int,
    *,
    selector=select_representative_samples,
    n_folds: int = 5,
    seed: int = 0,
) -> dict:
    """Honest held-out drift via k-fold CV over the model panel.

    For each fold: refit ``selector`` on the training-panel pass-vectors,
    measure drift on held-out models. Returns aggregate metrics.

    ``seed`` controls only the fold-permutation randomness, not any selector
    internals. The selector is called positionally — if you need to pass
    ``seed`` or other kwargs to the selector itself, wrap it with
    ``functools.partial`` before passing.
    """
    n_models = len(next(iter(pass_vectors.values())))
    full_acc = _full_accuracy(pass_vectors, n_models)
    rng = random.Random(seed)
    perm = list(range(n_models))
    rng.shuffle(perm)
    # Even split into n_folds, remainder spread across the last folds.
    base, extra = divmod(n_models, n_folds)
    folds: list[list[int]] = []
    cursor = 0
    for fi in range(n_folds):
        sz = base + (1 if fi >= n_folds - extra else 0)
        folds.append(perm[cursor : cursor + sz])
        cursor += sz

    held_est = [0.0] * n_models
    for test_cols in folds:
        train_cols = [m for m in range(n_models) if m not in set(test_cols)]
        pv_train = {sid: tuple(pv[m] for m in train_cols) for sid, pv in pass_vectors.items()}
        selected = selector(pv_train, groups, num_samples)
        for tm in test_cols:
            acc = sum(int(pass_vectors[sid][tm]) for sid in selected) / len(selected)
            held_est[tm] = acc

    drifts = [abs(held_est[m] - full_acc[m]) for m in range(n_models)]
    tau = _kendall_tau(full_acc, held_est)
    top5_full = {i for i, _ in heapq.nlargest(5, enumerate(full_acc), key=lambda t: t[1])}
    top5_held = {i for i, _ in heapq.nlargest(5, enumerate(held_est), key=lambda t: t[1])}

    return {
        "cv_folds": n_folds,
        "cv_seed": seed,
        "mean_deviation": statistics.fmean(drifts),
        "max_deviation": max(drifts),
        "kendall_tau": tau,
        "top5_agreement": len(top5_full & top5_held) / 5.0,
    }


def _kendall_tau(x: list[float], y: list[float]) -> float:
    """Kendall rank correlation (simple O(n²) form).

    Hand-rolled because ``scipy`` is not a direct dependency of
    ``pipette-calibration`` and n is small (model-panel size, typically ≤100).
    """
    n = len(x)
    num = den = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (x[i] - x[j]) * (y[i] - y[j])
            if s > 0:
                num += 1
            elif s < 0:
                num -= 1
            den += 1
    return num / den if den else 0.0


def _log_comparison(
    selected: list[str],
    pass_vectors: dict[str, tuple[bool, ...]],
    model_names: list[str],
) -> None:
    n_models = len(model_names)
    n_total = len(pass_vectors)

    full_correct = [0] * n_models
    for pv in pass_vectors.values():
        for m in range(n_models):
            full_correct[m] += int(pv[m])

    sub_correct = [0] * n_models
    for sid in selected:
        pv = pass_vectors[sid]
        for m in range(n_models):
            sub_correct[m] += int(pv[m])

    logger.info("")
    logger.info("%-40s %10s %10s %10s", "Model", "Full Acc", "Sub Acc", "Deviation")
    logger.info("-" * 72)
    max_dev = 0.0
    for m in range(n_models):
        fa = full_correct[m] / n_total
        sa = sub_correct[m] / len(selected)
        dev = abs(sa - fa)
        max_dev = max(max_dev, dev)
        logger.info("%-40s %9.1f%% %9.1f%% %9.2f%%", model_names[m], fa * 100, sa * 100, dev * 100)
    logger.info("-" * 72)
    logger.info("Max deviation: %.2f%%", max_dev * 100)


def _write_metadata(
    *,
    eval_id: str,
    dataset: str,
    source_dataset: str,
    num_samples: int,
    model_names: list[str],
    skipped: list[dict],
) -> None:
    """Write a metadata JSON file alongside the dataset."""
    from pipette_scores.dataset_catalog import _dataset_dir

    from pipette_calibration.inference import spec_for_metadata

    metadata = {
        "eval_id": eval_id,
        "dataset": dataset,
        "source_dataset": source_dataset,
        "num_samples": num_samples,
        "models": model_names,
        "skipped": skipped,
        "generation_params": spec_for_metadata(eval_id),
    }

    meta_path = _dataset_dir(eval_id, dataset) / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("Wrote metadata to %s", meta_path)


def create_representative_dataset(
    eval_id: EvalId,
    source_dataset: str,
    dataset: str,
    results_path: pathlib.Path,
    num_samples: int = 200,
) -> int:
    """Discover models, load results, select representative samples, save dataset. Returns count."""
    # Discover model result directories.
    model_dirs = discover_model_results(results_path, eval_id, source_dataset)
    model_names = [d.parent.name for d in model_dirs]
    logger.info(
        "Discovered %d model(s) under %s/%s: %s",
        len(model_dirs),
        results_path,
        eval_id,
        ", ".join(model_names),
    )

    # Build group map first so we know the full sample set.
    group_of = GROUP_OF[eval_id]
    samples = load_eval_samples(eval_id, source_dataset)
    all_sample_ids = {s.id for s in samples}
    groups = {s.id: group_of(s) for s in samples}

    # Load per-model results in parallel and filter out models with incomplete results.
    max_workers = min(len(model_dirs), (os.cpu_count() or 1))
    t0 = time.monotonic()
    logger.info("Scoring %d model(s) with %d parallel workers...", len(model_dirs), max_workers)

    _score_args = [(eval_id, source_dataset, d) for d in model_dirs]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score_one_model, *a): name for a, name in zip(_score_args, model_names)}
        results_map: dict[str, dict[str, bool] | None] = {}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name = futures[future]
            try:
                results_map[name] = future.result()
                logger.info("  Scored %d/%d: %s", i, len(futures), name)
            except concurrent.futures.process.BrokenProcessPool:
                results_map[name] = None
                logger.error(
                    "  Failed %d/%d: %s — worker process was killed",
                    i,
                    len(futures),
                    name,
                )
                break  # pool is broken, remaining futures will also fail
            except Exception:
                results_map[name] = None
                logger.exception("  Failed %d/%d: %s", i, len(futures), name)

    per_model: list[dict[str, bool]] = []
    kept_names: list[str] = []
    skipped: list[dict] = []
    for name in model_names:
        results = results_map.get(name)
        if results is None:
            skipped.append({"model": name, "reason": "scoring failed"})
            logger.warning("Excluding %s: scoring failed", name)
            continue
        missing_ids = all_sample_ids - results.keys()
        extra_ids = results.keys() - all_sample_ids
        if missing_ids:
            skipped.append(
                {
                    "model": name,
                    "reason": "incomplete results",
                    "have": len(results.keys() & all_sample_ids),
                    "missing": len(missing_ids),
                    "total": len(all_sample_ids),
                }
            )
            logger.warning(
                "Excluding %s: has %d results, missing %d of %d samples",
                name,
                len(results),
                len(missing_ids),
                len(all_sample_ids),
            )
            continue
        if extra_ids:
            logger.warning(
                "Model %s has %d extra sample IDs not in dataset (ignored)",
                name,
                len(extra_ids),
            )
        # Exclude models that scored 0% — they add no discriminative signal.
        n_correct = sum(1 for sid in all_sample_ids if results.get(sid, False))
        if n_correct == 0:
            skipped.append({"model": name, "reason": "zero accuracy"})
            logger.warning("Excluding %s: 0%% accuracy (no correct answers)", name)
            continue
        per_model.append(results)
        kept_names.append(name)
    model_names = kept_names

    logger.info(
        "Scoring completed in %.1fs, %d model(s) kept, %d skipped", time.monotonic() - t0, len(kept_names), len(skipped)
    )

    if not per_model:
        raise RuntimeError("No models with complete results")

    # Build pass vectors over the full sample set.
    pass_vectors: dict[str, tuple[bool, ...]] = {sid: tuple(m[sid] for m in per_model) for sid in all_sample_ids}
    logger.info("Loaded pass vectors for %d samples across %d models", len(pass_vectors), len(per_model))

    # Select representative subset.
    selected_ids_list = select_representative_samples(pass_vectors, groups, num_samples)
    selected_ids = set(selected_ids_list)

    _log_comparison(selected_ids_list, pass_vectors, model_names)

    # Save the new dataset.
    primary_split = PRIMARY_SPLIT[eval_id]
    raw_rows = load_raw_rows(eval_id, source_dataset, primary_split)
    selected_raw = [row for sample, row in zip(samples, raw_rows) if sample.id in selected_ids]
    save_raw_split(eval_id, dataset, primary_split, selected_raw)

    for split_name in SPLITS[eval_id]:
        if split_name != primary_split:
            rows = load_raw_rows(eval_id, source_dataset, split_name)
            save_raw_split(eval_id, dataset, split_name, rows)

    # Write metadata alongside the dataset.
    _write_metadata(
        eval_id=eval_id,
        dataset=dataset,
        source_dataset=source_dataset,
        num_samples=len(selected_raw),
        model_names=model_names,
        skipped=skipped,
    )

    logger.info("Created representative dataset '%s' for %s (%d samples)", dataset, eval_id, len(selected_raw))
    return len(selected_raw)
