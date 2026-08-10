#!/usr/bin/env python3
"""Compact a calibration dataset via the filter chain + representative selector.

Pipeline: unanimous → dedup-behavior → representative (target_size).

Dataset names follow Ubuntu-style ``YYYY.MM.N`` where ``N`` is the release
index within that month and is **computed automatically** from today's date
and the highest existing ``N`` under ``datasets/<eval>/``. Pass
``--dataset-name`` only to override the generated release name.

Run from the workspace root:

    # Auto-named next release, 288 samples:
    uv run packages/pipette-calibration/scripts/compact-ifstruct.py

    # Different size or skip the representative stage:
    uv run packages/pipette-calibration/scripts/compact-ifstruct.py --target-size 360
    uv run packages/pipette-calibration/scripts/compact-ifstruct.py --target-size 0

    # Override the computed name:
    uv run packages/pipette-calibration/scripts/compact-ifstruct.py --dataset-name 2026.07.1
"""

import argparse
import datetime
import logging
import pathlib

from pipette_calibration.compact import compact_dataset
from pipette_scores.dataset_catalog import list_dataset_names
from pipette_scores.types import EvalId


def _next_release_name(eval_id: EvalId, today: datetime.date | None = None) -> str:
    """Compute the next ``YYYY.MM.N`` dataset name for ``eval_id``.

    Scans existing datasets under ``datasets/<eval_id>/`` and returns the
    next N after the highest N that matches today's year.month prefix.
    """
    today = today or datetime.date.today()
    prefix = f"{today.year}.{today.month:02d}."
    existing_ns = []
    for name in list_dataset_names(eval_id):
        if name.startswith(prefix):
            tail = name[len(prefix) :]
            if tail.isdigit():
                existing_ns.append(int(tail))
    return f"{prefix}{max(existing_ns, default=0) + 1}"


p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument(
    "--dataset-name",
    default=None,
    help="Override the auto-computed YYYY.MM.N dataset name.",
)
p.add_argument("--eval", type=EvalId, default=EvalId.IFSTRUCT, choices=list(EvalId))
p.add_argument("--source-dataset", default="release_v1_0")
p.add_argument("--results-path", type=pathlib.Path, default=pathlib.Path("_calibration_results"))
p.add_argument(
    "--target-size",
    type=int,
    default=288,
    help="Size of the representative subset. Pass 0 to skip the representative stage entirely.",
)
p.add_argument(
    "--cv-folds",
    type=int,
    default=5,
    help="K in k-fold CV for held-out drift reporting. Pass 0 to skip CV. Ignored when --target-size 0.",
)
p.add_argument(
    "--cv-seed",
    type=int,
    default=0,
    help="Seed for model-panel fold permutation in held-out CV. Recorded in metadata.",
)
args = p.parse_args()

logging.basicConfig(level=logging.INFO)

dataset_name = args.dataset_name or _next_release_name(args.eval)
logging.getLogger(__name__).info("compacting %s/%s → %s", args.eval, args.source_dataset, dataset_name)

compact_dataset(
    eval_id=args.eval,
    source_dataset=args.source_dataset,
    results_path=args.results_path,
    new_dataset=dataset_name,
    target_size=args.target_size if args.target_size > 0 else None,
    cv_folds=args.cv_folds,
    cv_seed=args.cv_seed,
)
