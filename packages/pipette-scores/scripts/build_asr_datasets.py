#!/usr/bin/env python
"""Materialize the ASR eval datasets (Open ASR Leaderboard English + FLEURS) from their pinned sources.

Run from the repo root (build deps: ``uv sync --extra build``), writing OUTSIDE
the checkout (the builder refuses a path inside it):

    uv run --project packages/pipette-scores \\
        python packages/pipette-scores/scripts/build_asr_datasets.py \\
        --out /data/pipette-scores/datasets [EVAL_ID ...]

With no EVAL_ID, builds every ASR eval (about 20 GB of audio downloaded; SPGISpeech
alone is 11.5 GB). Writes ``<out>/<eval_id>/2026.10.1/{test.parquet,metadata.json}``;
serve it with ``PIPETTE_SCORES_DATA_DIR=<out>/..``. A full build must match the
catalog's clip count and ``expected_source_sha256`` where filled; a new
fingerprint is printed for the catalog. ``--limit N`` builds a smoke test under
``2026.10.1-limit<N>`` (never the canonical version), which is never served.

Never commit the output or bake it into an image: SPGISpeech and URGENT24 are
non-commercial and GigaSpeech gated (see docs/dataset-licenses.md). Sources and
pins: pipette_scores/dataset_catalog/asr_datasets.py; each set is read as the
Open ASR Leaderboard loads it (docs/asr-evals.md).
"""

import argparse
import logging
import pathlib
import sys

from pipette_scores.dataset_catalog import _asr_build, asr_datasets
from pipette_scores.types import EvalId


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_ids", nargs="*", help="EvalIds to build (default: all ASR evals)")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="datasets root (…/datasets), outside the repo")
    parser.add_argument("--version", default=None, help=f"default {asr_datasets.DATASET_VERSION}")
    parser.add_argument(
        "--limit", type=int, default=None, help="first N clips only: a smoke test, never served (no count check)"
    )
    args = parser.parse_args()
    # The log goes to stdout too, so each dataset's summary line lands after its own log.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    for noisy in ("httpx", "huggingface_hub", "datasets", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    version = args.version or (
        asr_datasets.DATASET_VERSION if args.limit is None else f"{asr_datasets.DATASET_VERSION}-limit{args.limit}"
    )
    _asr_build.refuse_inside_checkout(args.out)
    ids = [EvalId(e) for e in args.eval_ids] or list(asr_datasets.ASR_DATASETS)
    for eval_id in ids:
        dataset = asr_datasets.ASR_DATASETS[eval_id]
        meta = _asr_build.build(dataset, args.out, version, limit=args.limit)
        audio = meta["audio"]
        if args.limit is not None:
            pin = f"--limit {args.limit} smoke build, never served"
        elif version != asr_datasets.DATASET_VERSION:
            pin = f"version {version}, not checked against the pin"
        elif dataset.expected_source_sha256:
            pin = "matches the pin"
        else:
            pin = "NOT PINNED: add it to EXPECTED_SOURCE_SHA256"
        # One line per dataset, flushed, so it cannot interleave with the logging.
        print(
            f"{eval_id.value}: {meta['num_samples']} clips, source_sha256 {meta['source_sha256']} ({pin}), "
            f"{audio['total_seconds'] / 3600:.2f} h, max {audio['max_seconds']} s, "
            f"{audio['clips_over_30_s']} over 30 s, {audio['audio_bytes'] / 1e6:.0f} MB, "
            f"{audio['formats']} {audio['subtypes']} {audio['sample_rates']} Hz x{audio['channels']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
