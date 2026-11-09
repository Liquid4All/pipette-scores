#!/usr/bin/env python
"""Offline integrity check of materialized ASR datasets: every clip's audio bytes
against its recorded sha256, then the checks the server makes before it scores or
serves a dataset (content_sha256 against metadata.json and, for the canonical
version, source_sha256 against the catalog's expected_source_sha256; each sample id its position; the
scoring recipe, language and clip count the catalog's; no smoke build). A set
split into test-<i>-of-<n>.parquet parts is checked as the one table they hold.

    uv run --project packages/pipette-scores \\
        python packages/pipette-scores/scripts/verify_asr_datasets.py --root /data/pipette-scores/datasets
"""

import argparse
import hashlib
import pathlib
import sys

import pyarrow.parquet as pq

from pipette_scores.dataset_catalog import _asr_common, asr_datasets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=pathlib.Path, required=True, help="datasets root (…/datasets)")
    parser.add_argument("--version", default=asr_datasets.DATASET_VERSION)
    args = parser.parse_args()
    failed = 0
    for eval_id, dataset in asr_datasets.ASR_DATASETS.items():
        d = args.root / eval_id.value / args.version
        if not d.is_dir():
            continue
        bad = clips = 0
        files: list[pathlib.Path] = []
        digest = None
        try:
            files = _asr_common.parquet_files(d)
            for path in files:
                file = pq.ParquetFile(path)
                clips += file.metadata.num_rows
                for batch in file.iter_batches(columns=["audio", "audio_sha256"], batch_size=256):
                    pairs = zip(batch.column("audio").to_pylist(), batch.column("audio_sha256").to_pylist())
                    bad += sum(hashlib.sha256(audio).hexdigest() != digest for audio, digest in pairs)
            digest = _asr_common.source_sha256(_asr_common.verified_rows(d))
            problem = ""
        # OSError and ValueError cover a missing file, bad JSON and a corrupt parquet.
        except (OSError, ValueError) as e:
            problem = str(e)
        pinned = "pinned" if dataset.expected_source_sha256 else "NOT PINNED"
        status = "ok" if not bad and not problem else "FAILED"
        failed += status != "ok"
        layout = f"{len(files)} parts" if len(files) > 1 else (files[0].name if files else "no parquet")
        print(f"{eval_id.value}: {status} ({clips} clips in {layout}, {bad} audio mismatches, {pinned} {digest})")
        if problem:
            print(f"  {problem}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
