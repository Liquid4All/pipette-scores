#!/usr/bin/env python
"""Lay out built ASR datasets for git-LFS: a ``test.parquet`` over the per-file
bound is rewritten as ordered parts ``test-00000-of-0000N.parquet``, … (cut
between row groups, rows in order); one under it is copied as it is.
``metadata.json`` is copied byte for byte: its ``content_sha256`` is a function of
the rows, not of the files, so the parts carry the build's fingerprint, and the
server reads them as one table (_asr_common.parquet_files).

    uv run --project packages/pipette-scores \\
        python packages/pipette-scores/scripts/split_asr_dataset.py \\
        --src /data/pipette-scores/datasets --out <lfs-checkout>/datasets [EVAL_ID ...]

Writes ``<out>/<eval_id>/<version>/`` from ``<src>/<eval_id>/<version>/`` and then
runs the server's integrity checks on the result (verify_asr_datasets.py re-hashes
every clip). The default bound, 1.9 GB, stays under GitHub's smallest git-LFS
per-file limit (2 GB; 5 GB on Enterprise Cloud).
"""

import argparse
import pathlib
import shutil
import sys

from pipette_scores.dataset_catalog import _asr_common, asr_datasets
from pipette_scores.types import EvalId

DEFAULT_MAX_PART_BYTES = 1_900_000_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_ids", nargs="*", help="EvalIds to lay out (default: every ASR eval built under --src)")
    parser.add_argument("--src", type=pathlib.Path, required=True, help="built datasets root (…/datasets)")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="datasets root to write (…/datasets)")
    parser.add_argument("--version", default=asr_datasets.DATASET_VERSION)
    parser.add_argument("--max-part-bytes", type=int, default=DEFAULT_MAX_PART_BYTES)
    args = parser.parse_args()
    ids = [EvalId(e) for e in args.eval_ids] or [
        e for e in asr_datasets.ASR_DATASETS if (args.src / e.value / args.version).is_dir()
    ]
    for eval_id in ids:
        src = args.src / eval_id.value / args.version
        out = args.out / eval_id.value / args.version
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty; remove it first")
        files = _asr_common.split_parts(src / _asr_common.SINGLE_FILE, out, args.max_part_bytes)
        shutil.copyfile(src / "metadata.json", out / "metadata.json")
        rows = _asr_common.verified_rows(out)
        sizes = ", ".join(f"{f.stat().st_size / 1e9:.2f}" for f in files)
        print(f"{eval_id.value}: {len(rows)} clips in {len(files)} file(s) of {sizes} GB, checks pass", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
