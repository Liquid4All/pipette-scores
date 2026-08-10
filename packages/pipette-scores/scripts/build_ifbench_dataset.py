#!/usr/bin/env python
"""One-shot materializer for the IFBench eval dataset.

Run from the repo root:

    uv run --project packages/pipette-scores \
        python packages/pipette-scores/scripts/build_ifbench_dataset.py

Reproduces the dataset from the vendored upstream test set
(`vendor/ifbench/data/IFBench_test.jsonl`, pinned by the submodule commit), so
the dataset itself is gitignored rather than committed. The rows are projected
to the four columns the loader uses and written via pyarrow, whose struct-schema
unification produces the fully-expanded `kwargs`. The build self-asserts a
content hash so a drifted upstream fails loud rather than shipping a bad blob.
"""

import argparse
import json
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq

from pipette_scores.dataset_catalog.ifbench import content_sha256
from pipette_scores.parquet_io import read_parquet

# --- Upstream source (vendored submodule, pinned by its commit) ---
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SOURCE_JSONL = _REPO_ROOT / "vendor" / "ifbench" / "data" / "IFBench_test.jsonl"
SOURCE_COMMIT = "cb932e352a505306ad0115272211df14bb8f628f"  # allenai/IFBench @ 2026-04-11

# --- Expected output (the build asserts against these) ---
EXPECTED_SAMPLES = 300
EXPECTED_CONTENT_SHA256 = "02e1c189ca2529603f6d9a64c2edd0d5a0c1f4fefe0ba9a2a04d4ecb9c9f2128"

# The columns the loader consumes; n_constraints is derived at load, not stored.
_COLUMNS = ("key", "instruction_id_list", "prompt", "kwargs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="2026.06.1", help="Dataset version dir name")
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("datasets/ifbench"),
        help="datasets/ifbench root (version subdir is created under it)",
    )
    args = parser.parse_args()

    with SOURCE_JSONL.open() as f:
        raw = [json.loads(line) for line in f if line.strip()]
    rows = [{col: r[col] for col in _COLUMNS} for r in raw]
    if len(rows) != EXPECTED_SAMPLES:
        raise SystemExit(f"expected {EXPECTED_SAMPLES} IFBench rows, got {len(rows)} — wrong source?")

    out = args.out_dir / args.version
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out / "train.parquet")

    # Hash the parquet round-trip (the kwargs struct schema is unified on write),
    # matching what the loader verifies.
    digest = content_sha256(read_parquet(out / "train.parquet"))
    if digest != EXPECTED_CONTENT_SHA256:
        raise SystemExit(
            f"content hash {digest} != expected {EXPECTED_CONTENT_SHA256} — the vendored "
            "IFBench test set or the parquet schema changed; review before updating "
            "EXPECTED_CONTENT_SHA256."
        )

    metadata = {
        "eval_id": "ifbench",
        "dataset": args.version,
        "source_dataset": "original",
        "num_samples": len(rows),
        "repeats": 5,
        "content_sha256": digest,
        "generation_params": {"temperature": 0.6},
        "source": {
            "repo": "https://github.com/allenai/IFBench",
            "commit": SOURCE_COMMIT,
            "commit_date": "2026-04-11T18:35:39-04:00",
            "file": "data/IFBench_test.jsonl",
            "hf_dataset": "allenai/IFBench_test",
            "data_license": "ODC-By-1.0",
            "data_license_url": "https://opendatacommons.org/licenses/by/1-0/",
            "responsible_use_guidelines": "https://allenai.org/responsible-use",
        },
        "notes": (
            "Copy of `original` (verbatim upstream IFBench test set) versioned to carry the "
            "evaluation methodology: 5 repeats per logical sample (pass@1 over 5, loose), generated "
            "at temperature 0.6. The 300 logical samples are served as 1500 #k attempt ids at load time."
        ),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {len(rows)} rows to {out}/train.parquet")


if __name__ == "__main__":
    main()
