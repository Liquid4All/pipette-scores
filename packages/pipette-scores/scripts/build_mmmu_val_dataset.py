#!/usr/bin/env python
"""One-shot materializer for the MMMU validation-split eval dataset.

Run from the repo root:

    uv run --project packages/pipette-scores \
        python packages/pipette-scores/scripts/build_mmmu_val_dataset.py

Downloads VLMEvalKit's pinned ``MMMU_DEV_VAL.tsv`` (~81 MB; URL + MD5 from
VLMEvalKit @ 8eca8e5) and writes ``datasets/mmmu_val/2026.10.1/`` (test.parquet +
metadata.json). Gitignored, materialized at deploy — never committed. MMMU
images may not be redistributed in transformed form (see
docs/dataset-licenses.md).

The TSV holds 1050 rows (150 dev + 900 validation); only the 900 VAL rows are
kept. Multi-image rows store a stringified list in the ``image`` cell; the
question text keeps its ``<image N>`` placeholders unchanged — the prompt
loader (dataset_catalog/_vlm_common.py) interleaves images at those positions,
matching VLMEvalKit ``MMMUDataset.split_MMMU``. Option columns run A-I and are
absent (NaN) for open questions, which the scorer rewrites via MMMU_preproc.
"""

import pandas as pd

from pipette_scores.dataset_catalog._vlm_build import BenchmarkSpec, run_cli

# --- Upstream source (pinned for reproducibility; VLMEvalKit @ 8eca8e5) ---
URL = "https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv"
MD5 = "585e8ad75e73f75dcad265dfd0417d64"

# --- Expected output (the build asserts against these) ---
EXPECTED_SAMPLES = 900  # VAL rows only; the 150 DEV rows are excluded
# Fingerprint of the built rows. (Filled after the first build.)
EXPECTED_CONTENT_SHA256 = "3a3e615964a3ab63988cf441a153e38250b91cdcadbcd615d315b88ae3202ee4"


def keep_validation(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["split"] == "validation"].reset_index(drop=True)
    if len(out) != EXPECTED_SAMPLES:
        raise SystemExit(f"expected {EXPECTED_SAMPLES} MMMU validation rows, got {len(out)} — TSV layout changed?")
    return out


SPEC = BenchmarkSpec(
    eval_id="mmmu_val",
    vlmevalkit_name="MMMU_DEV_VAL",
    url=URL,
    md5=MD5,
    prompt_style="mcq",
    expected_samples=EXPECTED_SAMPLES,
    expected_content_sha256=EXPECTED_CONTENT_SHA256,
    filter_fn=keep_validation,
    license="Apache-2.0 (dataset card)",
    license_notes=(
        "The MMMU dataset card is Apache-2.0, but the images come from "
        "textbooks and online sources with uncleared rights, and the card "
        "carries a takedown clause. You must self-assess redistribution and "
        "commercial-use risk."
    ),
)

if __name__ == "__main__":
    run_cli(SPEC, description=__doc__)
