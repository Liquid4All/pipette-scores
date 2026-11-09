#!/usr/bin/env python
"""One-shot materializer for the MMBench EN dev (v1.1) eval dataset.

Run from the repo root:

    uv run --project packages/pipette-scores \
        python packages/pipette-scores/scripts/build_mmbench_en_dev_dataset.py

Downloads VLMEvalKit's pinned ``MMBench_DEV_EN_V11.tsv`` (~52 MB; URL + MD5 from
VLMEvalKit @ 8eca8e5) and writes ``datasets/mmbench_en_dev/2026.10.1/``
(test.parquet + metadata.json). Gitignored, materialized at deploy — never
committed. MMBench images may not be redistributed in transformed form (see
docs/dataset-licenses.md).

TSV columns: index, question, hint, A, B, C, D, answer, category, image,
l2-category, split. Prompt = VLMEvalKit MCQ format
(``Hint:`` line included when the row has one).

The TSV contains ALL circular-rotation rows (rotations of one base question
share ``index % 1e6``); every row is kept so circular scoring works. Rotation
rows store an index pointer in the ``image`` cell instead of base64 — the
builder resolves those to the actual bytes (VLMEvalKit image-dedup scheme).
"""

from pipette_scores.dataset_catalog._vlm_build import BenchmarkSpec, run_cli

# --- Upstream source (pinned for reproducibility; VLMEvalKit @ 8eca8e5) ---
URL = "https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN_V11.tsv"
MD5 = "30c05be8f2f347a50be25aa067248184"

# --- Expected output (the build asserts against these) ---
# Total row count incl. circular rotations. (Filled after the first build.)
EXPECTED_SAMPLES = 4876  # MMBench_DEV_EN_V11.tsv: 1292 base questions incl. circular rotations
# Fingerprint of the built rows. (Filled after the first build.)
EXPECTED_CONTENT_SHA256 = "06d4cc91d3fa972f706731623e3cdf46c2cf10dfaa6f9366e81458b7358b2b3a"

SPEC = BenchmarkSpec(
    eval_id="mmbench_en_dev",
    vlmevalkit_name="MMBench_DEV_EN_V11",
    url=URL,
    md5=MD5,
    prompt_style="mcq",
    expected_samples=EXPECTED_SAMPLES,
    expected_content_sha256=EXPECTED_CONTENT_SHA256,
    license="Apache-2.0 (repository)",
    license_notes=(
        "The MMBench repository is Apache-2.0, but the images come from public "
        "datasets and the internet with uncleared rights. You must self-assess "
        "redistribution and commercial-use risk."
    ),
)

if __name__ == "__main__":
    run_cli(SPEC, description=__doc__)
