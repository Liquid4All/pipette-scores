#!/usr/bin/env python
"""One-shot materializer for the MMStar eval dataset (full set).

Run from the repo root:

    uv run --project packages/pipette-scores \
        python packages/pipette-scores/scripts/build_mmstar_dataset.py

Downloads VLMEvalKit's pinned ``MMStar.tsv`` (~60 MB; URL + MD5 from
VLMEvalKit @ 8eca8e5) and writes ``datasets/mmstar/2026.10.1/`` (test.parquet +
metadata.json). Gitignored, materialized at deploy — never committed. MMStar
may not be redistributed in transformed form (see docs/dataset-licenses.md).

TSV columns: index, question, answer, category, l2_category, bench, image,
A, B, C, D. Prompt = VLMEvalKit MCQ format (no hint column in this TSV).
"""

from pipette_scores.dataset_catalog._vlm_build import BenchmarkSpec, run_cli

# --- Upstream source (pinned for reproducibility; VLMEvalKit @ 8eca8e5) ---
URL = "https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv"
MD5 = "e1ecd2140806c1b1bbf54b43372efb9e"

# --- Expected output (the build asserts against these) ---
EXPECTED_SAMPLES = 1500
# Fingerprint of the built rows. (Filled after the first build.)
EXPECTED_CONTENT_SHA256 = "9851cbed9513819f5aee26bbe2b92ea98b984fef59289ab36b085c13183f49f0"

SPEC = BenchmarkSpec(
    eval_id="mmstar",
    vlmevalkit_name="MMStar",
    url=URL,
    md5=MD5,
    prompt_style="mcq",
    expected_samples=EXPECTED_SAMPLES,
    expected_content_sha256=EXPECTED_CONTENT_SHA256,
    license="none (no declared license)",
    license_notes=(
        "MMStar declares no license of its own and aggregates content from "
        "source benchmarks including SEED-Bench (CC-BY-NC) and ScienceQA "
        "(CC-BY-NC-SA), whose terms it inherits. You must self-assess whether "
        "your use complies with those non-commercial terms."
    ),
)

if __name__ == "__main__":
    run_cli(SPEC, description=__doc__)
