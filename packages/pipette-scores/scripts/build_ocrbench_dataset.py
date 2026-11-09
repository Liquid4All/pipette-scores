#!/usr/bin/env python
"""One-shot materializer for the OCRBench eval dataset (full set).

Run from the repo root:

    uv run --project packages/pipette-scores \
        python packages/pipette-scores/scripts/build_ocrbench_dataset.py

Downloads VLMEvalKit's pinned ``OCRBench.tsv`` (~114 MB; URL + MD5 from
VLMEvalKit @ 8eca8e5) and writes ``datasets/ocrbench/2026.10.1/`` (test.parquet +
metadata.json). Gitignored, materialized at deploy — never committed.
OCRBench's image sources may not be redistributed in transformed form (see
docs/dataset-licenses.md).

TSV columns: index, image, question, answer, category. Prompt = the raw
question. ``answer`` is a stringified list of acceptable strings; ``category``
is one of the 10 OCRBench categories, exact case (incl. the case-sensitive
``Handwritten Mathematical Expression Recognition``) — both carried verbatim
in raw_doc for the scorer.
"""

from pipette_scores.dataset_catalog._vlm_build import BenchmarkSpec, run_cli

# --- Upstream source (pinned for reproducibility; VLMEvalKit @ 8eca8e5) ---
URL = "https://opencompass.openxlab.space/utils/VLMEval/OCRBench.tsv"
MD5 = "e953d98a987cc6e26ef717b61260b778"

# --- Expected output (the build asserts against these) ---
EXPECTED_SAMPLES = 1000  # OCRBench has exactly 1000 questions (Final Score denominator)
# Fingerprint of the built rows. (Filled after the first build.)
EXPECTED_CONTENT_SHA256 = "b46dd26966effff2319ab30520d479063cce5c217617cb047a7f2515ab4010fe"

SPEC = BenchmarkSpec(
    eval_id="ocrbench",
    vlmevalkit_name="OCRBench",
    url=URL,
    md5=MD5,
    prompt_style="bare",
    expected_samples=EXPECTED_SAMPLES,
    expected_content_sha256=EXPECTED_CONTENT_SHA256,
    license="MIT (benchmark code); images: mixed, ~29 source OCR datasets",
    license_notes=(
        "The OCRBench repository code is MIT-licensed, but the images aggregate "
        "roughly 29 OCR datasets with mixed and partly research-only terms. You "
        "must self-assess whether your use complies with each source dataset's "
        "terms."
    ),
)

if __name__ == "__main__":
    run_cli(SPEC, description=__doc__)
