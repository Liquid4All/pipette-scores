#!/usr/bin/env python
"""One-shot materializer for the RealWorldQA eval dataset (full set).

Run from the repo root:

    uv run --project packages/pipette-scores \
        python packages/pipette-scores/scripts/build_realworldqa_dataset.py

Downloads VLMEvalKit's pinned ``RealWorldQA.tsv`` (~184 MB; URL + MD5 from
VLMEvalKit @ 8eca8e5) and writes ``datasets/realworldqa/2026.10.1/`` (test.parquet
+ metadata.json). Gitignored, materialized at deploy — never committed.

LICENSING: RealWorldQA is CC-BY-ND-4.0. The parquet this script produces is
Adapted Material under that license and MUST NOT be redistributed — build it
locally, use it locally (see docs/dataset-licenses.md).

TSV columns: image, answer, index, A, B, C, D, question. Prompt = VLMEvalKit
MCQ format.
"""

from pipette_scores.dataset_catalog._vlm_build import BenchmarkSpec, run_cli

# --- Upstream source (pinned for reproducibility; VLMEvalKit @ 8eca8e5) ---
URL = "https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv"
MD5 = "4de008f55dc4fd008ca9e15321dc44b7"

# --- Expected output (the build asserts against these) ---
EXPECTED_SAMPLES = 765
# Fingerprint of the built rows. (Filled after the first build.)
EXPECTED_CONTENT_SHA256 = "b05826e7058eb131c6ed787c5271e49454e911a01645f30feb71c7681bb1bb38"

_ND_WARNING = (
    "*** RealWorldQA is CC-BY-ND-4.0 (NoDerivatives). The parquet file just\n"
    "*** built is Adapted Material and MUST NOT be redistributed, shared,\n"
    "*** uploaded, or served outside your organization in ANY form. Keep it\n"
    "*** on the machine that runs the eval."
)

SPEC = BenchmarkSpec(
    eval_id="realworldqa",
    vlmevalkit_name="RealWorldQA",
    url=URL,
    md5=MD5,
    prompt_style="mcq",
    expected_samples=EXPECTED_SAMPLES,
    expected_content_sha256=EXPECTED_CONTENT_SHA256,
    license="CC-BY-ND-4.0",
    license_notes=(
        "CC-BY-ND-4.0 prohibits sharing adaptations. The built parquet "
        "(repacked images + reformatted prompts) is Adapted Material and must "
        "not be redistributed in any form."
    ),
    extra_notice=_ND_WARNING,
)

if __name__ == "__main__":
    run_cli(SPEC, description=__doc__)
