#!/usr/bin/env python
"""One-shot materializer for the MME eval dataset (full set).

Run from the repo root:

    uv run --project packages/pipette-scores \
        python packages/pipette-scores/scripts/build_mme_dataset.py

Downloads VLMEvalKit's pinned ``MME.tsv`` (~48 MB; URL + MD5 from VLMEvalKit
@ 8eca8e5) and writes ``datasets/mme/2026.10.1/`` (test.parquet + metadata.json).
Like the other third-party datasets, the output is gitignored and materialized
at deploy — never committed. MME may not be redistributed in transformed form
(see docs/dataset-licenses.md).

TSV columns: index, category, image_path, question, answer, image. The prompt
is the raw question (ends with "Please answer yes or no."); raw_doc keeps
``image_path`` — the acc_plus pairing key shared by the two yes/no questions
of each image (see scoring/vlm_upstream/PORTING_NOTES.md).
"""

from pipette_scores.dataset_catalog._vlm_build import BenchmarkSpec, run_cli

# --- Upstream source (pinned for reproducibility; VLMEvalKit @ 8eca8e5) ---
URL = "https://opencompass.openxlab.space/utils/VLMEval/MME.tsv"
MD5 = "b36b43c3f09801f5d368627fb92187c3"

# --- Expected output (the build asserts against these) ---
EXPECTED_SAMPLES = 2374
# Fingerprint of the built rows (prompt/target/raw_doc/image digests). A
# mismatch means upstream drift or a formatting change — fail before writing a
# bad blob. (Filled after the first build.)
EXPECTED_CONTENT_SHA256 = "a9cbb72ac15c431eea45521f1fbb06da5d7374c63ac8c92eb517b5460d00613a"

SPEC = BenchmarkSpec(
    eval_id="mme",
    vlmevalkit_name="MME",
    url=URL,
    md5=MD5,
    prompt_style="bare",
    expected_samples=EXPECTED_SAMPLES,
    expected_content_sha256=EXPECTED_CONTENT_SHA256,
    license="none (no declared license)",
    license_notes=(
        "MME has no declared license; the authors' stated terms are "
        "academic-research-only / non-commercial use. You must self-assess "
        "whether your use (especially any commercial use) complies with the "
        "authors' terms before running this benchmark."
    ),
)

if __name__ == "__main__":
    run_cli(SPEC, description=__doc__)
