#!/usr/bin/env python
"""One-shot materializer for the POPE eval dataset (full set).

Run from the repo root:

    uv run --project packages/pipette-scores \
        python packages/pipette-scores/scripts/build_pope_dataset.py

Downloads VLMEvalKit's pinned ``POPE.tsv`` (~1.1 GB — the largest of the 7;
URL + MD5 from VLMEvalKit @ 8eca8e5) and writes ``datasets/pope/2026.10.1/``
(test.parquet + metadata.json). Gitignored, materialized at deploy — never
committed. POPE's COCO/Flickr images may not be redistributed (see
docs/dataset-licenses.md).

TSV columns: image, question, answer, category, index. Prompt = the raw
question ("Is there a ... in the image?"). ``category`` is the subset name
(adversarial / popular / random; may be comma-joined — the scorer explodes
it) and is carried verbatim in raw_doc.
"""

from pipette_scores.dataset_catalog._vlm_build import BenchmarkSpec, run_cli

# --- Upstream source (pinned for reproducibility; VLMEvalKit @ 8eca8e5) ---
URL = "https://opencompass.openxlab.space/utils/VLMEval/POPE.tsv"
MD5 = "c12f5acb142f2ef1f85a26ba2fbe41d5"

# --- Expected output (the build asserts against these) ---
# Row count after VLMEvalKit's merge of overlapping subset questions
# (comma-joined categories). (Filled after the first build.)
EXPECTED_SAMPLES = 5127  # POPE.tsv rows (overlapping subset rows merged with comma-joined categories)
# Fingerprint of the built rows. (Filled after the first build.)
EXPECTED_CONTENT_SHA256 = "b47995b5ea4310771c31be26d9487fe8d3d9f75ae53f09cce8806c9147c680f0"

SPEC = BenchmarkSpec(
    eval_id="pope",
    vlmevalkit_name="POPE",
    url=URL,
    md5=MD5,
    prompt_style="bare",
    expected_samples=EXPECTED_SAMPLES,
    expected_content_sha256=EXPECTED_CONTENT_SHA256,
    license="MIT (annotations); images: COCO val2014 / Flickr per-photo terms",
    license_notes=(
        "POPE annotations are MIT-licensed, but the images are COCO val2014 "
        "photos governed by Flickr per-photo terms that do not permit "
        "redistribution. Do not redistribute the built dataset."
    ),
)

if __name__ == "__main__":
    run_cli(SPEC, description=__doc__)
