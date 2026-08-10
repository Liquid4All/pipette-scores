#!/usr/bin/env python
"""One-shot materializer for the GPQA Diamond eval dataset.

Run once with HF access to materialize the dataset blob. It is gitignored — not
committed; supply it out of band (a local run here, or the CDK uploads it at
deploy). Run from the repo root:

    HF_TOKEN=... uv run --project packages/pipette-scores \
        python packages/pipette-scores/scripts/build_gpqa_diamond_dataset.py

Downloads the gated `Idavidrein/gpqa` (config `gpqa_diamond`, 198 questions)
and writes the dataset the scorer expects:

    datasets/gpqa_diamond/<version>/test.parquet   columns: prompt, answer
    datasets/gpqa_diamond/<version>/metadata.json  repeats: 5 (AA pass@1)

Prompt + option-shuffle reproduce liquid_evals' `GPQAFormatter` exactly (AA
methodology): options are correct + 3 incorrect, shuffled with a per-question
`md5(Question)` seed so the prompt is stable and the recorded `answer` letter
tracks the shuffled position. The scorer (`pipette_scores.scoring.gpqa_diamond`)
runs each completion through the same AA MCQ extractor against this letter.
"""

import argparse
import hashlib
import json
import pathlib
import random

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from pipette_scores.dataset_catalog.gpqa_diamond import content_sha256

# --- Upstream source (pinned for reproducibility) ---
HF_DATASET = "Idavidrein/gpqa"
HF_CONFIG = "gpqa_diamond"
HF_SPLIT = "train"  # GPQA ships a single `train` split
# Pin the revision — without it a rebuild silently tracks the dataset's HEAD.
HF_REVISION = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"  # Idavidrein/gpqa @ 2026-03-05

# --- Expected output (the build asserts against these) ---
EXPECTED_SAMPLES = 198  # GPQA Diamond subset; the full set is 448
# Fingerprint of the formatted rows at HF_REVISION. A mismatch means upstream
# drift, a formatter change, or a datasets-lib parsing change — fail before
# writing a bad blob rather than ship one.
EXPECTED_CONTENT_SHA256 = "60d4d93801f979b3dc52847f1eadb9d14a495a427fbb4934858f18530fab9281"

# --- Prompt formatting (AA methodology; matches liquid_evals GPQAFormatter) ---
OPTION_LETTERS = "ABCD"
PROMPT_TEMPLATE = """Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A').

{question}

{options}"""


def format_row(doc: dict) -> dict:
    """Build one {prompt, answer} row from a raw GPQA doc (AA methodology)."""
    options = [(doc["Correct Answer"], True)] + [(doc[f"Incorrect Answer {i}"], False) for i in (1, 2, 3)]
    # Deterministic per-question shuffle: same question -> same option order.
    seed = int(hashlib.md5(doc["Question"].encode()).hexdigest(), 16) % (2**32)
    random.Random(seed).shuffle(options)

    options_str = "\n".join(f"{letter}) {opt}" for letter, (opt, _) in zip(OPTION_LETTERS, options))
    correct_idx = next(i for i, (_, is_correct) in enumerate(options) if is_correct)
    return {
        "prompt": PROMPT_TEMPLATE.format(question=doc["Question"], options=options_str),
        "answer": OPTION_LETTERS[correct_idx],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="2026.06.1", help="Dataset version dir name")
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("datasets/gpqa_diamond"),
        help="datasets/gpqa_diamond root (version subdir is created under it)",
    )
    args = parser.parse_args()

    dataset = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT, revision=HF_REVISION)
    rows = [format_row(doc) for doc in dataset]
    if len(rows) != EXPECTED_SAMPLES:
        raise SystemExit(
            f"expected {EXPECTED_SAMPLES} GPQA Diamond questions, got {len(rows)} — wrong config/split/revision?"
        )

    digest = content_sha256([(r["prompt"], r["answer"]) for r in rows])
    if digest != EXPECTED_CONTENT_SHA256:
        raise SystemExit(
            f"content hash {digest} != expected {EXPECTED_CONTENT_SHA256} — upstream "
            f"data at {HF_REVISION} or the formatting changed; review before updating "
            "EXPECTED_CONTENT_SHA256."
        )

    out = args.out_dir / args.version
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out / "test.parquet")

    metadata = {
        "eval_id": "gpqa_diamond",
        "dataset": args.version,
        "source_dataset": HF_DATASET,
        "num_samples": len(rows),
        "repeats": 5,
        # Fingerprint of the logical (prompt, answer) rows; the loader re-verifies
        # it so a materialized blob fetched from outside git is provably canonical.
        "content_sha256": digest,
        # Informational — actual eval temperature is a client-side policy.
        "generation_params": {"temperature": 0.6},
        "source": {
            "hf_dataset": HF_DATASET,
            "hf_config": HF_CONFIG,
            "split": HF_SPLIT,
            "hf_revision": HF_REVISION,
        },
        "notes": (
            "GPQA Diamond (graduate-level science MCQ), Artificial Analysis methodology: "
            "5 repeats per question, pass@1. Options are the correct answer + 3 distractors, "
            "deterministically shuffled per question with an md5(Question) seed (matches "
            "liquid_evals GPQAFormatter). The N logical questions are served as N*5 #k attempt "
            "ids at load time."
        ),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    distribution = {letter: sum(r["answer"] == letter for r in rows) for letter in OPTION_LETTERS}
    print(f"Wrote {len(rows)} rows to {out}/test.parquet")
    print(f"Answer-letter distribution: {distribution}")


if __name__ == "__main__":
    main()
