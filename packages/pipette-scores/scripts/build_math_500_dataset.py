#!/usr/bin/env python
"""One-shot materializer for the MATH-500 eval dataset.

Run from the repo root:

    uv run --project packages/pipette-scores \
        python packages/pipette-scores/scripts/build_math_500_dataset.py

MATH-500 is third-party data, so (like gpqa_diamond) the emitted
`datasets/math_500/<version>/` is gitignored and materialized at deploy from the
pinned upstream revision — not committed.

Reproduces the Artificial Analysis MATH methodology via liquid_evals'
**standardized** track (`tasks/math500_std.py`): the `MathGenericFormatter`
chain-of-thought prompt asking for the final answer in `\\boxed{}`, graded by
`pipette_scores.scoring.math_500.math_generic` (the AA-aligned grader that the
`_std` task pairs with). The gold answer is the upstream `answer` column.

Prompt + grader are taken from the same track on purpose: the plain
`tasks/math500.py` uses a different prompt and a bespoke grader, so mixing the
two would not reproduce the published AA number.
"""

import argparse
import json
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from pipette_scores.dataset_catalog.math_500 import content_sha256

# --- Upstream source (pinned for reproducibility) ---
HF_DATASET = "HuggingFaceH4/MATH-500"
HF_SPLIT = "test"  # MATH-500 ships a single `test` split
# Pin the revision — without it a rebuild silently tracks the dataset's HEAD.
HF_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"  # HuggingFaceH4/MATH-500 @ 2026-06

# --- Expected output (the build asserts against these) ---
EXPECTED_SAMPLES = 500
# Fingerprint of the formatted rows at HF_REVISION. A mismatch means upstream
# drift, a formatter change, or a datasets-lib parsing change — fail before
# writing a bad blob rather than ship one. (Filled after the first build.)
EXPECTED_CONTENT_SHA256 = "03d8a41616e2b3f5e0c317e8de72d3a88b930186a0ef7d00686e90051004f588"


# --- Prompt formatting (AA methodology) ---
# The `MathGenericFormatter` template from liquid_evals' standardized math track
# (`tasks/math500_std.py`): instruction, then the problem, then a trailing
# reminder. Kept byte-identical to liquid-verifiers
# `tasks/backends/math_generic.py` so the prompt matches the score_math_generic
# grader it is paired with.
def format_row(doc: dict) -> dict:
    """Build one {prompt, answer} row from a raw MATH-500 doc (AA methodology)."""
    prompt = (
        "Solve the following math problem step by step. "
        "Put your answer inside \\boxed{}.\n\n"
        f"{doc['problem']}\n\n"
        "Remember to put your answer inside \\boxed{}."
    )
    return {"prompt": prompt, "answer": doc["answer"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="2026.06.1", help="Dataset version dir name")
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("datasets/math_500"),
        help="datasets/math_500 root (version subdir is created under it)",
    )
    args = parser.parse_args()

    dataset = load_dataset(HF_DATASET, split=HF_SPLIT, revision=HF_REVISION)
    rows = [format_row(doc) for doc in dataset]
    if len(rows) != EXPECTED_SAMPLES:
        raise SystemExit(f"expected {EXPECTED_SAMPLES} MATH-500 rows, got {len(rows)} — wrong split/revision?")

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
        "eval_id": "math_500",
        "dataset": args.version,
        "source_dataset": HF_DATASET,
        "num_samples": len(rows),
        "repeats": 5,
        "content_sha256": digest,
        # Informational — actual eval temperature is a client-side policy.
        "generation_params": {"temperature": 0.6},
        "source": {"hf_dataset": HF_DATASET, "split": HF_SPLIT, "hf_revision": HF_REVISION},
        "notes": (
            "MATH-500 (Hendrycks MATH 500-problem subset), Artificial Analysis methodology: "
            "5 repeats per question, pass@1. Prompt + grader are the standardized (_std) track "
            "from liquid_evals tasks/math500_std.py: the MathGenericFormatter chain-of-thought "
            "prompt (instruction / problem / trailing reminder, final answer in \\boxed{}) graded "
            "by pipette_scores.scoring.math_500.math_generic (boxed extraction -> PRM800K grader -> "
            "robust_boxed). Gold answer is the upstream `answer` column. The N logical questions "
            "are served as N*5 #k attempt ids at load time."
        ),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {len(rows)} rows to {out}/test.parquet")


if __name__ == "__main__":
    main()
