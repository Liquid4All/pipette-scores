#!/usr/bin/env python
"""One-shot materializer for the MATH-500 eval dataset.

Run from the repo root, once per dataset version:

    uv run --project packages/pipette-scores \
        python packages/pipette-scores/scripts/build_math_500_dataset.py --version 2026.10.1

MATH-500 is third-party data, so (like gpqa_diamond) the emitted
`datasets/math_500/<version>/` is gitignored and materialized at deploy from the
pinned upstream revision — not committed.

Both versions hold the same 500 problems with the upstream `answer` column as
gold. They differ in the liquid_evals track they reproduce:

- `2026.06.1` — the Artificial Analysis methodology via liquid_evals'
  **standardized** track (`tasks/math500_std.py`): the `MathGenericFormatter`
  chain-of-thought prompt asking for the final answer in `\\boxed{}`, 5 repeats,
  graded by `pipette_scores.scoring.math_500.math_generic` (the AA-aligned
  grader that the `_std` task pairs with).
- `2026.10.1` — liquid_evals' **plain** `tasks/math500.py`, the MATH-500 on
  Liquid's model cards: the problem, then "Please reason step by step, and put
  your final answer within \\boxed{}.", 5 repeats, graded by liquid_evals'
  `extract_math_answer` + `verify_math_answer`
  (`pipette_scores.scoring.math_500.liquid_evals_parser`). One task run is one
  pass over the 500 problems; the model-card number is the mean of 5 such
  runs, and pass@1 over the 5 repeats is that mean. The task takes its gold
  from `extract_math_answer(solution)`; the build checks that this equals
  `answer` on every row.

Each version's prompt and grader come from the same track on purpose: mixing
them would reproduce neither number. The grader is recorded as
`metadata.grader` (absent in 2026.06.1, which predates the key and so keeps
math_generic).
"""

import argparse
import json
import pathlib
from collections.abc import Callable
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from pipette_scores.dataset_catalog.math_500 import content_sha256
from pipette_scores.scoring.math_500.liquid_evals_parser import extract_math_answer
from pipette_scores.types import Math500Grader

# --- Upstream source (pinned for reproducibility) ---
HF_DATASET = "HuggingFaceH4/MATH-500"
HF_SPLIT = "test"  # MATH-500 ships a single `test` split
# Pin the revision — without it a rebuild silently tracks the dataset's HEAD.
HF_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"  # HuggingFaceH4/MATH-500 @ 2026-06

EXPECTED_SAMPLES = 500


# --- Prompt formatting ---
# The `MathGenericFormatter` template from liquid_evals' standardized math track
# (`tasks/math500_std.py`): instruction, then the problem, then a trailing
# reminder. Kept byte-identical to liquid-verifiers
# `tasks/backends/math_generic.py` so the prompt matches the score_math_generic
# grader it is paired with.
def std_prompt(problem: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "Put your answer inside \\boxed{}.\n\n"
        f"{problem}\n\n"
        "Remember to put your answer inside \\boxed{}."
    )


# liquid_evals `tasks/math500.py` prepare_prompt (d5534cf, line 49): the single
# user message; the task sends no system prompt.
def liquid_evals_prompt(problem: str) -> str:
    return f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


@dataclass(frozen=True)
class DatasetVersion:
    prompt: Callable[[str], str]
    repeats: int
    # Fingerprint of the formatted rows at HF_REVISION. A mismatch means
    # upstream drift, a formatter change, or a datasets-lib parsing change —
    # fail before writing a bad blob rather than ship one.
    content_sha256: str
    # metadata.grader; None leaves the key out.
    grader: Math500Grader | None
    # The gold must equal liquid_evals' extract_math_answer(solution).
    check_gold_extraction: bool
    notes: str


VERSIONS = {
    "2026.06.1": DatasetVersion(
        prompt=std_prompt,
        repeats=5,
        content_sha256="03d8a41616e2b3f5e0c317e8de72d3a88b930186a0ef7d00686e90051004f588",
        grader=None,
        check_gold_extraction=False,
        notes=(
            "MATH-500 (Hendrycks MATH 500-problem subset), Artificial Analysis methodology: "
            "5 repeats per question, pass@1. Prompt + grader are the standardized (_std) track "
            "from liquid_evals tasks/math500_std.py: the MathGenericFormatter chain-of-thought "
            "prompt (instruction / problem / trailing reminder, final answer in \\boxed{}) graded "
            "by pipette_scores.scoring.math_500.math_generic (boxed extraction -> PRM800K grader -> "
            "robust_boxed). Gold answer is the upstream `answer` column. The N logical questions "
            "are served as N*5 #k attempt ids at load time."
        ),
    ),
    "2026.10.1": DatasetVersion(
        prompt=liquid_evals_prompt,
        repeats=5,
        content_sha256="5a0da41f1322ab852cfe7ea2e51fca9277f9142cc64a011f07dd597bb6d5eaa2",
        grader=Math500Grader.LIQUID_EVALS_MATH500,
        check_gold_extraction=True,
        notes=(
            "MATH-500 (Hendrycks MATH 500-problem subset), liquid_evals' plain math500 track "
            "(tasks/math500.py @ d5534cf), the MATH-500 Liquid reports on its model cards: 5 repeats "
            "per question, pass@1. One math500 task run is one pass over the 500 problems, and the "
            "model-card number is the mean of 5 runs; pass@1 over the 5 repeats is that mean. The N "
            "logical questions are served as N*5 #k attempt ids at load time. Prompt: one user "
            "message, no system prompt -- the problem, then 'Please reason step by step, and put "
            "your final answer within \\boxed{}.' "
            "Graded by liquid_evals' extract_math_answer + verify_math_answer (mathd/sympy "
            "equivalence on the extracted answer, then on each of the response's last 150 lines), "
            "vendored verbatim as pipette_scores.scoring.math_500.liquid_evals_parser and selected "
            "by `grader`. Gold answer is the upstream `answer` column, identical on all 500 rows to "
            "the task's extract_math_answer(solution). liquid_evals shuffles the problems "
            "(random_state=777); order does not affect the score."
        ),
    ),
}


def format_row(doc: dict, version: DatasetVersion) -> dict:
    """Build one {prompt, answer} row from a raw MATH-500 doc."""
    return {"prompt": version.prompt(doc["problem"]), "answer": doc["answer"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="2026.06.1", choices=sorted(VERSIONS), help="Dataset version")
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("datasets/math_500"),
        help="datasets/math_500 root (version subdir is created under it)",
    )
    args = parser.parse_args()
    version = VERSIONS[args.version]

    dataset = load_dataset(HF_DATASET, split=HF_SPLIT, revision=HF_REVISION)
    if version.check_gold_extraction:
        differ = [doc["unique_id"] for doc in dataset if extract_math_answer(doc["solution"]) != doc["answer"]]
        if differ:
            raise SystemExit(
                f"`answer` differs from extract_math_answer(solution) on {len(differ)} rows "
                f"(e.g. {differ[:3]}); liquid_evals grades against the latter."
            )
    rows = [format_row(doc, version) for doc in dataset]
    if len(rows) != EXPECTED_SAMPLES:
        raise SystemExit(f"expected {EXPECTED_SAMPLES} MATH-500 rows, got {len(rows)} — wrong split/revision?")

    digest = content_sha256([(r["prompt"], r["answer"]) for r in rows])
    if digest != version.content_sha256:
        raise SystemExit(
            f"content hash {digest} != expected {version.content_sha256} — upstream "
            f"data at {HF_REVISION} or the formatting changed; review before updating "
            f"the {args.version} content_sha256."
        )

    out = args.out_dir / args.version
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out / "test.parquet")

    metadata = {
        "eval_id": "math_500",
        "dataset": args.version,
        "source_dataset": HF_DATASET,
        "num_samples": len(rows),
        "repeats": version.repeats,
        "content_sha256": digest,
        **({"grader": version.grader.value} if version.grader else {}),
        # Informational — actual eval temperature is a client-side policy.
        "generation_params": {"temperature": 0.6},
        "source": {"hf_dataset": HF_DATASET, "split": HF_SPLIT, "hf_revision": HF_REVISION},
        "notes": version.notes,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {len(rows)} rows to {out}/test.parquet")


if __name__ == "__main__":
    main()
