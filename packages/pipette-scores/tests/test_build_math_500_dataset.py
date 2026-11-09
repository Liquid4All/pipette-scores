"""scripts/build_math_500_dataset.py: the per-version prompt, repeats and grader."""

import dataclasses
import importlib.util
import json
import pathlib
import sys

import pytest

from pipette_scores.dataset_catalog import load_eval_samples, load_prompt_samples
from pipette_scores.dataset_catalog.math_500 import content_sha256
from pipette_scores.types import EvalId, Math500Grader

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "build_math_500_dataset.py"
_spec = importlib.util.spec_from_file_location("build_math_500_dataset", _SCRIPT)
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)

_DOCS = [
    {"problem": "What is $2+2$?", "solution": "$2+2=\\boxed{4}$.", "answer": "4", "unique_id": "test/a/1.json"},
    {
        "problem": "Simplify $\\frac{2}{4}$.",
        "solution": "It is $\\boxed{\\frac{1}{2}}$.",
        "answer": "\\frac{1}{2}",
        "unique_id": "test/a/2.json",
    },
]


def test_liquid_evals_prompt_is_the_math500_task_prompt():
    # liquid_evals tasks/math500.py prepare_prompt, line 49.
    assert build.VERSIONS["2026.10.1"].prompt("What is $2+2$?") == (
        "What is $2+2$?\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    )


def test_std_prompt_is_unchanged():
    assert build.VERSIONS["2026.06.1"].prompt("What is $2+2$?") == (
        "Solve the following math problem step by step. Put your answer inside \\boxed{}.\n\n"
        "What is $2+2$?\n\n"
        "Remember to put your answer inside \\boxed{}."
    )


def test_versions():
    std, le = build.VERSIONS["2026.06.1"], build.VERSIONS["2026.10.1"]
    assert (std.repeats, std.grader, std.check_gold_extraction) == (5, None, False)
    assert std.content_sha256 == "03d8a41616e2b3f5e0c317e8de72d3a88b930186a0ef7d00686e90051004f588"
    assert (le.repeats, le.grader, le.check_gold_extraction) == (5, Math500Grader.LIQUID_EVALS_MATH500, True)
    assert le.content_sha256 == "5a0da41f1322ab852cfe7ea2e51fca9277f9142cc64a011f07dd597bb6d5eaa2"
    assert "tasks/math500.py" in le.notes and "_std" not in le.notes


def _run(monkeypatch, tmp_path, version, docs, *, pin_hash=True):
    """Run the builder on ``docs`` in place of the HF dataset; return the dataset root."""
    monkeypatch.setattr(build, "load_dataset", lambda *args, **kwargs: docs)
    monkeypatch.setattr(build, "EXPECTED_SAMPLES", len(docs))
    if pin_hash:
        spec = build.VERSIONS[version]
        rows = [build.format_row(doc, spec) for doc in docs]
        digest = content_sha256([(r["prompt"], r["answer"]) for r in rows])
        monkeypatch.setitem(build.VERSIONS, version, dataclasses.replace(spec, content_sha256=digest))
    out = tmp_path / "datasets" / "math_500"
    monkeypatch.setattr(sys, "argv", ["build", "--version", version, "--out-dir", str(out)])
    build.main()
    return tmp_path


@pytest.mark.parametrize(
    "version, repeats, grader",
    [("2026.06.1", 5, Math500Grader.MATH_GENERIC), ("2026.10.1", 5, Math500Grader.LIQUID_EVALS_MATH500)],
)
def test_build_writes_a_dataset_the_catalog_loads(monkeypatch, tmp_path, version, repeats, grader):
    root = _run(monkeypatch, tmp_path, version, _DOCS)
    meta = json.loads((root / "datasets" / "math_500" / version / "metadata.json").read_text())
    assert meta["dataset"] == version
    assert meta["repeats"] == repeats
    assert meta.get("grader") == (None if grader is Math500Grader.MATH_GENERIC else grader.value)

    samples = load_eval_samples(EvalId.MATH_500, version, root_dir=root)
    prompts = load_prompt_samples(EvalId.MATH_500, version, root_dir=root)
    assert len(samples) == len(prompts) == len(_DOCS) * repeats
    # Each problem is served as the attempt ids <id>#0 .. <id>#(repeats-1).
    assert [s.id for s in samples] == [p.id for p in prompts]
    assert [p.id.rsplit("#", 1)[1] for p in prompts[:repeats]] == [str(k) for k in range(repeats)]
    assert {s.grader for s in samples} == {grader}
    assert [s.answer for s in samples[::repeats]] == [d["answer"] for d in _DOCS]
    first = prompts[0].messages
    assert len(first) == 1 and first[0].role == "user"
    assert first[0].content == build.VERSIONS[version].prompt(_DOCS[0]["problem"])


def test_build_rejects_gold_that_liquid_evals_would_extract_differently(monkeypatch, tmp_path):
    docs = [{**_DOCS[0], "answer": "5"}]
    with pytest.raises(SystemExit, match="differs from extract_math_answer"):
        _run(monkeypatch, tmp_path, "2026.10.1", docs)


def test_build_rejects_unexpected_content(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="content hash"):
        _run(monkeypatch, tmp_path, "2026.10.1", _DOCS, pin_hash=False)
