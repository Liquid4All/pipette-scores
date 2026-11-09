"""The vendored liquid_evals MATH-500 grader gives the original's outputs.

The expectations in fixtures/math_500_liquid_evals_parity.json were produced by
running the ORIGINAL liquid_evals/parser.py (the commit is recorded in the
fixture) with scripts/gen_math_500_liquid_evals_parity.py. Regenerate them with
that script after updating the vendored copy.
"""

import json
import pathlib

import pytest

from pipette_scores.scoring.completion_text import remove_thinking_tags
from pipette_scores.scoring.math_500 import liquid_evals_parser as le
from pipette_scores.scoring.math_500 import score
from pipette_scores.types import Math500Grader, Math500Sample, SampleCompletion

_FIXTURE = json.loads((pathlib.Path(__file__).parent / "fixtures" / "math_500_liquid_evals_parity.json").read_text())


def _short(case: dict, key: str) -> str:
    return repr(case[key][:60])


def test_fixture_is_from_the_vendored_commit():
    assert _FIXTURE["source"]["commit"] in le.__doc__
    assert _FIXTURE["source"]["blob"] in le.__doc__


@pytest.mark.parametrize("case", _FIXTURE["extract"], ids=lambda c: _short(c, "text"))
def test_extract_math_answer(case):
    assert le.extract_math_answer(case["text"]) == case["expected"]


@pytest.mark.parametrize("case", _FIXTURE["check"], ids=lambda c: f"{_short(c, 'given')} vs {_short(c, 'truth')}")
def test_check_math_result(case):
    given, truth = case["given"], case["truth"]
    assert le.mathd_normalize_answer(given) == case["mathd_normalize_given"]
    assert le._normalize(given) == case["normalize_given"]
    assert le.grade_answer_mathd(given, truth) is case["mathd"]
    assert le.grade_answer_sympy(given, truth) is case["sympy"]
    assert le.check_math_result(given, truth) is case["check"]


@pytest.mark.parametrize("case", _FIXTURE["verify"], ids=lambda c: f"{_short(c, 'raw')} vs {_short(c, 'gold')}")
def test_math500_task_pipeline(case):
    # tasks/math500.py: remove_thinking_tags, extract_math_answer, verify_math_answer.
    response = remove_thinking_tags(case["raw"])
    predicted = le.extract_math_answer(response)
    assert predicted == case["predicted"]
    assert le.verify_math_answer(case["gold"], predicted, response) is case["correct"]


def test_scorer_matches_the_task_verdicts():
    # The verify cases carry no markup that normalize_reasoning_markup rewrites,
    # so the scorer must give the task's verdict on every one.
    cases = _FIXTURE["verify"]
    samples = [
        Math500Sample(id=f"m{i}", prompt=f"P{i}", answer=c["gold"], grader=Math500Grader.LIQUID_EVALS_MATH500)
        for i, c in enumerate(cases)
    ]
    completions = [SampleCompletion(id=f"m{i}", completion=c["raw"]) for i, c in enumerate(cases)]
    scores, ctx = score(completions, samples)
    assert [s.is_correct for s in scores] == [c["correct"] for c in cases]
    assert ctx["math_500_pass_at_1"] == pytest.approx(sum(c["correct"] for c in cases) / len(cases))
