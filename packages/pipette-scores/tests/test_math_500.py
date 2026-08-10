import pytest

from pipette_scores.scoring.math_500 import score
from pipette_scores.types import Math500Sample, SampleCompletion


def _sample(id, answer):
    return Math500Sample(id=id, prompt=f"Problem {id}", answer=answer)


@pytest.mark.parametrize(
    "completion, answer, expected",
    [
        ("The answer is \\boxed{42}", "42", True),
        ("\\boxed{7}", "42", False),
        # reasoning is stripped before grading, so the in-think boxed value is ignored
        ("<think>\\boxed{7}</think> So the answer is \\boxed{42}", "42", True),
        ("no boxed answer at all", "42", False),
    ],
)
def test_single_sample(completion, answer, expected):
    scores, ctx = score([SampleCompletion(id="m1", completion=completion)], [_sample("m1", answer)])
    assert scores[0].is_correct is expected
    assert ctx["math_500_pass_at_1"] == (1.0 if expected else 0.0)


def test_pass_at_1_over_repeats():
    # one logical question salted into 4 repeats (#0..#3); 2 of 4 correct -> 0.5
    samples = [_sample(f"m1#{k}", "42") for k in range(4)]
    completions = [
        SampleCompletion(id="m1#0", completion="\\boxed{42}"),
        SampleCompletion(id="m1#1", completion="\\boxed{42}"),
        SampleCompletion(id="m1#2", completion="\\boxed{7}"),
        SampleCompletion(id="m1#3", completion="no answer"),
    ]
    scores, ctx = score(completions, samples)
    assert sum(s.is_correct for s in scores) == 2
    assert ctx["math_500_pass_at_1"] == 0.5
    assert ctx["math_500_pass_at_1_per_sample"] == {"m1": 0.5}


def test_missing_attempt_counts_as_fail():
    # 2 samples but only 1 completion -> denominator is sample count (2)
    samples = [_sample("m1", "1"), _sample("m2", "2")]
    scores, ctx = score([SampleCompletion(id="m1", completion="\\boxed{1}")], samples)
    assert len(scores) == 1
    assert ctx["math_500_pass_at_1"] == 0.5


def test_context_schema_stable_regardless_of_availability():
    empty_ctx = score([], [])[1]
    populated_ctx = score([SampleCompletion(id="m1", completion="\\boxed{1}")], [_sample("m1", "1")])[1]
    assert empty_ctx.keys() == populated_ctx.keys()
    for key, value in empty_ctx.items():
        assert type(value) is type(populated_ctx[key]), key
    assert isinstance(empty_ctx["math_500_pass_at_1"], float)
    assert isinstance(empty_ctx["math_500_pass_at_1_stderr"], float)
    assert isinstance(empty_ctx["math_500_unparsed_rate"], float)
    assert isinstance(empty_ctx["math_500_pass_at_1_per_sample"], dict)
    for method in ["prm800k", "robust_boxed", "no_match", "failed"]:
        assert empty_ctx[f"math_500_method/{method}"] == 0


def test_context_method_diagnostics():
    samples = [_sample("m1", "1"), _sample("m2", "1")]
    completions = [
        SampleCompletion(id="m1", completion="\\boxed{1}"),  # graded by prm800k
        SampleCompletion(id="m2", completion="totally unparseable, no number"),  # no usable answer
    ]
    _, ctx = score(completions, samples)
    # one of two had no extractable answer -> distinguishes parse-miss from wrong answer
    assert ctx["math_500_unparsed_rate"] == pytest.approx(0.5)
    assert ctx["math_500_method/prm800k"] == 1
