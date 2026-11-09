import pytest

from pipette_scores.scoring.math_500 import score
from pipette_scores.types import Math500Grader, Math500Sample, SampleCompletion


def _sample(id, answer):
    return Math500Sample(id=id, prompt=f"Problem {id}", answer=answer)


@pytest.mark.parametrize(
    "completion, answer, expected",
    [
        ("The answer is \\boxed{42}", "42", True),
        ("\\boxed{7}", "42", False),
        # reasoning is stripped before grading, so the in-think boxed value is ignored
        ("<think>\\boxed{7}</think> So the answer is \\boxed{42}", "42", True),
        # same for Cohere North's markers; a Gemma 4 thought cut off is all reasoning
        ("\\boxed{7}?<|END_THINKING|><|START_TEXT|>\\boxed{42}<|END_TEXT|>", "42", True),
        ("<|channel>thought\nSo \\boxed{42}, but check the sign", "42", False),
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
    assert empty_ctx["math_500_blanked_truncated"] == 0
    assert isinstance(empty_ctx["math_500_pass_at_1_blanked"], float)
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


@pytest.mark.parametrize("stop_reason", ["truncated", "doom_loop"])
@pytest.mark.parametrize("enable_thinking, blanked", [(True, 1), (False, 0), (None, 0)])
def test_cut_reasoning_is_graded_raw_and_counted(stop_reason, enable_thinking, blanked):
    samples = [_sample("m1", "42"), _sample("m2", "42")]
    completions = [
        # The prompt opened the reasoning and generation was cut: graded whole,
        # as liquid_evals grades it.
        SampleCompletion(id="m1", completion="So \\boxed{42}? Check the sign of", stop_reason=stop_reason),
        # The reasoning closed before the cut.
        SampleCompletion(id="m2", completion="So 42.</think>The answer is \\boxed{42}. Also", stop_reason=stop_reason),
    ]
    scores, ctx = score(completions, samples, enable_thinking=enable_thinking)
    assert [s.is_correct for s in scores] == [True, True]
    assert ctx["math_500_pass_at_1"] == 1.0
    assert ctx["math_500_unparsed_rate"] == 0.0
    # The strict view counts m1 wrong.
    assert ctx["math_500_blanked_truncated"] == blanked
    assert ctx["math_500_pass_at_1_blanked"] == (0.5 if blanked else 1.0)


# --- liquid_evals_math500 grader (dataset 2026.10.1) ---


def _le_sample(id, answer):
    return Math500Sample(id=id, prompt=f"Problem {id}", answer=answer, grader=Math500Grader.LIQUID_EVALS_MATH500)


def _filler(n):
    return "\n".join(f"step {i} of the work" for i in range(n))


@pytest.mark.parametrize(
    "completion, answer, expected, extraction",
    [
        ("So \\boxed{42}.", "42", True, "boxed"),
        ("So \\boxed{41}.", "42", False, "boxed"),
        # nested braces and nested boxes: the scan keeps the outer box's content
        ("\\boxed{\\frac{\\sqrt{3}}{2}}", "\\frac{\\sqrt{3}}{2}", True, "boxed"),
        ("\\boxed{\\boxed{7}}", "7", True, "boxed"),
        # \frac spellings and decimals are equivalent; an unreduced fraction is not
        ("\\boxed{\\dfrac{1}{2}}", "\\frac12", True, "boxed"),
        ("\\boxed{0.5}", "\\frac{1}{2}", True, "boxed"),
        ("\\boxed{2/4}", "\\frac{1}{2}", False, "boxed"),
        # units, degrees and dollars are dropped
        ("\\boxed{5\\text{ cm}}", "5", True, "boxed"),
        ("\\boxed{90^\\circ}", "90", True, "boxed"),
        ("\\boxed{\\$18.90}", "\\$18.90", True, "boxed"),
        # no box: the other patterns, then the last number
        ("The answer is 7", "7", True, "answer_is"),
        ("Final answer: 5", "5", True, "answer_colon"),
        ("It is 7 apples", "7", True, "numeric"),
        ("no answer here", "7", False, "none"),
        # nothing extracted, but a response line matches the text answer
        ("Hmm.\nMonday", "\\text{Monday}", True, "none"),
        # a wrong box is rescued by a bare line holding the answer ...
        ("\\boxed{3}\n4", "4", True, "boxed"),
        # ... within the last 150 lines, and not beyond
        ("4\n" + _filler(148) + "\n\\boxed{3}", "4", True, "boxed"),
        ("4\n" + _filler(149) + "\n\\boxed{3}", "4", False, "boxed"),
        # reasoning is stripped first: a box inside it does not count
        ("<think>\\boxed{42}</think>\nSo \\boxed{7}", "42", False, "boxed"),
        ("<think>\\boxed{7}</think>\nSo \\boxed{42}", "42", True, "boxed"),
    ],
)
def test_liquid_evals_single_sample(completion, answer, expected, extraction):
    scores, ctx = score([SampleCompletion(id="m1", completion=completion)], [_le_sample("m1", answer)])
    assert scores[0].is_correct is expected
    assert ctx["math_500_pass_at_1"] == (1.0 if expected else 0.0)
    assert ctx[f"math_500_extraction/{extraction}"] == 1
    assert ctx["math_500_unparsed_rate"] == (1.0 if extraction == "none" else 0.0)


def test_liquid_evals_context_schema_stable():
    samples = [_le_sample("m1", "1"), _le_sample("m2", "2")]
    empty_ctx = score([], samples)[1]
    populated_ctx = score([SampleCompletion(id="m1", completion="\\boxed{1}")], samples)[1]
    assert empty_ctx.keys() == populated_ctx.keys()
    for key, value in empty_ctx.items():
        assert type(value) is type(populated_ctx[key]), key
    # the headline keys match math_generic's; only the per-grader counts differ
    generic_ctx = score([], [])[1]
    shared = {k for k in generic_ctx if not k.startswith("math_500_method/")}
    assert shared == {k for k in empty_ctx if not k.startswith("math_500_extraction/")}
    for bucket in ["boxed", "hash", "answer_colon", "answer_is", "other", "numeric", "none"]:
        assert empty_ctx[f"math_500_extraction/{bucket}"] == 0
    assert not any(k.startswith("math_500_method/") for k in empty_ctx)
    assert populated_ctx["math_500_pass_at_1"] == 0.5


def test_liquid_evals_cut_reasoning_is_graded_raw_and_counted():
    samples = [_le_sample("m1", "42"), _le_sample("m2", "42")]
    completions = [
        SampleCompletion(id="m1", completion="So \\boxed{42}? Check the sign of", stop_reason="truncated"),
        SampleCompletion(id="m2", completion="So 42.</think>The answer is \\boxed{42}. Also", stop_reason="truncated"),
    ]
    scores, ctx = score(completions, samples, enable_thinking=True)
    assert [s.is_correct for s in scores] == [True, True]
    assert ctx["math_500_blanked_truncated"] == 1
    assert ctx["math_500_pass_at_1_blanked"] == 0.5


def test_liquid_evals_pass_at_1_over_repeats_is_the_mean_of_the_passes():
    # 2026.10.1 serves 5 repeats. The model-card number is the mean of 5 math500
    # runs of one pass each; pass@1 over every attempt equals that mean.
    gold = {"a": "1", "b": "2", "c": "3"}
    right = {("a", 0), ("a", 1), ("a", 2), ("a", 3), ("a", 4), ("b", 0), ("b", 3), ("c", 2)}
    samples = [_le_sample(f"{lid}#{k}", answer) for lid, answer in gold.items() for k in range(5)]
    completions = [
        SampleCompletion(id=f"{lid}#{k}", completion=f"\\boxed{{{answer if (lid, k) in right else 9}}}")
        for lid, answer in gold.items()
        for k in range(5)
    ]
    _, ctx = score(completions, samples)
    per_pass = [sum((lid, k) in right for lid in gold) / len(gold) for k in range(5)]
    assert ctx["math_500_pass_at_1"] == pytest.approx(sum(per_pass) / 5)
    assert ctx["math_500_pass_at_1_per_sample"] == {"a": 1.0, "b": 0.4, "c": 0.2}


def test_samples_of_two_graders_are_rejected():
    samples = [_sample("m1", "1"), _le_sample("m2", "1")]
    with pytest.raises(ValueError, match="more than one grader"):
        score([], samples)


def test_default_grader_is_math_generic():
    # A sample without a grader (2026.06.1's metadata has none) keeps score_math_generic.
    assert _sample("m1", "1").grader is Math500Grader.MATH_GENERIC
    _, ctx = score([SampleCompletion(id="m1", completion="\\boxed{1}")], [_sample("m1", "1")])
    assert ctx["math_500_method/prm800k"] == 1
    assert not any(k.startswith("math_500_extraction/") for k in ctx)
