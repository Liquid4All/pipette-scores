import pytest
from pydantic import ValidationError

from pipette_scores.scoring.gpqa_diamond import score
from pipette_scores.types import GPQADiamondSample, SampleCompletion


def _sample(id, answer):
    return GPQADiamondSample(id=id, prompt=f"Question {id}\nA) a\nB) b\nC) c\nD) d", answer=answer)


@pytest.mark.parametrize("good, normalized", [("A", "A"), ("d", "D"), (" b ", "B")])
def test_answer_normalized(good, normalized):
    assert GPQADiamondSample(id="x", prompt="q", answer=good).answer == normalized


@pytest.mark.parametrize("bad", ["E", "", "AB", "1"])
def test_answer_out_of_domain_rejected(bad):
    with pytest.raises(ValidationError):
        GPQADiamondSample(id="x", prompt="q", answer=bad)


@pytest.mark.parametrize(
    "completion, answer, expected",
    [
        ("...reasoning... Answer: B", "B", True),
        ("Answer: A", "C", False),
        # reasoning is stripped before extraction, so the in-think "C" is ignored
        ("<think>maybe C</think> Answer: D", "D", True),
        # same for a Gemma 4 thought channel; one cut off is all reasoning
        ("<|channel>thought\nmaybe C<channel|>Answer: D", "D", True),
        ("<|channel>thought\nAnswer: D? Check the units first", "D", False),
        # extractor only accepts A-D, so an out-of-range / unparseable answer fails
        ("I am not sure", "A", False),
    ],
)
def test_single_sample(completion, answer, expected):
    scores, ctx = score([SampleCompletion(id="q1", completion=completion)], [_sample("q1", answer)])
    assert scores[0].is_correct is expected
    assert ctx["gpqa_diamond_pass_at_1"] == (1.0 if expected else 0.0)


def test_pass_at_1_over_repeats():
    # one logical question salted into 4 repeats (#0..#3); 2 of 4 correct -> 0.5
    samples = [_sample(f"q1#{k}", "B") for k in range(4)]
    completions = [
        SampleCompletion(id="q1#0", completion="Answer: B"),
        SampleCompletion(id="q1#1", completion="Answer: B"),
        SampleCompletion(id="q1#2", completion="Answer: A"),
        SampleCompletion(id="q1#3", completion="Answer: C"),
    ]
    scores, ctx = score(completions, samples)
    assert sum(s.is_correct for s in scores) == 2
    assert ctx["gpqa_diamond_pass_at_1"] == 0.5
    assert ctx["gpqa_diamond_pass_at_1_per_sample"] == {"q1": 0.5}


def test_missing_attempt_counts_as_fail():
    # 2 samples but only 1 completion -> denominator is sample count (2)
    samples = [_sample("q1", "A"), _sample("q2", "B")]
    scores, ctx = score([SampleCompletion(id="q1", completion="Answer: A")], samples)
    assert len(scores) == 1
    assert ctx["gpqa_diamond_pass_at_1"] == 0.5


def test_context_schema_stable_regardless_of_availability():
    # Same keys + same value types whether or not there's data — callers
    # (mgmt/warehouse) rely on a consistent context schema.
    empty_ctx = score([], [])[1]
    populated_ctx = score([SampleCompletion(id="q1", completion="Answer: A")], [_sample("q1", "A")])[1]
    assert empty_ctx.keys() == populated_ctx.keys()
    for key, value in empty_ctx.items():
        assert type(value) is type(populated_ctx[key]), key
    assert isinstance(empty_ctx["gpqa_diamond_pass_at_1"], float)
    assert isinstance(empty_ctx["gpqa_diamond_pass_at_1_stderr"], float)
    assert isinstance(empty_ctx["gpqa_diamond_unparsed_rate"], float)
    assert empty_ctx["gpqa_diamond_blanked_truncated"] == 0
    assert isinstance(empty_ctx["gpqa_diamond_pass_at_1_blanked"], float)
    assert isinstance(empty_ctx["gpqa_diamond_pass_at_1_per_sample"], dict)
    # flat, fixed choice keys — always present (count 0 when a letter is unused)
    for choice in ["A", "B", "C", "D", "none"]:
        assert empty_ctx[f"gpqa_diamond_choice/{choice}"] == 0


def test_context_extraction_diagnostics():
    samples = [_sample("q1", "A"), _sample("q2", "A"), _sample("q3", "A")]
    completions = [
        SampleCompletion(id="q1", completion="Answer: A"),  # extracted A
        SampleCompletion(id="q2", completion="Answer: B"),  # extracted B (wrong)
        SampleCompletion(id="q3", completion="totally unparseable"),  # none
    ]
    _, ctx = score(completions, samples)
    # one of three couldn't be parsed -> distinguishes parse-miss from wrong answer
    assert ctx["gpqa_diamond_unparsed_rate"] == pytest.approx(1 / 3)
    assert ctx["gpqa_diamond_choice/A"] == 1
    assert ctx["gpqa_diamond_choice/B"] == 1
    assert ctx["gpqa_diamond_choice/C"] == 0
    assert ctx["gpqa_diamond_choice/D"] == 0
    assert ctx["gpqa_diamond_choice/none"] == 1


@pytest.mark.parametrize("stop_reason", ["truncated", "doom_loop"])
@pytest.mark.parametrize("enable_thinking, blanked", [(True, 2), (False, 0), (None, 0)])
def test_cut_reasoning_is_graded_raw_and_counted(stop_reason, enable_thinking, blanked):
    samples = [_sample("q1", "B"), _sample("q2", "B"), _sample("q3", "B"), _sample("q4", "B")]
    completions = [
        # The prompt opened the reasoning and generation was cut: graded whole,
        # as liquid_evals grades it.
        SampleCompletion(id="q1", completion="Option B fits. Answer: B. Double-check option", stop_reason=stop_reason),
        # The completion opened <think> itself: nothing is left to grade.
        SampleCompletion(id="q2", completion="<think>Option B fits. Answer: B. Double-check", stop_reason=stop_reason),
        # The reasoning closed.
        SampleCompletion(id="q3", completion="Option B fits.\n</think>\nAnswer: B", stop_reason=stop_reason),
        SampleCompletion(id="q4", completion="Option B fits.\n</think>\nAnswer: B", stop_reason="eos"),
    ]
    scores, ctx = score(completions, samples, enable_thinking=enable_thinking)
    assert [s.is_correct for s in scores] == [True, False, True, True]
    assert ctx["gpqa_diamond_pass_at_1"] == 0.75
    assert ctx["gpqa_diamond_choice/none"] == 1
    # The strict view counts q1 and q2 wrong; only q1 was correct.
    assert ctx["gpqa_diamond_blanked_truncated"] == blanked
    assert ctx["gpqa_diamond_pass_at_1_blanked"] == (0.5 if blanked else 0.75)
