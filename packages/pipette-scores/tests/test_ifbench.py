import math

import pytest

from pipette_scores.scoring.ifbench import score
from pipette_scores.types import IFBenchSample, SampleCompletion

_KW = {"keyword1": "banana", "keyword2": "apple", "keyword3": "cherry", "keyword4": "date", "keyword5": "fig"}
_PROMPT = "Use banana once, apple twice, cherry three times, date five times, fig seven times."
_PASS = " ".join(["banana"] * 1 + ["apple"] * 2 + ["cherry"] * 3 + ["date"] * 5 + ["fig"] * 7)


def _repeated_samples(logical_id, repeats):
    return [_sample(f"{logical_id}#{k}", ["count:keywords_multiple"], _PROMPT, kwargs=[_KW]) for k in range(repeats)]


def _sample(id, instruction_ids, prompt, kwargs=None):
    if kwargs is None:
        kwargs = tuple({} for _ in instruction_ids)
    return IFBenchSample(
        id=id,
        key=id,
        instruction_id_list=tuple(instruction_ids),
        prompt=prompt,
        kwargs=tuple(kwargs),
        n_constraints=len(instruction_ids),
    )


def test_score_follows_instruction():
    samples = [
        _sample(
            "1",
            ["count:keywords_multiple"],
            "Use the words banana once, apple twice, cherry three times, date five times, and fig seven times.",
            kwargs=[
                {
                    "keyword1": "banana",
                    "keyword2": "apple",
                    "keyword3": "cherry",
                    "keyword4": "date",
                    "keyword5": "fig",
                }
            ],
        )
    ]
    completion_text = " ".join(["banana"] * 1 + ["apple"] * 2 + ["cherry"] * 3 + ["date"] * 5 + ["fig"] * 7)
    completions = [SampleCompletion(id="1", completion=completion_text)]
    scores, _ = score(completions, samples)
    assert scores[0].is_correct is True


def test_score_fails_instruction():
    samples = [
        _sample(
            "1",
            ["count:keywords_multiple"],
            "Use the words banana once, apple twice, cherry three times, date five times, and fig seven times.",
            kwargs=[
                {
                    "keyword1": "banana",
                    "keyword2": "apple",
                    "keyword3": "cherry",
                    "keyword4": "date",
                    "keyword5": "fig",
                }
            ],
        )
    ]
    completions = [SampleCompletion(id="1", completion="nothing relevant")]
    scores, _ = score(completions, samples)
    assert scores[0].is_correct is False


def test_zero_samples():
    scores, _ = score([], [])
    assert scores == []


def test_thinking_tags_stripped():
    samples = [
        _sample(
            "1",
            ["count:keywords_multiple"],
            "Use banana once, apple twice, cherry three times, date five times, fig seven times.",
            kwargs=[
                {
                    "keyword1": "banana",
                    "keyword2": "apple",
                    "keyword3": "cherry",
                    "keyword4": "date",
                    "keyword5": "fig",
                }
            ],
        )
    ]
    body = " ".join(["banana"] * 1 + ["apple"] * 2 + ["cherry"] * 3 + ["date"] * 5 + ["fig"] * 7)
    completions = [SampleCompletion(id="1", completion=f"<think>planning</think>{body}")]
    scores, _ = score(completions, samples)
    assert scores[0].is_correct is True


def test_loose_passes_where_strict_fails():
    # A trailing sign-off line adds an extra "banana", breaking the exact
    # keyword count under strict. Loose drops the last line and recovers it,
    # so the prompt is correct under loose (the reported headline) but not strict.
    samples = [
        _sample(
            "1",
            ["count:keywords_multiple"],
            "Use banana once, apple twice, cherry three times, date five times, fig seven times.",
            kwargs=[
                {
                    "keyword1": "banana",
                    "keyword2": "apple",
                    "keyword3": "cherry",
                    "keyword4": "date",
                    "keyword5": "fig",
                }
            ],
        )
    ]
    body = " ".join(["banana"] * 1 + ["apple"] * 2 + ["cherry"] * 3 + ["date"] * 5 + ["fig"] * 7)
    completions = [SampleCompletion(id="1", completion=f"{body}\nThanks for reading banana")]
    scores, context = score(completions, samples)
    assert scores[0].is_correct is True
    assert context["loose_prompt_accuracy"] == 1.0
    assert context["strict_prompt_accuracy"] == 0.0


def test_score_context_keys():
    samples = [
        _sample(
            "1",
            ["count:keywords_multiple"],
            "Use banana once, apple twice, cherry three times, date five times, fig seven times.",
            kwargs=[
                {
                    "keyword1": "banana",
                    "keyword2": "apple",
                    "keyword3": "cherry",
                    "keyword4": "date",
                    "keyword5": "fig",
                }
            ],
        )
    ]
    completions = [SampleCompletion(id="1", completion="placeholder")]
    _, context = score(completions, samples)
    assert set(context) == {
        "strict_prompt_accuracy",
        "strict_instruction_accuracy",
        "loose_prompt_accuracy",
        "loose_instruction_accuracy",
        "ifbench_pass_at_1",
        "ifbench_pass_at_1_stderr",
        "ifbench_pass_at_1_per_sample",
        "strict/count",  # tier0: category prefix
        "loose/count",
        "strict/count:keywords_multiple",  # tier1: full instruction id
        "loose/count:keywords_multiple",
    }
    assert context["strict/count"] == 0.0
    assert context["loose/count"] == 0.0
    assert context["strict/count:keywords_multiple"] == 0.0
    assert context["loose/count:keywords_multiple"] == 0.0


def test_tier0_groups_multiple_categories():
    # Two `count:*` samples (both passing) and one `words:*` sample exercise the
    # category split across distinct prefixes and the per-category aggregation.
    kw = {
        "keyword1": "banana",
        "keyword2": "apple",
        "keyword3": "cherry",
        "keyword4": "date",
        "keyword5": "fig",
    }
    prompt = "Use banana once, apple twice, cherry three times, date five times, fig seven times."
    body = " ".join(["banana"] * 1 + ["apple"] * 2 + ["cherry"] * 3 + ["date"] * 5 + ["fig"] * 7)
    samples = [
        _sample("a", ["count:keywords_multiple"], prompt, kwargs=[kw]),
        _sample("b", ["count:keywords_multiple"], prompt, kwargs=[kw]),
        _sample("c", ["words:vowel"], "Write a short paragraph."),
    ]
    completions = [
        SampleCompletion(id="a", completion=body),
        SampleCompletion(id="b", completion=body),
        SampleCompletion(id="c", completion="The quick brown fox jumps over the lazy dog."),
    ]
    _, context = score(completions, samples)

    # Both category prefixes are present.
    assert {"strict/count", "loose/count", "strict/words", "loose/words"} <= set(context)
    # `count` aggregates its two passing members: 2/2 = 1.0.
    assert context["strict/count"] == 1.0
    assert context["loose/count"] == 1.0
    # Single-member `words` category equals its sole tier1 instruction.
    assert context["strict/words"] == context["strict/words:vowel"]
    assert context["loose/words"] == context["loose/words:vowel"]


def test_pass_at_1_over_repeats():
    # One logical sample run 5×: 3 attempts pass, 2 fail → pass@1 = 3/5.
    samples = _repeated_samples("1", 5)
    completions = [SampleCompletion(id=f"1#{k}", completion=_PASS) for k in range(3)] + [
        SampleCompletion(id=f"1#{k}", completion="nope") for k in range(3, 5)
    ]
    scores, context = score(completions, samples)
    assert context["ifbench_pass_at_1"] == 0.6
    assert context["ifbench_pass_at_1_per_sample"] == {"1": 0.6}
    # Headline equals correct/total over the boolean rows, by construction.
    correct = sum(1 for s in scores if s.is_correct)
    assert context["ifbench_pass_at_1"] == correct / len(samples)


def test_pass_at_1_missing_attempt_counts_as_fail():
    # 5 attempts expected, only 2 (both passing) arrive. Denominator is N·R from
    # `samples` (5), not received completions (2), so the 3 missing attempts fail.
    samples = _repeated_samples("1", 5)
    completions = [SampleCompletion(id=f"1#{k}", completion=_PASS) for k in range(2)]
    _, context = score(completions, samples)
    assert context["ifbench_pass_at_1"] == 0.4
    assert context["ifbench_pass_at_1_per_sample"] == {"1": 0.4}
    assert context["ifbench_pass_at_1_stderr"] == pytest.approx(math.sqrt(0.4 * 0.6 / 5))


def test_pass_at_1_groups_by_logical_id():
    # Two logical samples, 2 repeats each: "a" passes 1/2, "b" passes 2/2.
    samples = _repeated_samples("a", 2) + _repeated_samples("b", 2)
    completions = [
        SampleCompletion(id="a#0", completion=_PASS),
        SampleCompletion(id="a#1", completion="nope"),
        SampleCompletion(id="b#0", completion=_PASS),
        SampleCompletion(id="b#1", completion=_PASS),
    ]
    _, context = score(completions, samples)
    assert context["ifbench_pass_at_1_per_sample"] == {"a": 0.5, "b": 1.0}
    assert context["ifbench_pass_at_1"] == 0.75  # 3/4 boolean rows


def test_pass_at_1_repeats_one_transparent():
    # No #k suffix: each id is its own logical id, so repeats=1 is a no-op.
    samples = [_sample("1", ["count:keywords_multiple"], _PROMPT, kwargs=[_KW])]
    completions = [SampleCompletion(id="1", completion=_PASS)]
    _, context = score(completions, samples)
    assert context["ifbench_pass_at_1"] == 1.0
    assert context["ifbench_pass_at_1_per_sample"] == {"1": 1.0}
