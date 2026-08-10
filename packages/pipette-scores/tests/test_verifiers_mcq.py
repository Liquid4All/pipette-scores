"""MCQ verifier tests — GPQA Diamond uses valid_options="ABCD"."""

import pytest

from pipette_scores.scoring.gpqa_diamond.mcq import extract_mcq_answer, score_mcq


@pytest.mark.parametrize(
    "response, letter, pattern",
    [
        ("B", "B", "single_letter"),
        ("...reasoning... Answer: C", "C", "primary"),
        ("Therefore the **Answer:** D", "D", "primary"),
        # AA methodology takes the LAST match (model self-corrects)
        ("Answer: A. Wait, no. Answer: C", "C", "primary"),
        (r"so \boxed{A}", "A", "boxed"),
        ("I think the answer is b.", "B", "answer_is"),
        ("E is the correct answer", "E", "explicit"),
        # remaining fallback patterns (each input is crafted to fall through to
        # exactly this pattern; expected values probed against the code)
        ("I believe the answer is (C)", "C", "answer_is_paren"),
        ("C) this is the chosen option text", "C", "choice_format"),
        ("after weighing it, my pick is\nD", "D", "standalone_end"),
        ("The reasoning leads to F.", "F", "letter_period"),
        ("so the pick, ultimately, is G,", "G", "letter_nonword"),
    ],
)
def test_extract_patterns(response, letter, pattern):
    assert extract_mcq_answer(response) == (letter, pattern)


@pytest.mark.parametrize(
    "response, options",
    [
        # all-lowercase, no standalone capital for the fallback regexes to grab
        ("i am not sure which option fits here", "ABCDEFGHIJ"),
        # "E" is extracted by the regex but rejected against ABCD valid options
        ("Answer: E", "ABCD"),
    ],
)
def test_extract_returns_none(response, options):
    assert extract_mcq_answer(response, valid_options=options) == (None, None)


@pytest.mark.parametrize(
    "response, ground_truth, correct, extracted",
    [
        ("Answer: C", "C", True, "C"),
        ("Answer: A", "C", False, "A"),
        ("no clear choice here", "B", False, None),
    ],
)
def test_score_mcq(response, ground_truth, correct, extracted):
    result = score_mcq(response, ground_truth, valid_options="ABCD")
    assert result.correct is correct
    assert result.extracted == extracted
    assert result.expected == ground_truth
