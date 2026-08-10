"""math_generic verifier tests (MATH-500). Expectations reflect the actual
PRM800K-backed grader behavior, probed against the vendored grader."""

import pytest

from pipette_scores.scoring.math_500.math_generic import (
    extract_boxed_answer,
    extract_last_number,
    normalize_numeric_answer,
    score_math_generic,
)
from pipette_scores.scoring.math_500.prm800k.grading import grade_answer


@pytest.mark.parametrize(
    "response, ground_truth, correct, method",
    [
        # boxed extraction + exact match
        (r"final answer: \boxed{42}", "42", True, "prm800k"),
        # last-number fallback when no \boxed
        ("the result is 7", "7", True, "prm800k"),
        # symbolic equivalence: decimal <-> fraction
        (r"\boxed{0.5}", "1/2", True, "prm800k"),
        (r"\boxed{\frac{1}{2}}", "0.5", True, "prm800k"),
        # negatives are allowed (unlike AIME 0-999)
        (r"so x = -3 hence \boxed{-3}", "-3", True, "prm800k"),
        # wrong answer
        (r"\boxed{3}", "4", False, "no_match"),
        # unreduced fraction is NOT normalized by the grader -> not equal
        (r"\boxed{2/4}", "1/2", False, "no_match"),
        # stage 2 (robust_boxed): PRM800K rejects, leading-int normalization matches
        (r"\boxed{12abc}", "12", True, "robust_boxed"),
        # nothing numeric to extract
        ("no numbers in this response", "5", False, "failed"),
    ],
)
def test_score_math_generic(response, ground_truth, correct, method):
    result = score_math_generic(response, ground_truth)
    assert result.correct is correct
    assert result.method == method


@pytest.mark.parametrize(
    "response, expected",
    [
        (r"answer is \boxed{123}", "123"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        ("no box here", None),
    ],
)
def test_extract_boxed_answer(response, expected):
    assert extract_boxed_answer(response) == expected


@pytest.mark.parametrize(
    "response, expected",
    [
        ("the value is 256", "256"),
        ("first 12 then 34", "34"),  # last number wins
        ("temperature -7 degrees", "-7"),
        ("no digits at all", None),
    ],
)
def test_extract_last_number(response, expected):
    assert extract_last_number(response) == expected


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("1,000", "1000"),  # commas stripped
        ("  42 ", "42"),  # whitespace stripped
        ("12abc", "12"),  # leading integer extracted
        ("5.0", "5"),  # leading-int match wins over float path
        ("3.140", "3"),  # ditto — leading int, not "3.14"
        ("abc", None),  # no number
        ("", None),  # empty
    ],
)
def test_normalize_numeric_answer(answer, expected):
    assert normalize_numeric_answer(answer) == expected


@pytest.mark.parametrize(
    "given, gold, correct",
    [
        ("0.5", "1/2", True),  # symbolic equivalence
        (r"\frac{1}{2}", "0.5", True),  # latex fraction <-> decimal
        ("3", "4", False),  # plain mismatch
    ],
)
def test_prm800k_grade_answer(given, gold, correct):
    assert grade_answer(given, gold) is correct
