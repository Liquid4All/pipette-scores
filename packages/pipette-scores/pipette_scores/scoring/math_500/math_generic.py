"""Generic math verifier with multi-stage validation.

Two-stage verification:
1. PRM800K script-based grading (AIME-like, but allowing non-ints & negatives)
2. Openbench robust_boxed extraction + normalization

This is a generalization of the AIME verifier that works for any math problem,
not just competition math with integer 0-999 constraints. The optional
LLM-equality fallback from upstream is dropped in this extraction.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MathResult:
    """Result of math answer extraction and scoring."""

    extracted: str | None
    expected: str
    correct: bool
    method: str  # "prm800k", "robust_boxed", "no_match", "failed"
    details: dict[str, Any] | None = None


# =============================================================================
# Answer extraction — implements the Artificial Analysis methodology
# =============================================================================

# Pattern to extract content from \boxed{}, \fbox{}, \framebox{}
# This handles one level of nested braces (e.g., \frac{1}{2})
BOXED_PATTERN = re.compile(r"\\(?:boxed|fbox|framebox)\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

# Pattern for last number in text (integer or decimal)
LAST_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def extract_boxed_answer(response: str) -> str | None:
    """Extract answer from LaTeX boxed notation.

    Looks for \\boxed{}, \\fbox{}, or \\framebox{} and extracts content.
    Handles nested braces.

    Args:
        response: Model's response text

    Returns:
        Extracted content or None if not found
    """
    matches = list(BOXED_PATTERN.finditer(response))
    if matches:
        # Take the last boxed answer (accounts for self-correction)
        return matches[-1].group(1).strip()
    return None


def extract_last_number(response: str) -> str | None:
    """Extract the last number from response text.

    Args:
        response: Model's response text

    Returns:
        Last number found, or None
    """
    matches = list(LAST_NUMBER_PATTERN.finditer(response))
    if matches:
        return matches[-1].group(0)
    return None


# =============================================================================
# Normalization - copied from openbench (openbench/scorers/robust_boxed.py)
# =============================================================================


def normalize_numeric_answer(answer: str) -> str | None:
    """Normalize a numeric answer for comparison.

    Handles various number formats including:
    - Removing commas
    - Extracting leading integers
    - Removing trailing zeros after decimal

    Copied from openbench robust_boxed.py:61-97

    Args:
        answer: The answer string to normalize

    Returns:
        Normalized answer string, or None if not a valid number
    """
    if not answer:
        return None

    # Remove commas and extra whitespace
    answer = answer.replace(",", "").strip()

    # Try to extract integer from start (for AIME-style answers)
    match = re.match(r"^-?\d+", answer)
    if match:
        return match.group(0)

    # Try to parse as float and normalize
    try:
        num = float(answer)
        # If it's a whole number, return as integer
        if num == int(num):
            return str(int(num))
        # Otherwise remove trailing zeros
        return str(num).rstrip("0").rstrip(".")
    except (ValueError, TypeError):
        return None


# =============================================================================
# PRM800K grading (per the Artificial Analysis methodology)
# =============================================================================


def _get_prm800k_grade_answer():
    """Load PRM800K's `grade_answer` (vendored under `prm800k.grading`)."""
    from pipette_scores.scoring.math_500.prm800k.grading import grade_answer

    return grade_answer


# Cache the grade_answer function after first successful load
_cached_grade_answer = None


def grade_with_prm800k(given_answer: str, ground_truth: str) -> bool:
    """Grade an answer using the OpenAI PRM800K grading script.

    This is the exact grading code referenced by Artificial Analysis:
    https://github.com/openai/prm800k/blob/main/prm800k/grading/grader.py

    Args:
        given_answer: The extracted answer from the model
        ground_truth: The expected answer

    Returns:
        True if the answer is correct, False otherwise
    """
    global _cached_grade_answer
    if _cached_grade_answer is None:
        _cached_grade_answer = _get_prm800k_grade_answer()

    return _cached_grade_answer(given_answer, ground_truth)


# =============================================================================
# Core scoring function
# =============================================================================


def score_math_generic(
    response: str,
    ground_truth: str,
) -> MathResult:
    """Score a math response using multi-stage validation.

    Stage 1: PRM800K grader (AIME-like but allowing non-ints & negatives)
             - Extract from boxed, fall back to last number
             - No range validation (unlike AIME's 0-999)
             - Use PRM800K for symbolic equivalence

    Stage 2: Openbench robust_boxed (if PRM800K fails)
             - Extract from boxed, fall back to last number
             - Normalize both sides
             - String comparison

    Args:
        response: Model's response text
        ground_truth: Expected answer

    Returns:
        MathResult with extraction details and correctness
    """
    expected = ground_truth.strip()
    details: dict[str, Any] = {}

    # =========================================================================
    # Stage 1: PRM800K (AIME-like but allowing non-ints & negatives)
    # Per the Artificial Analysis methodology, but without the integer 0-999 constraint
    # =========================================================================

    # Step 1a: Extract answer from \boxed{} first
    extracted = extract_boxed_answer(response)
    if extracted:
        details["extraction_method"] = "boxed"
    else:
        # Fall back to last number (not just integer like AIME)
        extracted = extract_last_number(response)
        if extracted:
            details["extraction_method"] = "last_number"

    if extracted is None:
        return MathResult(
            extracted=None,
            expected=expected,
            correct=False,
            method="failed",
            details={"error": "extraction_failed"},
        )

    details["extracted_raw"] = extracted

    # Step 1b: Use PRM800K grader for script-based grading
    # Unlike AIME, we skip the integer validation and range check
    try:
        is_correct = grade_with_prm800k(extracted, expected)
        if is_correct:
            details["prm800k_match"] = True
            return MathResult(
                extracted=extracted,
                expected=expected,
                correct=True,
                method="prm800k",
                details=details,
            )
        else:
            details["prm800k_match"] = False
    except (ImportError, ModuleNotFoundError):
        # Missing dependencies should fail loudly - don't silently fall back
        raise
    except Exception as e:
        # Other errors (sympy parsing, etc.) are expected for some inputs
        details["prm800k_error"] = str(e)
        logger.debug(f"PRM800K grading failed for '{extracted}' vs '{expected}': {e}")

    # =========================================================================
    # Stage 2: Openbench robust_boxed extraction + normalization
    # Based on openbench's robust_boxed_scorer
    # =========================================================================

    # Normalize both sides
    extracted_norm = normalize_numeric_answer(extracted)
    expected_norm = normalize_numeric_answer(expected)
    details["extracted_normalized"] = extracted_norm
    details["expected_normalized"] = expected_norm

    # Compare normalized values
    if extracted_norm is not None and expected_norm is not None:
        if extracted_norm == expected_norm:
            return MathResult(
                extracted=extracted,
                expected=expected,
                correct=True,
                method="robust_boxed",
                details=details,
            )

    return MathResult(
        extracted=extracted,
        expected=expected,
        correct=False,
        method="no_match",
        details=details,
    )
