"""Multiple choice answer extraction and scoring.

Implements the Artificial Analysis MCQ extraction methodology:
- Primary pattern: "Answer: X" format with optional markdown
- Multiple fallback patterns for various answer formats
- Always takes the last match to account for self-correction

Reference: Artificial Analysis Intelligence Benchmarking Methodology V3.0
"""

import re
from dataclasses import dataclass


@dataclass
class MCQResult:
    """Result of MCQ extraction and scoring."""

    extracted: str | None
    expected: str
    correct: bool
    pattern_used: str | None


# Primary pattern: "Answer: X" with optional markdown formatting
# (?i) = case insensitive
# [\*\_]{0,2} = optional markdown (**, __, *, _)
# \s* = optional whitespace
# ([A-Z]) = capture single letter
# (?![a-zA-Z0-9]) = not followed by alphanumeric (word boundary)
PRIMARY_PATTERN = re.compile(r"(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])")

# Fallback patterns in order of precedence (per AA methodology)
FALLBACK_PATTERNS = [
    # LaTeX boxed notation: \boxed{A} or \boxed{The answer is A}
    (re.compile(r"\\boxed\{[^}]*([A-Z])[^}]*\}"), "boxed"),
    # Natural language: "answer is B"
    (re.compile(r"answer is ([A-Za-z])", re.IGNORECASE), "answer_is"),
    # With parenthesis: "answer is (C" (AA spec shows \\( but example shows plain paren)
    (re.compile(r"answer is \(([A-Za-z])", re.IGNORECASE), "answer_is_paren"),
    # Choice format: "D) some answer text"
    (re.compile(r"([A-Z])\)\s*[^A-Z]*"), "choice_format"),
    # Explicit statement: "E is the correct answer"
    (re.compile(r"([A-Z])\s+is\s+the\s+correct\s+answer", re.IGNORECASE), "explicit"),
    # Standalone letter at end of response
    (re.compile(r"([A-Z])\s*$"), "standalone_end"),
    # Letter followed by period: "F."
    (re.compile(r"([A-Z])\s*\."), "letter_period"),
    # Letter followed by non-word character
    (re.compile(r"([A-Z])\s*[^\w]"), "letter_nonword"),
]


def extract_mcq_answer(
    response: str,
    valid_options: str = "ABCDEFGHIJ",
) -> tuple[str | None, str | None]:
    """Extract multiple choice answer from model response.

    Implements the Artificial Analysis multi-stage extraction approach:
    1. For single-letter responses, use directly
    2. Try primary "Answer: X" pattern
    3. Fall back through alternative patterns
    4. Always take the LAST match (accounts for self-correction)

    Args:
        response: Model's response text
        valid_options: Valid option letters (default A-Z for 10-option MCQ)

    Returns:
        Tuple of (extracted_answer, pattern_used) or (None, None) if no match
    """
    response = response.strip()

    # Single letter response
    if len(response) == 1 and response.upper() in valid_options:
        return response.upper(), "single_letter"

    # Try primary pattern - get last match
    matches = list(PRIMARY_PATTERN.finditer(response))
    if matches:
        extracted = matches[-1].group(1).upper()
        if extracted in valid_options:
            return extracted, "primary"

    # Try fallback patterns
    for pattern, name in FALLBACK_PATTERNS:
        matches = list(pattern.finditer(response))
        if matches:
            extracted = matches[-1].group(1).upper()
            if extracted in valid_options:
                return extracted, name

    return None, None


def score_mcq(
    response: str,
    ground_truth: str,
    valid_options: str = "ABCDEFGHIJ",
) -> MCQResult:
    """Score a multiple choice response.

    Args:
        response: Model's response text
        ground_truth: Expected answer letter (e.g., "A", "B")
        valid_options: Valid option letters

    Returns:
        MCQResult with extraction details and correctness
    """
    expected = ground_truth.strip().upper()
    extracted, pattern = extract_mcq_answer(response, valid_options)

    correct = extracted is not None and extracted == expected

    return MCQResult(
        extracted=extracted,
        expected=expected,
        correct=correct,
        pattern_used=pattern,
    )
