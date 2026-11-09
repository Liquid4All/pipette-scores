import math
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pipette_scores.scoring.math_500 import liquid_evals_parser
from pipette_scores.scoring.math_500.math_generic import score_math_generic

from pipette_scores.repeats import logical_id
from pipette_scores.scoring.completion_text import grading_text, remove_thinking_tags
from pipette_scores.types import Math500Grader, Math500Sample, SampleCompletion, SampleScore


def _grade_math_generic(text: str, gold: str) -> tuple[bool, str]:
    result = score_math_generic(text, gold)
    return result.correct, result.method


# extract_math_answer's patterns, in its priority order: the first that finds
# an answer is the one its prediction came from.
_LIQUID_EVALS_EXTRACTORS = (
    ("boxed", liquid_evals_parser._extract_boxed_pattern),
    ("hash", liquid_evals_parser._extract_hash_pattern),
    ("answer_colon", liquid_evals_parser._extract_answer_colon_pattern),
    ("answer_is", liquid_evals_parser._extract_answer_is_pattern),
    ("other", liquid_evals_parser._extract_other_patterns),
    ("numeric", liquid_evals_parser._extract_numeric_fallback),
)


def liquid_evals_extraction(text: str) -> str:
    """Name the extract_math_answer pattern that yields ``text``'s answer, or ``none``."""
    return next((name for name, extract in _LIQUID_EVALS_EXTRACTORS if extract(text)), "none")


def _grade_liquid_evals(text: str, gold: str) -> tuple[bool, str]:
    # tasks/math500.py: predicted = extract_math_answer(model_response), then
    # verify_math_answer(true_label, predicted, model_response).
    predicted = liquid_evals_parser.extract_math_answer(text)
    correct = liquid_evals_parser.verify_math_answer(gold, predicted, text)
    return bool(correct), liquid_evals_extraction(text)


@dataclass(frozen=True)
class _Grader:
    # (grading text, gold answer) -> (is correct, diagnostic bucket)
    grade: Callable[[str, str], tuple[bool, str]]
    # Context key prefix of the bucket counts, and every bucket, so the key set
    # stays fixed whichever buckets were hit.
    bucket_key: str
    buckets: tuple[str, ...]
    # The buckets in which the grader found no usable answer.
    unparsed: frozenset[str]


_GRADERS = {
    # score_math_generic's grading paths.
    Math500Grader.MATH_GENERIC: _Grader(
        grade=_grade_math_generic,
        bucket_key="math_500_method",
        buckets=("prm800k", "robust_boxed", "no_match", "failed"),
        unparsed=frozenset({"no_match", "failed"}),
    ),
    # The extract_math_answer pattern the prediction came from.
    Math500Grader.LIQUID_EVALS_MATH500: _Grader(
        grade=_grade_liquid_evals,
        bucket_key="math_500_extraction",
        buckets=(*(name for name, _ in _LIQUID_EVALS_EXTRACTORS), "none"),
        unparsed=frozenset({"none"}),
    ),
}


def _dataset_grader(samples: list[Math500Sample]) -> Math500Grader:
    graders = {s.grader for s in samples}
    if len(graders) > 1:
        raise ValueError(f"MATH-500 samples name more than one grader: {sorted(graders)}")
    return graders.pop() if graders else Math500Grader.MATH_GENERIC


def score(
    completions: list[SampleCompletion], samples: list[Math500Sample], *, enable_thinking: bool | None = None
) -> tuple[list[SampleScore], dict[str, Any]]:
    """Score MATH-500 completions against the gold answer.

    The dataset version picks the grader (``metadata.grader``, carried on each
    sample):

    - ``math_generic`` (2026.06.1, liquid_evals' math500_std track): this
      eval's ``math_generic.score_math_generic`` (boxed-answer extraction →
      PRM800K symbolic grader → robust-boxed fallback).
    - ``liquid_evals_math500`` (2026.10.1, liquid_evals' plain math500 task):
      ``extract_math_answer`` then ``verify_math_answer``, vendored verbatim in
      ``liquid_evals_parser`` — mathd/sympy equivalence on the extracted answer,
      then on each of the response's last 150 lines.

    Either way the completion's reasoning is stripped first via
    ``grading_text`` and ``remove_thinking_tags``; the grader does its own
    answer extraction from what is left.

    Datasets may run each question with repeats (R) and report pass@1 over the
    N·R rows; repeats are realized as ``#k`` salted ids (see ``repeats.expand``).
    The pass@1 denominator is the SAMPLE count from ``samples`` (not received
    completions), so a missing attempt counts as a fail — matching the management
    "missing = incorrect" contract.

    Context keys: ``math_500_pass_at_1`` (+ ``_stderr`` and ``_per_sample``
    breakdown by logical id), ``math_500_unparsed_rate`` (fraction of received
    completions the grader found no usable answer in),
    ``math_500_blanked_truncated`` (completions cut short inside their
    reasoning, which the strict view grades as empty, see ``grading_text``),
    ``math_500_pass_at_1_blanked`` (pass@1 with those counted wrong), and flat
    per-grader counts: ``math_500_method/{prm800k,robust_boxed,no_match,failed}``
    for math_generic, and
    ``math_500_extraction/{boxed,hash,answer_colon,answer_is,other,numeric,none}``
    (the extract_math_answer pattern that found the answer) for
    liquid_evals_math500.
    """
    grader = _GRADERS[_dataset_grader(samples)]
    by_id = {s.id: s for s in samples}
    scores: list[SampleScore] = []
    correct = 0
    # Which grading path each completion took, so the context can separate a
    # wrong answer from one the grader couldn't extract an answer out of.
    buckets: Counter[str] = Counter()
    blanked = 0
    blanked_correct = 0

    for c in completions:
        sample = by_id[c.id]
        text, cut = grading_text(c, enable_thinking)
        text = remove_thinking_tags(text)
        is_correct, bucket = grader.grade(text, sample.answer)
        scores.append(SampleScore(id=c.id, is_correct=is_correct))
        if is_correct:
            correct += 1
        blanked += cut
        blanked_correct += cut and is_correct
        buckets[bucket] += 1

    n_samples = len(samples)
    per_sample_total: dict[str, int] = defaultdict(int)
    for s in samples:
        per_sample_total[logical_id(s.id)] += 1
    per_sample_correct: dict[str, int] = defaultdict(int)
    for sc in scores:
        if sc.is_correct:
            per_sample_correct[logical_id(sc.id)] += 1

    pass_at_1 = correct / n_samples if n_samples else 0.0
    n_completions = len(completions)
    # No usable answer extracted — separates "wrong answer" from "couldn't
    # parse an answer".
    unparsed = sum(buckets[b] for b in grader.unparsed)
    context: dict[str, Any] = {
        "math_500_pass_at_1": pass_at_1,
        "math_500_pass_at_1_stderr": (math.sqrt(pass_at_1 * (1.0 - pass_at_1) / n_samples) if n_samples else 0.0),
        "math_500_pass_at_1_per_sample": {
            lid: per_sample_correct[lid] / per_sample_total[lid] for lid in sorted(per_sample_total)
        },
        "math_500_unparsed_rate": unparsed / n_completions if n_completions else 0.0,
        "math_500_blanked_truncated": blanked,
        "math_500_pass_at_1_blanked": (correct - blanked_correct) / n_samples if n_samples else 0.0,
    }
    # Flat, fixed-key grading-path distribution — always emit every key so the
    # context schema stays stable regardless of which paths were hit.
    for bucket in grader.buckets:
        context[f"{grader.bucket_key}/{bucket}"] = buckets[bucket]
    return scores, context
