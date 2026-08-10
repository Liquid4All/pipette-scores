import math
from collections import Counter, defaultdict
from typing import Any

from pipette_scores.scoring.math_500.math_generic import score_math_generic

from pipette_scores.repeats import logical_id
from pipette_scores.scoring.completion_text import remove_thinking_tags
from pipette_scores.types import Math500Sample, SampleCompletion, SampleScore

# Extraction/grading methods reported by score_math_generic.
_METHODS = ("prm800k", "robust_boxed", "no_match", "failed")


def score(
    completions: list[SampleCompletion], samples: list[Math500Sample]
) -> tuple[list[SampleScore], dict[str, Any]]:
    """Score MATH-500 completions against the gold answer.

    Each completion is graded by this eval's ``math_generic.score_math_generic``
    (boxed-answer extraction → PRM800K symbolic grader → robust-boxed fallback)
    after reasoning is stripped via ``remove_thinking_tags``. The raw text is
    passed through — the grader does its own ``\\boxed{}``/last-number extraction.

    AA methodology runs each question with repeats (R) and reports pass@1 over the
    N·R rows; repeats are realized as ``#k`` salted ids (see ``repeats.expand``).
    The pass@1 denominator is the SAMPLE count from ``samples`` (not received
    completions), so a missing attempt counts as a fail — matching the management
    "missing = incorrect" contract.

    Context keys: ``math_500_pass_at_1`` (+ ``_stderr`` and ``_per_sample``
    breakdown by logical id), ``math_500_unparsed_rate`` (fraction of received
    completions the grader found no usable answer in), and flat
    ``math_500_method/{prm800k,robust_boxed,no_match,failed}`` counts.
    """
    by_id = {s.id: s for s in samples}
    scores: list[SampleScore] = []
    correct = 0
    # Which grading path each completion took, so the context can separate a
    # wrong answer from one the grader couldn't extract an answer out of.
    methods: Counter[str] = Counter()

    for c in completions:
        sample = by_id[c.id]
        text = remove_thinking_tags(c.completion)
        result = score_math_generic(text, sample.answer)
        scores.append(SampleScore(id=c.id, is_correct=result.correct))
        if result.correct:
            correct += 1
        methods[result.method] += 1

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
    # No usable answer extracted (no boxed/number match, or the grader errored) —
    # separates "wrong answer" from "couldn't parse an answer".
    unparsed = methods["no_match"] + methods["failed"]
    context: dict[str, Any] = {
        "math_500_pass_at_1": pass_at_1,
        "math_500_pass_at_1_stderr": (math.sqrt(pass_at_1 * (1.0 - pass_at_1) / n_samples) if n_samples else 0.0),
        "math_500_pass_at_1_per_sample": {
            lid: per_sample_correct[lid] / per_sample_total[lid] for lid in sorted(per_sample_total)
        },
        "math_500_unparsed_rate": unparsed / n_completions if n_completions else 0.0,
    }
    # Flat, fixed-key grading-method distribution — always emit every key so the
    # context schema stays stable regardless of which methods were hit.
    for method in _METHODS:
        context[f"math_500_method/{method}"] = methods[method]
    return scores, context
