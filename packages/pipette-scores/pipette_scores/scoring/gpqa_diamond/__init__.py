import math
from collections import Counter, defaultdict
from typing import Any

from pipette_scores.scoring.gpqa_diamond.mcq import score_mcq

from pipette_scores.repeats import logical_id
from pipette_scores.scoring.completion_text import remove_thinking_tags
from pipette_scores.types import GPQADiamondSample, SampleCompletion, SampleScore

# GPQA Diamond is a 4-option MCQ (Artificial Analysis methodology).
_VALID_OPTIONS = "ABCD"


def score(
    completions: list[SampleCompletion], samples: list[GPQADiamondSample]
) -> tuple[list[SampleScore], dict[str, Any]]:
    """Score GPQA Diamond completions against the correct option letter.

    Each completion is run through the Artificial Analysis MCQ extractor
    (``score_mcq`` from this eval's ``mcq`` module, with ``valid_options="ABCD"``):
    the chosen letter is extracted from the response (after reasoning is
    stripped via ``remove_thinking_tags``) and compared to the ground-truth
    letter.

    AA methodology runs each question with repeats (R=5) and reports pass@1
    over the N·R rows. Repeats are realized as ``#k`` salted ids (see
    ``repeats.expand``); the per-sample breakdown groups scored ids back by
    logical id. The pass@1 denominator is the SAMPLE count from ``samples``
    (not received completions), so a missing attempt counts as a fail —
    matching the management "missing = incorrect" contract. ``repeats: 1``
    (no ``#k`` suffix) is transparent: each id is its own logical id.

    Context keys: ``gpqa_diamond_pass_at_1`` (+ ``_stderr`` and ``_per_sample``
    breakdown by logical id), ``gpqa_diamond_unparsed_rate`` (fraction of
    received completions whose answer couldn't be extracted), and flat
    ``gpqa_diamond_choice/{A,B,C,D,none}`` counts.
    """
    by_id = {s.id: s for s in samples}
    scores: list[SampleScore] = []
    correct = 0
    # Track the extracted letter per completion ("none" when the extractor
    # couldn't parse one) so the context can distinguish a wrong answer from an
    # extraction miss — a low score that's really a formatting/parse problem.
    choices: Counter[str] = Counter()

    for c in completions:
        sample = by_id[c.id]
        text = remove_thinking_tags(c.completion)
        result = score_mcq(text, sample.answer, valid_options=_VALID_OPTIONS)
        scores.append(SampleScore(id=c.id, is_correct=result.correct))
        if result.correct:
            correct += 1
        choices[result.extracted or "none"] += 1

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
    context: dict[str, Any] = {
        "gpqa_diamond_pass_at_1": pass_at_1,
        "gpqa_diamond_pass_at_1_stderr": (math.sqrt(pass_at_1 * (1.0 - pass_at_1) / n_samples) if n_samples else 0.0),
        "gpqa_diamond_pass_at_1_per_sample": {
            lid: per_sample_correct[lid] / per_sample_total[lid] for lid in sorted(per_sample_total)
        },
        # Of the completions we received, the fraction whose answer couldn't be
        # extracted (counts as wrong) — separates "wrong answer" from "unparsable".
        "gpqa_diamond_unparsed_rate": choices["none"] / n_completions if n_completions else 0.0,
    }
    # Flat, fixed-key choice distribution (counts), mirroring ifbench's
    # `/`-delimited breakdown keys. Always emit every key (A-D + none) so the
    # context schema stays stable regardless of which letters were chosen.
    for choice in [*_VALID_OPTIONS, "none"]:
        context[f"gpqa_diamond_choice/{choice}"] = choices[choice]
    return scores, context
