"""MMMU (val) scorer — VLMEvalKit MCQ path with MMMU_preproc.

Open questions are re-formulated as 2-choice MCQ (A = gold, B = 'Other
Answers') per VLMEvalKit's `MMMU_preproc`; the official-MMMU
`parse_open_response` path is deliberately NOT used. Unparseable = wrong
(no judge, no random fallback). Headline = micro accuracy.
"""

from typing import Any

from pipette_scores.scoring.completion_text import ReasoningTally, split_reasoning
from pipette_scores.scoring.vlm_answer import option_hit
from pipette_scores.scoring.vlm_upstream.multiple_choice import (
    mmmu_preproc_choices,
    mmmu_preproc_gt,
)
from pipette_scores.types import SampleCompletion, SampleScore, VlmSample


def score(
    completions: list[SampleCompletion],
    samples: list[VlmSample],
    *,
    enable_thinking: bool | None = None,
    opens_reasoning: bool = False,
) -> tuple[list[SampleScore], dict[str, Any]]:
    # Headline accuracy is over the FULL dataset: a sample with no submitted
    # completion counts as wrong (no partial-submission inflation).
    by_id = {s.id: s for s in samples}
    scores: list[SampleScore] = []
    blanked = 0
    tally = ReasoningTally()
    fallbacks = 0
    hits: dict[str, bool] = {}
    n_open = 0
    for c in completions:
        sample = by_id[c.id]
        graded = tally.add(split_reasoning(c, enable_thinking, opens_reasoning=opens_reasoning))
        blanked += graded.blanked
        text = graded.answer
        choices = mmmu_preproc_choices(sample.raw_doc)
        if choices.get("B") == "Other Answers" and "C" not in choices:
            n_open += 1
        gt = mmmu_preproc_gt(sample.raw_doc.get("answer", ""))
        hit, fallback = option_hit(text, choices, gt)
        fallbacks += fallback
        scores.append(SampleScore(id=c.id, is_correct=hit))
        hits[c.id] = hit

    cat_buckets: dict[str, list[bool]] = {}
    for s in samples:
        cat = str(s.raw_doc.get("category", "") or "unknown")
        cat_buckets.setdefault(cat, []).append(hits.get(s.id, False))

    total = len(samples)
    correct = sum(1 for h in hits.values() if h)
    accuracy = correct / total if total else 0.0
    context = {
        "accuracy": round(accuracy, 6),
        "leaderboard_score": round(accuracy * 100, 4),
        "samples_missing": total - len(hits),
        "blanked_truncated": blanked,
        **tally.context(),
        "fallback_answers": fallbacks,
        "num_open_questions_reformulated": n_open,
        "per_category_accuracy": {cat: round(sum(v) / len(v), 6) for cat, v in sorted(cat_buckets.items()) if v},
    }
    return scores, context
