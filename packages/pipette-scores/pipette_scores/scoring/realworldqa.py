"""RealWorldQA scorer — VLMEvalKit ImageMCQ path (can_infer over A-D choices)."""

from typing import Any

from pipette_scores.scoring.completion_text import ReasoningTally, split_reasoning
from pipette_scores.scoring.vlm_answer import option_hit
from pipette_scores.scoring.vlm_upstream.multiple_choice import build_choices
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
    for c in completions:
        sample = by_id[c.id]
        graded = tally.add(split_reasoning(c, enable_thinking, opens_reasoning=opens_reasoning))
        blanked += graded.blanked
        text = graded.answer
        choices = build_choices(sample.raw_doc)
        hit, fallback = option_hit(text, choices, str(sample.raw_doc.get("answer", "")))
        fallbacks += fallback
        scores.append(SampleScore(id=c.id, is_correct=hit))
        hits[c.id] = hit

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
    }
    return scores, context
