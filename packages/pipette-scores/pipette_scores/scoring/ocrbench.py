"""OCRBench scorer — VLMEvalKit category-keyed matching; headline = Final Score / 1000."""

from typing import Any

from pipette_scores.scoring.completion_text import ReasoningTally, split_reasoning
from pipette_scores.scoring.vlm_upstream.ocrbench import (
    ocrbench_final_score,
    ocrbench_match,
    parse_answers,
)
from pipette_scores.types import SampleCompletion, SampleScore, VlmSample

# OCRBench has exactly 1000 questions; the conventional 'Final Score' is the
# number correct (denominator fixed at 1000 regardless of subset size).
_OCRBENCH_TOTAL = 1000


def _category(raw_doc: dict) -> str:
    # VLMEvalKit column is `category`; the lmms-eval export calls the same
    # values `question_type`.
    return str(raw_doc.get("category") or raw_doc.get("question_type") or "")


def score(
    completions: list[SampleCompletion],
    samples: list[VlmSample],
    *,
    enable_thinking: bool | None = None,
    opens_reasoning: bool = False,
) -> tuple[list[SampleScore], dict[str, Any]]:
    by_id = {s.id: s for s in samples}
    submitted_ids = {c.id for c in completions}
    scores: list[SampleScore] = []
    blanked = 0
    tally = ReasoningTally()
    per_category_correct: dict[str, int] = {}
    for c in completions:
        sample = by_id[c.id]
        graded = tally.add(split_reasoning(c, enable_thinking, opens_reasoning=opens_reasoning))
        blanked += graded.blanked
        text = graded.answer
        category = _category(sample.raw_doc)
        answers = parse_answers(sample.raw_doc.get("answer", ""))
        hit = ocrbench_match(text, answers, category)
        scores.append(SampleScore(id=c.id, is_correct=hit))
        if hit:
            per_category_correct[category] = per_category_correct.get(category, 0) + 1

    final = ocrbench_final_score(per_category_correct)

    total = len(scores)
    correct = sum(1 for s in scores if s.is_correct)
    context = {
        # NOTE: for OCRBench, context["accuracy"] is Final Score / 1000 (the
        # conventional fixed-denominator scale); on a subset this is NOT the
        # per-sample hit rate (see per_sample_accuracy).
        "accuracy": round(final["Final Score"] / _OCRBENCH_TOTAL, 6),
        "leaderboard_score": round(final["Final Score"], 2),
        "final_score": round(final["Final Score"], 2),
        "final_score_norm": round(final["Final Score Norm"], 2),  # 0-100 scale
        # A dataset sample with no submitted completion simply contributes no
        # correct count; the /1000 denominator is fixed, so a partial
        # submission cannot inflate the headline.
        "samples_missing": sum(1 for s in samples if s.id not in submitted_ids),
        "blanked_truncated": blanked,
        **tally.context(),
        "per_category_score": {
            k: round(v, 2) for k, v in final.items() if k not in ("Final Score", "Final Score Norm")
        },
        # Diagnostic: hit rate over the SUBMITTED completions only.
        "per_sample_accuracy": round(correct / total, 4) if total else 0.0,
    }
    return scores, context
