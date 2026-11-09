"""POPE scorer — VLMEvalKit Y/N extraction; headline metric is F1, not accuracy."""

from typing import Any

from pipette_scores.scoring.completion_text import ReasoningTally, split_reasoning
from pipette_scores.scoring.vlm_answer import yes_no
from pipette_scores.scoring.vlm_upstream.yorn import pope_rating
from pipette_scores.types import SampleCompletion, SampleScore, VlmSample


def _gt(sample: VlmSample) -> str:
    gt = sample.raw_doc.get("answer")
    if not isinstance(gt, str):
        gt = str(sample.answer or "")
    return gt.strip().capitalize()  # 'Yes' / 'No'


def score(
    completions: list[SampleCompletion],
    samples: list[VlmSample],
    *,
    enable_thinking: bool | None = None,
    opens_reasoning: bool = False,
) -> tuple[list[SampleScore], dict[str, Any]]:
    by_id = {s.id: s for s in samples}
    completions_by_id = {c.id: c for c in completions}
    scores: list[SampleScore] = []
    blanked = 0
    tally = ReasoningTally()
    fallbacks = 0
    extracted_by_id: dict[str, str] = {}
    for c in completions:
        sample = by_id[c.id]
        graded = tally.add(split_reasoning(c, enable_thinking, opens_reasoning=opens_reasoning))
        blanked += graded.blanked
        text = graded.answer
        gt = _gt(sample)
        extracted, fallback = yes_no(text)
        fallbacks += fallback
        extracted_by_id[c.id] = extracted
        scores.append(SampleScore(id=c.id, is_correct=extracted == gt and gt in ("Yes", "No")))

    # The confusion matrix is built over ALL dataset samples so a partial
    # submission cannot inflate the headline F1: a sample with no submitted
    # completion counts as the wrong answer to it, a false negative or a
    # false positive. 'Unknown' (a 'No' prediction, as upstream reads an
    # unparseable answer) would make every missing negative a free true
    # negative, which F1 does not see.
    records: list[dict] = []
    for sample in samples:
        gt = _gt(sample)
        records.append(
            {
                "answer": gt,
                "extracted": extracted_by_id.get(sample.id, "No" if gt == "Yes" else "Yes"),
                "category": str(sample.raw_doc.get("category", "") or ""),
            }
        )

    rating = pope_rating(records)
    overall = rating["Overall"]

    total = len(scores)
    correct = sum(1 for s in scores if s.is_correct)
    context = {
        # NOTE: for POPE, context["accuracy"] is the F1 SCORE (VLMEvalKit's
        # 'Overall' headline for POPE), on a 0-1 scale — not raw accuracy.
        "accuracy": round(overall["f1"], 6),
        "leaderboard_score": round(overall["f1"] * 100, 4),
        "f1": round(overall["f1"], 6),
        "precision": round(overall["precision"], 6),
        "recall": round(overall["recall"], 6),
        "yes_no_accuracy": round(overall["accuracy"], 6),
        "samples_missing": sum(1 for s in samples if s.id not in completions_by_id),
        "blanked_truncated": blanked,
        **tally.context(),
        "fallback_answers": fallbacks,
        "per_category": {
            k: {m: round(v, 6) for m, v in stats.items()} for k, stats in rating.items() if k != "Overall"
        },
        # Diagnostic: hit rate over the SUBMITTED completions only.
        "per_sample_accuracy": round(correct / total, 4) if total else 0.0,
    }
    return scores, context
