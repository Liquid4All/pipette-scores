"""MME scorer — VLMEvalKit Y/N extraction + MME rating (perception/cognition)."""

from typing import Any

from pipette_scores.scoring.completion_text import ReasoningTally, split_reasoning
from pipette_scores.scoring.vlm_answer import yes_no
from pipette_scores.scoring.vlm_upstream.yorn import mme_rating
from pipette_scores.types import SampleCompletion, SampleScore, VlmSample

# Pairing key: VLMEvalKit's MME_rating groups the two yes/no questions of each
# image by `image_path`; the lmms-eval MME export carries the same image
# identity in `question_id` (the image's relative path, shared by both
# questions). Prefer `image_path`, fall back to `question_id`.
_PAIR_KEYS = ("image_path", "question_id")


def _image_key(raw_doc: dict) -> str:
    for k in _PAIR_KEYS:
        v = raw_doc.get(k)
        if v is not None and str(v) != "":
            return str(v)
    # Last resort: no pairing key -> every question its own "image", which
    # collapses acc_plus to acc; dataset generation must provide a pair key.
    return str(raw_doc.get("index", id(raw_doc)))


def score(
    completions: list[SampleCompletion],
    samples: list[VlmSample],
    *,
    enable_thinking: bool | None = None,
    opens_reasoning: bool = False,
) -> tuple[list[SampleScore], dict[str, Any]]:
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
        extracted, fallback = yes_no(text)
        fallbacks += fallback
        gt = str(sample.raw_doc.get("answer", "")).strip().capitalize()  # 'Yes' / 'No'
        hit = extracted == gt and gt in ("Yes", "No")
        scores.append(SampleScore(id=c.id, is_correct=hit))
        hits[c.id] = hit

    # Rating records cover ALL dataset samples so a partial submission cannot
    # inflate the headline: a sample with no submitted completion scores 0.0,
    # contributing 0 to both acc (it stays in the denominator) and acc_plus
    # (its image pair can no longer be fully correct) — exactly like a wrong
    # answer.
    records = [
        {
            "category": str(s.raw_doc.get("category", "")),
            "image_key": _image_key(s.raw_doc),
            "score": 1.0 if hits.get(s.id, False) else 0.0,
        }
        for s in samples
    ]

    rating = mme_rating(records)
    perception = rating.pop("perception")
    cognition = rating.pop("cognition")
    mme_total = perception + cognition

    total = len(scores)
    correct = sum(1 for s in scores if s.is_correct)
    context = {
        # NOTE: for MME, context["accuracy"] is the MME total score / 2800
        # (the conventional 0-2800 scale normalized to 0-1), not raw hit rate.
        "accuracy": round(mme_total / 2800, 6),
        "leaderboard_score": round(mme_total, 2),
        "mme_perception_score": round(perception, 2),
        "mme_cognition_score": round(cognition, 2),
        "mme_total_score": round(mme_total, 2),
        "samples_missing": sum(1 for s in samples if s.id not in hits),
        "blanked_truncated": blanked,
        **tally.context(),
        "fallback_answers": fallbacks,
        "per_category_score": {k: round(v, 2) for k, v in sorted(rating.items())},
        # Diagnostic: hit rate over the SUBMITTED completions only.
        "per_sample_accuracy": round(correct / total, 4) if total else 0.0,
    }
    return scores, context
