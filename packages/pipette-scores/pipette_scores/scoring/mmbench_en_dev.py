"""MMBench EN dev scorer — VLMEvalKit can_infer + circular (rotation) grouping.

VLMEvalKit evaluates MMBench with `mcq_circular_eval`: the dataset contains
rotated copies of each base question (rotation rows share `index % 1e6`), and
a base question only counts as a hit when EVERY rotation is answered
correctly. Extraction is `can_infer`; unparseable = wrong (no judge). On a
dataset without rotation rows this degenerates to vanilla per-question
accuracy.

Groups are enumerated from the dataset, so a partial submission cannot
inflate the headline: an unsubmitted rotation counts as wrong and makes its
base question a miss.
"""

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

    # Circular groups are built from the DATASET rows (all base questions),
    # not just the submitted ones: a rotation with no submitted completion
    # counts as wrong, so a group with any missing rotation is a miss (no
    # partial-submission inflation).
    per_base: dict[int, list[bool]] = {}
    per_base_category: dict[int, str] = {}
    for s in samples:
        base = int(s.raw_doc.get("index", 0)) % 1_000_000
        per_base.setdefault(base, []).append(hits.get(s.id, False))
        per_base_category.setdefault(base, str(s.raw_doc.get("category", "") or "unknown"))

    total = len(scores)
    correct = sum(1 for s in scores if s.is_correct)
    num_questions = len(per_base)
    base_hits = {base: all(rots) for base, rots in per_base.items()}
    num_hits = sum(1 for h in base_hits.values() if h)
    hit_rate = num_hits / num_questions if num_questions else 0.0

    cat_buckets: dict[str, list[bool]] = {}
    for base, h in base_hits.items():
        cat_buckets.setdefault(per_base_category[base], []).append(h)

    context = {
        # Circular accuracy over base questions (VLMEvalKit MMBench headline).
        "accuracy": round(hit_rate, 6),
        "leaderboard_score": round(hit_rate * 100, 4),
        "num_questions": num_questions,
        "num_hits": num_hits,
        "samples_missing": len(samples) - len(hits),
        "blanked_truncated": blanked,
        **tally.context(),
        "fallback_answers": fallbacks,
        # Diagnostic: hit rate over the SUBMITTED rotations only.
        "per_rotation_accuracy": round(correct / total, 4) if total else 0.0,
        "per_category_accuracy": {cat: round(sum(v) / len(v), 6) for cat, v in sorted(cat_buckets.items()) if v},
    }
    return scores, context
