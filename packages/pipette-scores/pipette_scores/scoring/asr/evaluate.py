"""The ASR evals' scorer: the standard `score(completions, samples)` contract over `wer.py`.

A completion is cleaned as its prompt profile says (``prompt_profile``; none
leaves it as sent; dataset_catalog/asr_prompt_profiles.py), then the
dataset's recipe (its metadata.json ``scoring``, checked against the catalog at
load) normalizes and aligns it. A completion whose ``stop_reason`` is
``"failure"`` is a failed generation, not an answer.

A result is an answer to every sample, or no result: a run that leaves a
sample unanswered, or carries a failed generation, is refused (``ValueError``;
``/score`` refuses it with a 400 first), never scored with the gap as an empty
transcript. The result (``context``) is the dataset's corpus rate; see
docs/asr-evals.md §3 for every field.

Per-sample ``measures`` are sum-safe: a unit's counts and normalized texts are on
one sample (an Earnings22 call's on its first chunk; the others carry only the
unit id), so summing them over the samples gives the totals.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from pipette_scores.scoring.asr.align import Aligner
from pipette_scores.scoring.asr.normalize import NormalizerRecipe
from pipette_scores.scoring.asr.wer import AsrReference, AsrScoringSpec, score_asr
from pipette_scores.types import AsrSample, SampleCompletion, SampleScore

FAILED_STOP_REASON = "failure"


def spec_of(samples: Sequence[AsrSample]) -> AsrScoringSpec:
    recipes = {tuple(sorted(s.scoring.items())) for s in samples}
    if len(recipes) != 1:
        raise ValueError(f"an ASR dataset has one scoring recipe, these samples carry {len(recipes)}")
    recipe = samples[0].scoring
    return AsrScoringSpec(
        normalizer=NormalizerRecipe(recipe["normalizer"]),
        aligner=Aligner(recipe["aligner"]),
        open_asr_exclusions=bool(recipe["open_asr_exclusions"]),
    )


def _cleaner(prompt_profile: str | None) -> Callable[[str], str]:
    if prompt_profile is None:
        return lambda text: text
    from pipette_scores.dataset_catalog.asr_prompt_profiles import profile

    return profile(prompt_profile).postprocess


def score(
    completions: Sequence[SampleCompletion],
    samples: Sequence[AsrSample],
    *,
    enable_thinking: bool | None = None,
    prompt_profile: str | None = None,
) -> tuple[list[SampleScore], dict[str, Any]]:
    """Per-sample scores for the given completions, and the dataset's result.

    ``enable_thinking=True`` is refused: every ASR profile runs with thinking
    off, so such a run is not comparable.
    """
    if enable_thinking:
        raise ValueError("ASR evals run with thinking off; enable_thinking=true is not comparable")
    if not samples:
        raise ValueError("no samples to score")
    known = {s.id for s in samples}
    seen = Counter(c.id for c in completions)
    if duplicates := sorted(i for i, n in seen.items() if n > 1):
        raise ValueError(f"duplicate completion ids: {', '.join(duplicates[:10])}")
    if unknown := sorted(set(seen) - known):
        raise ValueError(f"completions reference ids not in the dataset: {', '.join(unknown[:10])}")

    unanswered = sorted(known - set(seen))
    failed = sorted(c.id for c in completions if c.stop_reason == FAILED_STOP_REASON)
    if unanswered or failed:
        gaps = [
            f"{label}: {', '.join(ids[:10])}" for label, ids in (("unanswered", unanswered), ("failed", failed)) if ids
        ]
        raise ValueError(f"an ASR eval is scored only with an answer to every sample; {'; '.join(gaps)}")

    spec = spec_of(samples)
    clean = _cleaner(prompt_profile)
    hypotheses = {c.id: clean(c.completion) for c in completions}
    references = [
        AsrReference(
            id=s.id,
            reference=s.reference,
            language=s.language,
            session=s.session,
            chunk_index=s.chunk_index,
            session_size=s.raw_doc.get("parent_chunk_count") if s.session is not None else None,
        )
        for s in samples
    ]
    result = score_asr(spec, references, hypotheses)

    measures_of: dict[str, dict[str, Any]] = {}
    correct: set[str] = set()
    for unit in result.units:
        counts = unit.counts
        owner, *members = unit.sample_ids
        measures_of[owner] = {
            "unit": unit.unit_id,
            "hits": counts.hits,
            "substitutions": counts.substitutions,
            "deletions": counts.deletions,
            "insertions": counts.insertions,
            "reference_units": counts.reference_units,
            "reference": unit.reference,
            "hypothesis": unit.hypothesis,
        }
        for member in members:
            measures_of[member] = {"unit": unit.unit_id}
        if counts.errors == 0:
            correct.update(unit.sample_ids)
    for sample_id in result.excluded_sample_ids:
        measures_of[sample_id] = {"excluded": True}

    scores = [SampleScore(id=c.id, is_correct=c.id in correct, measures=measures_of[c.id]) for c in completions]
    totals = result.totals
    rate = result.corpus_error_rate
    context = {
        "error_unit": result.error_unit,
        # The dataset's corpus error rate (summed edits / summed reference units)
        # as a ratio, and in percent (`result`, the leaderboard's unit). Both
        # unrounded (the leaderboard shows round(result, 2)); null only when the
        # dataset has no reference unit.
        "error_rate": rate,
        "result": None if rate is None else rate * 100,
        "hits": totals.hits,
        "substitutions": totals.substitutions,
        "deletions": totals.deletions,
        "insertions": totals.insertions,
        "reference_units": totals.reference_units,
        "units_scored": len(result.units),
        "samples_total": len(samples),
        "samples_excluded": len(result.excluded_sample_ids),
        "stop_reasons": dict(sorted(Counter(c.stop_reason or "unreported" for c in completions).items())),
        "prompt_profile": prompt_profile,
        "scoring": {
            "normalizer": spec.normalizer.value,
            "aligner": spec.aligner.value,
            "open_asr_exclusions": spec.open_asr_exclusions,
        },
    }
    return scores, context
