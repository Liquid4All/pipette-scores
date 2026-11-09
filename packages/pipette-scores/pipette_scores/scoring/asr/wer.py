"""Corpus word error rates, as the Open ASR Leaderboard computes them.

Per scored unit: normalize the reference and the hypothesis with the dataset's
recipe and align them. Per dataset: sum the edits over every unit and divide by
the summed reference units (corpus-level, as the leaderboard's
``eval_utils.align_manifests`` does, for its English and its multilingual
scoring alike), never the mean of the per-unit rates. Values are kept
unrounded; the leaderboard reports ``round(100 * wer, 2)`` per set.

Two rules of the Open ASR Leaderboard (huggingface/open_asr_leaderboard
@ 62a457c5c32d0de4df77e9470308aae7d45b055f) apply to its English sets:

- a sample whose normalized reference is empty, or reads "ignore time segment
  in scoring", is not scored (``data_utils.prepare_data`` filters it with
  ``is_target_text_in_range`` before inference; here it is served and its
  generation required, then left out of the score);
- chunked long-form audio (Earnings22) is scored per session: the chunks'
  stripped outputs, the empty ones skipped, joined with one space in
  ``chunk_index`` order, against the session's reference
  (``eval_utils.merge_chunked_manifest``).

pipette's own rule: a result is an answer to every sample, or no result. Every
sample (excluded ones included) must have a successful generation; a missing
one is never scored as an empty transcript, and ``score_asr`` refuses the run
instead (a failed generation is refused before it, in evaluate.py). (The
leaderboard scores what its manifest holds and only warns about a missing
chunk.)
"""

from __future__ import annotations

import functools
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pipette_scores.scoring.asr.align import Aligner, Counts, align
from pipette_scores.scoring.asr.normalize import NormalizerRecipe, normalize

# The normalized references `is_target_text_in_range` rejects.
_EXCLUDED_REFERENCES = frozenset({"", "ignore time segment in scoring"})


@dataclass(frozen=True)
class AsrScoringSpec:
    """How a dataset is scored; read from its metadata.json."""

    normalizer: NormalizerRecipe
    aligner: Aligner
    # Open ASR Leaderboard: drop samples whose normalized reference is empty or the ignore marker.
    open_asr_exclusions: bool = False

    @property
    def error_unit(self) -> str:
        # Every aligner counts words; `/score` still names the unit, so that a
        # consumer checks it rather than assumes it.
        return "word"


@dataclass(frozen=True)
class AsrReference:
    """One sample's ground truth. A chunk of a long-form session names it."""

    id: str
    reference: str
    # ISO 639-3.
    language: str | None
    session: str | None = None
    chunk_index: int | None = None
    # The session's chunk count (Earnings22 `parent_chunk_count`), when known.
    session_size: int | None = None


@dataclass(frozen=True)
class ScoredUnit:
    """A sample, or a whole session for chunked audio, as scored."""

    unit_id: str
    sample_ids: tuple[str, ...]
    reference: str
    hypothesis: str
    counts: Counts


@dataclass(frozen=True)
class AsrScore:
    units: tuple[ScoredUnit, ...]
    excluded_sample_ids: tuple[str, ...]
    error_unit: str

    @property
    def totals(self) -> Counts:
        """Summed over the scored units."""
        return Counts(
            hits=sum(u.counts.hits for u in self.units),
            substitutions=sum(u.counts.substitutions for u in self.units),
            deletions=sum(u.counts.deletions for u in self.units),
            insertions=sum(u.counts.insertions for u in self.units),
            reference_units=sum(u.counts.reference_units for u in self.units),
        )

    @property
    def corpus_error_rate(self) -> float | None:
        """Summed edits over summed reference units, as a ratio (0.05 = 5%).
        None when there is no reference unit to divide by (the leaderboard
        would report 0.0, or infinity if there are errors)."""
        totals = self.totals
        if totals.reference_units == 0:
            return None
        return totals.errors / totals.reference_units


def score_asr(
    spec: AsrScoringSpec,
    references: Sequence[AsrReference],
    hypotheses: Mapping[str, str],
) -> AsrScore:
    """Score `hypotheses` (successful generations, by sample id) against `references`.

    Every reference needs its hypothesis: a run without an answer to each
    sample is no result, so it raises ``ValueError`` rather than score the gap.
    """
    if unanswered := [r.id for r in references if r.id not in hypotheses]:
        raise ValueError(
            f"an ASR eval is scored only with an answer to every sample; unanswered: {', '.join(unanswered[:10])}"
        )

    # Cached per call: every Earnings22 chunk carries its whole call's transcript.
    @functools.cache
    def norm(text: str, language: str | None) -> str:
        return normalize(spec.normalizer, text, language)

    def norm_hypothesis(text: str, language: str | None) -> str:
        # pipette's own: a hypothesis the normalizer raises on scores as empty
        # rather than failing the whole dataset.
        try:
            return norm(text, language)
        except ValueError:
            return ""

    excluded: list[str] = []
    units: list[ScoredUnit] = []
    sessions: dict[str, list[AsrReference]] = defaultdict(list)
    for ref in references:
        if spec.open_asr_exclusions and norm(ref.reference, ref.language).strip() in _EXCLUDED_REFERENCES:
            excluded.append(ref.id)
            continue
        if ref.session is not None:
            sessions[ref.session].append(ref)
            continue
        reference = norm(ref.reference, ref.language)
        hypothesis = norm_hypothesis(hypotheses[ref.id], ref.language)
        units.append(ScoredUnit(ref.id, (ref.id,), reference, hypothesis, align(spec.aligner, reference, hypothesis)))

    for session in sorted(sessions):
        chunks = sorted(sessions[session], key=lambda r: _chunk_index(r))
        indexes = [_chunk_index(r) for r in chunks]
        if indexes != list(range(len(chunks))):
            raise ValueError(f"session {session!r} has chunk indexes {indexes}, not 0..{len(chunks) - 1}")
        sizes = {r.session_size for r in chunks if r.session_size is not None}
        if sizes and sizes != {len(chunks)}:
            raise ValueError(f"session {session!r} declares {sorted(sizes)} chunks, the dataset holds {len(chunks)}")
        if len({r.reference for r in chunks}) != 1:
            raise ValueError(f"session {session!r}: its chunks carry different references")
        sample_ids = tuple(r.id for r in chunks)
        texts = (hypotheses[i].strip() for i in sample_ids)
        joined = " ".join(t for t in texts if t)
        language = chunks[0].language
        reference = norm(chunks[0].reference, language)
        hypothesis = norm_hypothesis(joined, language)
        units.append(ScoredUnit(session, sample_ids, reference, hypothesis, align(spec.aligner, reference, hypothesis)))

    return AsrScore(tuple(units), tuple(excluded), spec.error_unit)


def _chunk_index(ref: AsrReference) -> int:
    if ref.chunk_index is None:
        raise ValueError(f"sample {ref.id!r} names session {ref.session!r} but no chunk_index")
    return ref.chunk_index
