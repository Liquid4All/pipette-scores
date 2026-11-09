"""Edit-distance alignment for ASR scoring, with the Open ASR Leaderboard's aligners.

Both are the leaderboard's ``normalizer/eval_utils._align_pair``
(huggingface/open_asr_leaderboard @ 62a457c5c32d0de4df77e9470308aae7d45b055f),
over the normalized texts:

- ``KALDIALIGN_COMPOUND``: its English scoring,
  ``kaldialign.edit_distance(ref.split(), pred.split(), merge_compounds=True)``
  (kaldialign 0.12.0, the version its ``requirements/requirements_jobs.txt``
  pins), so that a split compound ("white paper" against "whitepaper") is no
  error in either direction.
- ``KALDIALIGN_COMPOUND_PAIRS``: its multilingual scoring (``multilingual=True``),
  which first aligns compound word boundaries between the two texts
  (``eval_utils.normalize_compound_pairs``: a mismatched region that is the same
  characters without spaces takes the joined form on both sides), then aligns
  as ``KALDIALIGN_COMPOUND``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import assert_never


class Aligner(enum.StrEnum):
    # Open ASR Leaderboard: kaldialign over whitespace words, compounds merged (eval_utils._align_pair).
    KALDIALIGN_COMPOUND = "kaldialign_compound"
    # Its multilingual scoring: normalize_compound_pairs, then the same kaldialign alignment.
    KALDIALIGN_COMPOUND_PAIRS = "kaldialign_compound_pairs"


@dataclass(frozen=True)
class Counts:
    """One sample's alignment. `reference_units` is the reference's words."""

    hits: int
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def _kaldialign_compound(reference: str, hypothesis: str) -> Counts:
    from kaldialign import edit_distance

    result = edit_distance(reference.split(), hypothesis.split(), merge_compounds=True)
    substitutions = int(result["sub"])
    deletions = int(result["del"])
    reference_units = int(result["ref_len"])
    return Counts(
        hits=reference_units - substitutions - deletions,
        substitutions=substitutions,
        deletions=deletions,
        insertions=int(result["ins"]),
        reference_units=reference_units,
    )


def align(aligner: Aligner, reference: str, hypothesis: str) -> Counts:
    """Align normalized `hypothesis` against normalized `reference`."""
    match aligner:
        case Aligner.KALDIALIGN_COMPOUND:
            return _kaldialign_compound(reference, hypothesis)
        case Aligner.KALDIALIGN_COMPOUND_PAIRS:
            from pipette_scores.scoring.asr.open_asr_normalizer.multilingual import normalize_compound_pairs

            # _align_pair: (ref,), (pred,) = normalize_compound_pairs([ref], [pred]).
            (reference,), (hypothesis,) = normalize_compound_pairs([reference], [hypothesis])
            return _kaldialign_compound(reference, hypothesis)
        case _:
            assert_never(aligner)
