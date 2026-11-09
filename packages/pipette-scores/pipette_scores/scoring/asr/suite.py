"""The ASR suite headlines over per-dataset `/score` results: the reference definition.

The scores server scores one dataset per request and keeps no run state, so a
suite headline is its consumer's (pipette-mgmt), as no VLM aggregate is computed
here either. These functions are the definition that consumer implements
(docs/asr-evals.md §3, "Suite headlines"), tested in tests/test_asr_suite.py:

- **Open ASR Leaderboard, English**: the equal-weight mean of its 9 public sets'
  corpus WERs (a macro mean: never the edits summed across datasets, never
  weighted by size), as the leaderboard's ``score_results`` averages its
  "public" columns (huggingface/open_asr_leaderboard @ 62a457c5
  ``normalizer/eval_utils.py``). ``result`` is that mean of the unrounded rates.
  The leaderboard averages the per-set WERs it has already rounded to two
  decimals (``round(100 * wer, 2)``): its public CSV ``avg`` is that mean,
  unrounded (``score_results``: "Unrounded, to match english_short_latest.csv"),
  and its console "Composite Results" line shows the same mean to two decimals
  (``:0.2f``). ``leaderboard_result`` is the console figure: the mean of the
  rounded per-set WERs, rounded to two decimals. It differs from ``result`` by
  at most 0.01 points (up to 0.005 from the per-set rounding, up to 0.005 from
  the final one).
- **FLEURS**: the equal-weight mean of its 7 locales' corpus WERs (en_us,
  es_419, pt_br, de_de, fr_fr, it_it, nl_nl), the same macro mean.

Each input is one dataset's ``context`` from `/score`. A suite headline needs
every one of its datasets exactly once, each with a corpus rate (a non-null
``error_rate``; `/score` gives none to a partial run) and scored in words;
anything else raises rather than averaging what is there.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pipette_scores.dataset_catalog.asr_datasets import FLEURS_EVAL_IDS, OPEN_ASR_EVAL_IDS
from pipette_scores.types import EvalId


@dataclass(frozen=True)
class SuiteMean:
    """An equal-weight mean of per-dataset corpus rates."""

    error_unit: str  # "word"
    eval_ids: tuple[EvalId, ...]
    # The mean of the datasets' `error_rate` ratios, and that mean x 100.
    error_rate: float
    result: float
    # The datasets' percent rates each rounded to 2 decimals, averaged, the mean
    # rounded to 2 decimals: the figure the leaderboard's console prints (its
    # CSV `avg` is the same mean unrounded). At most 0.01 from `result`.
    leaderboard_result: float


def _rates(
    results: Iterable[tuple[str, Mapping[str, Any]]], expected: Sequence[EvalId], suite: str
) -> dict[EvalId, float]:
    """Each expected dataset's corpus rate, refusing anything else."""
    pairs = [(EvalId(eval_id), context) for eval_id, context in results]
    counts = Counter(eval_id for eval_id, _ in pairs)
    if duplicates := sorted(e.value for e, n in counts.items() if n > 1):
        raise ValueError(f"{suite}: datasets given more than once: {', '.join(duplicates)}")
    if missing := [e.value for e in expected if e not in counts]:
        raise ValueError(f"{suite}: datasets missing: {', '.join(missing)}")
    if extra := sorted(e.value for e in counts if e not in expected):
        raise ValueError(f"{suite}: datasets not in the suite: {', '.join(extra)}")
    rates: dict[EvalId, float] = {}
    for eval_id, context in pairs:
        if context.get("error_rate") is None:
            raise ValueError(f"{suite}: {eval_id.value} has no corpus rate")
        if context.get("error_unit") != "word":
            raise ValueError(f"{suite}: {eval_id.value} is scored in {context.get('error_unit')!r}, not 'word'")
        rates[eval_id] = float(context["error_rate"])
    return rates


def _equal_weight(rates: Mapping[EvalId, float], eval_ids: Sequence[EvalId]) -> SuiteMean:
    mean = sum(rates[e] for e in eval_ids) / len(eval_ids)
    rounded = [round(100 * rates[e], 2) for e in eval_ids]
    return SuiteMean(
        error_unit="word",
        eval_ids=tuple(eval_ids),
        error_rate=mean,
        result=mean * 100,
        leaderboard_result=round(sum(rounded) / len(rounded), 2),
    )


def open_asr_mean(results: Iterable[tuple[str, Mapping[str, Any]]]) -> SuiteMean:
    """Open ASR Leaderboard, English: the equal-weight mean WER of its 9 public
    sets. ``results`` are (eval_id, `/score` context) pairs, exactly one per set."""
    rates = _rates(results, OPEN_ASR_EVAL_IDS, "Open ASR Leaderboard")
    return _equal_weight(rates, OPEN_ASR_EVAL_IDS)


def fleurs_mean(results: Iterable[tuple[str, Mapping[str, Any]]]) -> SuiteMean:
    """FLEURS: the equal-weight mean WER of its 7 locales. ``results`` are
    (eval_id, `/score` context) pairs, exactly one per locale."""
    rates = _rates(results, FLEURS_EVAL_IDS, "FLEURS")
    return _equal_weight(rates, FLEURS_EVAL_IDS)
