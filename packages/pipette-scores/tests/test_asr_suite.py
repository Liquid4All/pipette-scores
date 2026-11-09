"""The ASR suite headlines (scoring/asr/suite.py): the Open ASR Leaderboard's macro mean
and FLEURS' macro mean WER."""

from __future__ import annotations

from typing import Any

import pytest

from pipette_scores.dataset_catalog.asr_datasets import OPEN_ASR_EVAL_IDS, FLEURS_EVAL_IDS
from pipette_scores.scoring.asr.suite import fleurs_mean, open_asr_mean
from pipette_scores.types import EvalId


def _context(eval_id: EvalId, errors: int, reference_units: int) -> dict[str, Any]:
    """A /score context with the given corpus counts."""
    return {
        "error_unit": "word",
        "error_rate": errors / reference_units,
        "result": 100 * errors / reference_units,
        "substitutions": errors,
        "deletions": 0,
        "insertions": 0,
        "reference_units": reference_units,
    }


def _open_asr() -> list[tuple[str, dict[str, Any]]]:
    # Unequal sizes: dataset i has rate (i + 1)% over 10**(i % 4 + 2) words.
    return [
        (e.value, _context(e, (i + 1) * 10 ** (i % 4), 100 * 10 ** (i % 4))) for i, e in enumerate(OPEN_ASR_EVAL_IDS)
    ]


def test_the_open_asr_mean_is_the_equal_weight_mean_of_its_nine_public_sets() -> None:
    mean = open_asr_mean(_open_asr())
    # (1 + 2 + ... + 9) / 9 % = 5 %, whatever each dataset's size.
    assert mean.error_unit == "word" and mean.eval_ids == OPEN_ASR_EVAL_IDS and len(mean.eval_ids) == 9
    assert mean.error_rate == pytest.approx(0.05) and mean.result == pytest.approx(5.0)
    assert mean.leaderboard_result == 5.0
    # Not the micro-average (summed edits over summed words), which sizes weight.
    pairs = _open_asr()
    micro = sum(c["substitutions"] for _, c in pairs) / sum(c["reference_units"] for _, c in pairs)
    assert micro != pytest.approx(mean.error_rate)


def test_the_leaderboard_figure_rounds_each_set_then_the_mean() -> None:
    """The console figure of score_results: each set's round(100 * wer, 2),
    averaged, the mean shown to two decimals."""
    pairs = [(e.value, _context(e, 1, 3 if i == 0 else 100)) for i, e in enumerate(OPEN_ASR_EVAL_IDS)]
    mean = open_asr_mean(pairs)
    # 33.333...% and eight 1%: unrounded 4.592592..., rounded sets (33.33 + 8) / 9 = 4.5922...
    assert mean.result == pytest.approx((100 / 3 + 8) / 9)
    assert mean.leaderboard_result == round((33.33 + 8 * 1.0) / 9, 2) == 4.59


def test_the_leaderboard_figure_is_within_0_01_of_the_unrounded_mean_not_0_005() -> None:
    """Both roundings can push the same way: eight sets at 1.0049 % round down to
    1.00 and one at 1.0449 % to 1.04, so the console shows round(9.04 / 9, 2) =
    1.00 for an unrounded mean of 1.00934 %: off by more than 0.005, within 0.01."""
    rates = [0.010049] * 8 + [0.010449]
    pairs = [
        (e.value, {**_context(e, 1, 100), "error_rate": rate, "result": 100 * rate})
        for e, rate in zip(OPEN_ASR_EVAL_IDS, rates, strict=True)
    ]
    mean = open_asr_mean(pairs)
    assert mean.result == pytest.approx((8 * 1.0049 + 1.0449) / 9)
    assert mean.leaderboard_result == 1.0
    assert 0.005 < mean.result - mean.leaderboard_result <= 0.01
    # The CSV `avg` (the mean of the rounded sets, unrounded) is within 0.005.
    assert abs(mean.result - (8 * 1.0 + 1.04) / 9) <= 0.005


def _fleurs() -> list[tuple[str, dict[str, Any]]]:
    # Locale i at (i + 1) % over 100 * 10**(i % 3) words.
    return [(e.value, _context(e, (i + 1) * 10 ** (i % 3), 100 * 10 ** (i % 3))) for i, e in enumerate(FLEURS_EVAL_IDS)]


def test_fleurs_is_the_equal_weight_mean_wer_of_its_seven_locales() -> None:
    mean = fleurs_mean(_fleurs())
    assert [e.value.removeprefix("asr_fleurs_") for e in mean.eval_ids] == [
        "en_us", "es_419", "pt_br", "de_de", "fr_fr", "it_it", "nl_nl",
    ]  # fmt: skip
    # (1 + 2 + ... + 7) / 7 % = 4 %, whatever each locale's size.
    assert mean.error_unit == "word" and mean.result == pytest.approx(4.0) and mean.leaderboard_result == 4.0
    pairs = _fleurs()
    micro = sum(c["substitutions"] for _, c in pairs) / sum(c["reference_units"] for _, c in pairs)
    assert micro != pytest.approx(mean.error_rate)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda pairs: pairs[:-1], "missing"),
        (lambda pairs: [*pairs, pairs[0]], "more than once"),
        (lambda pairs: [*pairs, ("asr_fleurs_en_us", _context(EvalId.ASR_FLEURS_EN_US, 1, 10))], "not in the suite"),
        (
            lambda pairs: [(pairs[0][0], {**pairs[0][1], "error_rate": None}), *pairs[1:]],
            "no corpus rate",
        ),
        (lambda pairs: [(pairs[0][0], {**pairs[0][1], "error_unit": "character"}), *pairs[1:]], "scored in"),
    ],
)
def test_a_suite_headline_needs_every_dataset_once_with_a_rate(change, message) -> None:
    with pytest.raises(ValueError, match=message):
        open_asr_mean(change(_open_asr()))


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda pairs: pairs[:-1], "missing: asr_fleurs_nl_nl"),
        # A dropped locale is no eval id any more.
        (
            lambda pairs: [*pairs, ("asr_fleurs_ja_jp", _context(EvalId.ASR_FLEURS_EN_US, 1, 10))],
            "'asr_fleurs_ja_jp' is not a valid EvalId",
        ),
        (lambda pairs: [(pairs[0][0], {**pairs[0][1], "error_unit": "character"}), *pairs[1:]], "scored in"),
    ],
)
def test_fleurs_needs_its_seven_locales_in_words(change, message) -> None:
    with pytest.raises(ValueError, match=message):
        fleurs_mean(change(_fleurs()))
