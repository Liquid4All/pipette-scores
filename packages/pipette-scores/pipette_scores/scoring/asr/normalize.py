"""Text normalizers for ASR scoring, as the Open ASR Leaderboard applies them.

Each recipe maps raw text (and the sample's ISO 639-3 language) to the form the
aligner compares, on both the reference and the hypothesis. The vendored code is
unchanged (see NOTICE.md); this module only selects it, as the leaderboard's
``normalizer/eval_utils._normalize`` does (huggingface/open_asr_leaderboard
@ 62a457c5c32d0de4df77e9470308aae7d45b055f):

- ``OPEN_ASR_ENGLISH``: its ``EnglishTextNormalizer()`` (``normalizer/normalizer.py``),
  for its English sets.
- ``OPEN_ASR_MULTILINGUAL``: what ``_normalize(text, language)`` applies for the
  multilingual scoring (``score_results(multilingual=True, language=...)``), for
  FLEURS: the English normalizer when the language is ``"en"``, otherwise
  ``data_utils.ml_normalizer``, a ``MultilingualNormalizer(remove_diacritics=False)``
  called with ``lang=<code>`` (lowercase, bracketed and parenthesized spans
  dropped, symbols and punctuation to spaces with diacritics and combining marks
  kept, then every digit run spelled out by num2words in that language, so that
  "20" and "twenty" compare equal). The language is the leaderboard's two-letter
  code (its ``ML_LANG_DATASETS``), mapped from ISO 639-3 below.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from functools import cache
from typing import assert_never


class NormalizerRecipe(enum.StrEnum):
    # Open ASR Leaderboard sets: its English normalizer.
    OPEN_ASR_ENGLISH = "open_asr_english"
    # FLEURS: the leaderboard's multilingual scoring normalizer for the sample's language.
    OPEN_ASR_MULTILINGUAL = "open_asr_multilingual"


# ISO 639-3 to the code the leaderboard passes as `language` (its ML_LANG_DATASETS
# keys, and "en" for its English normalizer): the languages evaluated here. Any
# other language is refused rather than normalized without its number words.
OPEN_ASR_LANGUAGE_CODES: dict[str, str] = {
    "eng": "en",
    "spa": "es",
    "por": "pt",
    "deu": "de",
    "fra": "fr",
    "ita": "it",
    "nld": "nl",
}

type Normalizer = Callable[[str, str | None], str]


@cache
def _open_asr_english() -> Callable[[str], str]:
    from pipette_scores.scoring.asr.open_asr_normalizer.normalizer import EnglishTextNormalizer

    return EnglishTextNormalizer()


@cache
def _open_asr_multilingual() -> Callable[..., str]:
    from pipette_scores.scoring.asr.open_asr_normalizer.multilingual import MultilingualNormalizer

    # data_utils.ml_normalizer.
    return MultilingualNormalizer(remove_diacritics=False)


def open_asr_language(language: str | None) -> str:
    """The leaderboard's code for an ISO 639-3 `language`, refusing one not evaluated here."""
    code = OPEN_ASR_LANGUAGE_CODES.get((language or "").strip().lower())
    if code is None:
        raise ValueError(f"the Open ASR Leaderboard's multilingual scoring is not set up for language {language!r}")
    return code


def normalize(recipe: NormalizerRecipe, text: str, language: str | None) -> str:
    """`text` as `recipe` normalizes it for a sample in `language` (ISO 639-3)."""
    match recipe:
        case NormalizerRecipe.OPEN_ASR_ENGLISH:
            return _open_asr_english()(text)
        case NormalizerRecipe.OPEN_ASR_MULTILINGUAL:
            # eval_utils._normalize: the English normalizer for "en", else ml_normalizer(text, lang=language).
            code = open_asr_language(language)
            if code == "en":
                return _open_asr_english()(text)
            return _open_asr_multilingual()(text, lang=code)
        case _:
            assert_never(recipe)
