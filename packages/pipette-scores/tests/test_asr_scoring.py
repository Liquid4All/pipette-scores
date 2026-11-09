"""ASR scoring: the normalizer recipes, the aligners and the corpus rules."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pipette_scores.scoring.asr.align import Aligner, Counts, align
from pipette_scores.scoring.asr.normalize import NormalizerRecipe, normalize
from pipette_scores.scoring.asr.wer import AsrReference, AsrScoringSpec, score_asr

ASR = Path(__file__).parents[1] / "pipette_scores" / "scoring" / "asr"

OPEN_ASR = AsrScoringSpec(NormalizerRecipe.OPEN_ASR_ENGLISH, Aligner.KALDIALIGN_COMPOUND, open_asr_exclusions=True)
FLEURS_WER = AsrScoringSpec(NormalizerRecipe.OPEN_ASR_MULTILINGUAL, Aligner.KALDIALIGN_COMPOUND_PAIRS)


@pytest.mark.parametrize(
    ("path", "sha256"),
    [
        ("open_asr_normalizer/normalizer.py", "490b56393484ef386b486f3679cc2261264af280b3512154431e0bdcf2778295"),
        # Its multilingual normalizer and compound alignment, excerpted unchanged
        # (tests/test_asr_public_reference.py checks the excerpts against the source).
        (
            "open_asr_normalizer/multilingual.py",
            "9562b1d60fd0e691295879d4ea706497ff23e58d53b5f9ef1f33ce2b498678ce",
        ),
        (
            "open_asr_normalizer/english_abbreviations.py",
            "52997cc963e0bd6568d15554ad8d5fb0759f20f361483632a22f90d2d0e07db1",
        ),
    ],
)
def test_the_vendored_normalizers_are_the_recorded_bytes(path: str, sha256: str) -> None:
    """A changed byte changes the numbers; NOTICE.md says how to update."""
    assert hashlib.sha256((ASR / path).read_bytes()).hexdigest() == sha256


@pytest.mark.parametrize(
    ("recipe", "language", "raw", "normalized"),
    [
        # Open ASR English: contractions, titles, numbers, fillers, casing.
        (
            NormalizerRecipe.OPEN_ASR_ENGLISH,
            "eng",
            "Mr. Smith won't go, um, at 10:00!",
            "mister smith will not go at 10",
        ),
        (NormalizerRecipe.OPEN_ASR_ENGLISH, "eng", "It costs twenty five dollars.", "it costs $25"),
        # Open ASR multilingual: English text takes the English normalizer.
        (NormalizerRecipe.OPEN_ASR_MULTILINGUAL, "eng", "It costs twenty five dollars.", "it costs $25"),
        # Otherwise lowercase, punctuation and symbols to spaces (collapsed),
        # diacritics kept, digits spelled out in the language.
        (NormalizerRecipe.OPEN_ASR_MULTILINGUAL, "deu", "Grüße, Welt!", "grüße welt"),
        (NormalizerRecipe.OPEN_ASR_MULTILINGUAL, "fra", "L'été 2006 à Noël.", "l été deux mille six à noël"),
        # Parenthesized and bracketed spans are dropped.
        (NormalizerRecipe.OPEN_ASR_MULTILINGUAL, "nld", "Het is (lacht) [ruis] goed.", "het is goed"),
        # Space-separated thousands are joined first ("10 000" is one number).
        (NormalizerRecipe.OPEN_ASR_MULTILINGUAL, "nld", "10 000 mensen", "tienduizend mensen"),
        # The leaderboard's language codes: "pt" is num2words' European Portuguese.
        (NormalizerRecipe.OPEN_ASR_MULTILINGUAL, "por", "16 dias", "dezasseis dias"),
        # Number words come after the lowercase pass and keep num2words' casing.
        (NormalizerRecipe.OPEN_ASR_MULTILINGUAL, "deu", "1000000 Menschen", "eine Million menschen"),
    ],
)
def test_each_recipe_normalizes_as_its_public_protocol(
    recipe: NormalizerRecipe, language: str, raw: str, normalized: str
) -> None:
    assert normalize(recipe, raw, language) == normalized


# Each FLEURS language: a reference with digits and a hypothesis that spells the
# number out (and the reverse) score no error.
DIGITS_AND_WORDS = [
    ("eng", "He was 20 years old in 1999.", "he was twenty years old in nineteen ninety nine"),
    ("spa", "Tenía 20 años en 1999.", "tenía veinte años en mil novecientos noventa y nueve"),
    ("por", "Ela tinha 20 anos em 1999.", "ela tinha vinte anos em mil novecentos e noventa e nove"),
    (
        "deu",
        "Er war 20 Jahre alt, im Jahr 1999.",
        "er war zwanzig jahre alt im jahr eintausendneunhundertneunundneunzig",
    ),
    ("fra", "Il avait 20 ans en 1900.", "il avait vingt ans en mille neuf cents"),
    ("ita", "Aveva 20 anni nel 1999.", "aveva venti anni nel millenovecentonovantanove"),
    ("nld", "Hij was 20 jaar oud in 1999.", "hij was twintig jaar oud in duizendnegenhonderdnegenennegentig"),
]


@pytest.mark.parametrize(("language", "digits", "words"), DIGITS_AND_WORDS, ids=[c[0] for c in DIGITS_AND_WORDS])
def test_a_number_written_in_digits_or_in_words_is_no_error(language: str, digits: str, words: str) -> None:
    """The leaderboard's multilingual protocol spells digits out on both sides
    (num2words in the language; the English normalizer turns words into digits),
    so a model is not charged for writing "twenty" for "20", or the reverse."""
    for reference, hypothesis in ((digits, words), (words, digits)):
        score = score_asr(FLEURS_WER, [AsrReference("s", reference, language)], {"s": hypothesis})
        assert score.totals.errors == 0, (reference, hypothesis)
        assert score.totals.reference_units > 0


@pytest.mark.parametrize(
    ("language", "digits", "words"),
    [c for c in DIGITS_AND_WORDS if c[0] != "eng"],
    ids=[c[0] for c in DIGITS_AND_WORDS if c[0] != "eng"],
)
def test_without_its_number_words_the_protocol_would_charge_them(monkeypatch, language, digits, words) -> None:
    """Negative control: the same pairs with the number step switched off
    (MultilingualNormalizer._normalize_numbers as the identity) score errors, so
    the test above checks the number normalization, not the alignment."""
    from pipette_scores.scoring.asr.open_asr_normalizer.multilingual import MultilingualNormalizer

    monkeypatch.setattr(MultilingualNormalizer, "_normalize_numbers", lambda self, text, lang: text)
    score = score_asr(FLEURS_WER, [AsrReference("s", digits, language)], {"s": words})
    assert score.totals.errors > 0


def test_a_hyphenated_number_word_is_still_charged_as_on_the_leaderboard() -> None:
    """num2words' hyphens survive (it runs after punctuation becomes spaces), so
    "25" reads "vingt-cinq" while a hypothesis "vingt-cinq" splits into two
    words: one substitution and one insertion, as the leaderboard counts it."""
    score = score_asr(FLEURS_WER, [AsrReference("s", "25 euros", "fra")], {"s": "vingt-cinq euros"})
    assert (score.units[0].reference, score.units[0].hypothesis) == ("vingt-cinq euros", "vingt cinq euros")
    assert score.totals == Counts(1, 1, 0, 1, 2)


def test_the_multilingual_recipe_refuses_a_language_it_has_no_code_for() -> None:
    """Armenian (the leaderboard's own normalizer, not vendored) and any other
    language not evaluated here are refused, never scored without number words."""
    for language in ("hye", "cmn", None):
        with pytest.raises(ValueError, match="not set up for language"):
            normalize(NormalizerRecipe.OPEN_ASR_MULTILINGUAL, "1 2 3", language)
    with pytest.raises(ValueError, match="not set up for language"):
        score_asr(FLEURS_WER, [AsrReference("s", "ein Satz", "hye")], {"s": "ein Satz"})


@pytest.mark.parametrize(
    ("aligner", "reference", "hypothesis", "counts"),
    [
        # Compound-aware: "foot ball" against "football" is one match, not two errors.
        (Aligner.KALDIALIGN_COMPOUND, "the football game", "the foot ball game", Counts(3, 0, 0, 0, 3)),
        (Aligner.KALDIALIGN_COMPOUND, "a b", "", Counts(0, 0, 2, 0, 2)),
        (Aligner.KALDIALIGN_COMPOUND_PAIRS, "a b c d", "a x c", Counts(2, 1, 1, 0, 4)),
        (Aligner.KALDIALIGN_COMPOUND_PAIRS, "a b", "a b c", Counts(2, 0, 0, 1, 2)),
        (Aligner.KALDIALIGN_COMPOUND_PAIRS, "a b", "", Counts(0, 0, 2, 0, 2)),
        # A split compound, either way, as kaldialign's merge alone forgives it.
        (Aligner.KALDIALIGN_COMPOUND_PAIRS, "il dopo guerra", "il dopoguerra", Counts(2, 0, 0, 0, 2)),
        # normalize_compound_pairs: words cut at another point ("zumbei spiel")
        # are the same characters, so both sides take the joined form (one
        # reference word fewer); kaldialign's merge alone charges them.
        (
            Aligner.KALDIALIGN_COMPOUND_PAIRS,
            "das ist zum beispiel gut",
            "das ist zumbei spiel gut",
            Counts(4, 0, 0, 0, 4),
        ),
        (Aligner.KALDIALIGN_COMPOUND, "das ist zum beispiel gut", "das ist zumbei spiel gut", Counts(3, 2, 0, 0, 5)),
    ],
)
def test_each_aligner_counts_edits(aligner: Aligner, reference: str, hypothesis: str, counts: Counts) -> None:
    assert align(aligner, reference, hypothesis) == counts


def test_every_aligner_counts_words() -> None:
    """The character error rate is retired: no dataset counts characters, and a
    recipe naming the character aligner is not one this scorer reads."""
    assert {a.value for a in Aligner} == {"kaldialign_compound", "kaldialign_compound_pairs"}
    for retired in ("jiwer_characters", "jiwer_words"):
        with pytest.raises(ValueError):
            Aligner(retired)
    assert {AsrScoringSpec(NormalizerRecipe.OPEN_ASR_MULTILINGUAL, a).error_unit for a in Aligner} == {"word"}


def test_the_whisper_recipes_are_retired() -> None:
    """FLEURS is scored by the leaderboard's multilingual protocol; a dataset
    whose recipe still names Whisper's normalizers is not one this scorer reads."""
    assert {r.value for r in NormalizerRecipe} == {"open_asr_english", "open_asr_multilingual"}
    for retired in ("whisper_english", "whisper_multilingual"):
        with pytest.raises(ValueError):
            NormalizerRecipe(retired)


def test_the_corpus_rate_sums_edits_over_reference_units() -> None:
    """Not the mean of the per-sample rates: 1 error in 10 words and 1 in 1 is 2/11."""
    refs = [AsrReference("long", " ".join("abcdefghij"), "deu"), AsrReference("short", "z", "deu")]
    score = score_asr(FLEURS_WER, refs, {"long": " ".join("abcdefghiX"), "short": "y"})
    assert score.totals == Counts(9, 2, 0, 0, 11)
    assert score.corpus_error_rate == pytest.approx(2 / 11)


def test_open_asr_drops_references_that_normalize_to_nothing_or_the_ignore_marker() -> None:
    refs = [
        AsrReference("kept", "hello world", "eng"),
        AsrReference("empty", "<unk> [noise]", "eng"),
        AsrReference("marker", "IGNORE TIME SEGMENT IN SCORING", "eng"),
    ]
    score = score_asr(OPEN_ASR, refs, {"kept": "hello world", "empty": "something", "marker": "words"})
    assert [u.unit_id for u in score.units] == ["kept"]
    assert score.excluded_sample_ids == ("empty", "marker")


def test_fleurs_keeps_every_reference() -> None:
    score = score_asr(FLEURS_WER, [AsrReference("p", "...", "deu")], {"p": "ja"})
    assert score.excluded_sample_ids == ()
    assert score.totals.insertions == 1


def test_a_run_without_an_answer_to_every_sample_is_no_result() -> None:
    """Never an empty transcript (all deletions): a missing answer is refused
    (pipette's own rule); answered in full, the same run scores."""
    refs = [AsrReference(i, "one two", "deu") for i in ("a", "b", "c")]
    with pytest.raises(ValueError, match="an answer to every sample; unanswered: b, c"):
        score_asr(FLEURS_WER, refs, {"a": "one two"})
    complete = score_asr(FLEURS_WER, refs, {"a": "one two", "b": "", "c": "one"})
    assert complete.corpus_error_rate == pytest.approx(3 / 6)


def test_an_excluded_sample_must_still_have_its_generation() -> None:
    refs = [AsrReference("kept", "hello world", "eng"), AsrReference("noise", "<unk>", "eng")]
    with pytest.raises(ValueError, match="unanswered: noise"):
        score_asr(OPEN_ASR, refs, {"kept": "hello world"})
    score = score_asr(OPEN_ASR, refs, {"kept": "hello world", "noise": "words"})
    assert score.excluded_sample_ids == ("noise",) and score.corpus_error_rate == 0.0


def test_a_dataset_with_no_reference_unit_has_no_rate() -> None:
    """All references excluded, or only insertions: counts but no rate (the
    leaderboard would report 0.0, or infinity)."""
    excluded = score_asr(OPEN_ASR, [AsrReference("a", "<unk>", "eng")], {"a": "words"})
    assert excluded.units == () and excluded.corpus_error_rate is None
    empty = score_asr(FLEURS_WER, [AsrReference("p", "...", "deu")], {"p": "ja nee"})
    assert empty.totals == Counts(0, 0, 0, 2, 0) and empty.corpus_error_rate is None


def test_chunks_score_once_per_session_joined_in_order() -> None:
    """Stripped, empties skipped, joined with one space, then normalized whole."""
    session_ref = "Revenue grew five percent this quarter."
    refs = [
        AsrReference("c2", session_ref, "eng", session="call", chunk_index=2),
        AsrReference("c0", session_ref, "eng", session="call", chunk_index=0),
        AsrReference("c1", session_ref, "eng", session="call", chunk_index=1),
    ]
    score = score_asr(OPEN_ASR, refs, {"c0": " Revenue grew ", "c1": "   ", "c2": "5% this quarter. "})
    [unit] = score.units
    assert unit.unit_id == "call"
    assert unit.sample_ids == ("c0", "c1", "c2")
    assert unit.hypothesis == unit.reference
    assert unit.counts.errors == 0


@pytest.mark.parametrize(
    ("chunks", "size", "message"),
    [
        ([("c0", "same", 0), ("c2", "same", 2)], None, "chunk indexes"),
        ([("c0", "one", 0), ("c1", "other", 1)], None, "different references"),
        # The dataset holds two chunks of a call that has three.
        ([("c0", "same", 0), ("c1", "same", 1)], 3, "declares"),
    ],
)
def test_a_broken_session_is_refused(chunks: list[tuple[str, str, int]], size: int | None, message: str) -> None:
    refs = [AsrReference(i, ref, "eng", session="call", chunk_index=k, session_size=size) for i, ref, k in chunks]
    with pytest.raises(ValueError, match=message):
        score_asr(OPEN_ASR, refs, {i: "x" for i, _, _ in chunks})


def test_a_session_missing_a_chunk_is_refused() -> None:
    """pipette's own rule (the leaderboard warns and scores the call without it)."""
    refs = [AsrReference(f"c{k}", "a b c", "eng", session="call", chunk_index=k, session_size=3) for k in range(3)]
    with pytest.raises(ValueError, match="unanswered: c1"):
        score_asr(OPEN_ASR, refs, {"c0": "a", "c2": "c"})
