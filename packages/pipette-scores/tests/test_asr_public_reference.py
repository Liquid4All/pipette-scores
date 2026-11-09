"""ASR scoring and audio against the PUBLIC reference implementations.

- Open ASR Leaderboard (English sets): the leaderboard's own code
  (huggingface/open_asr_leaderboard @ 62a457c5: ``data_utils`` normalizer and
  ``prepare_data`` filter, ``eval_utils`` chunk merge and kaldialign alignment)
  must give exactly our counts and corpus rate, on every set.
- FLEURS: the same code as the leaderboard's multilingual scoring runs it
  (``eval_utils._normalize`` with the language, ``data_utils.ml_normalizer``
  and its num2words step, ``normalize_compound_pairs``, kaldialign) must give
  exactly our normalized texts, counts and corpus rate, and our vendored
  excerpts must be its source text.
- Audio: ``datasets.Audio(sampling_rate=16000)`` (datasets 2.19.0, librosa,
  soundfile) must give exactly our 16 kHz mono samples.

Two layers. The fixture tests always run: they compare us with the reference
answers recorded on the synthetic corpus (fixtures/asr_public_reference.json;
for the audio, each clip's shape and fixed projections of its samples). The
live tests need the reference environment
(``scripts/setup_asr_reference_env.sh DIR``, then ``PIPETTE_ASR_REFERENCE_ENV=DIR``)
and run the reference code itself on the synthetic corpus plus, with
``PIPETTE_SCORES_DATA_DIR`` set, every reference of each built dataset there; a
root that holds none fails the run. Every answer is paired with its input by
name, and their numbers must agree. Negative controls show the checks can
fail: what a near-miss method (no compound merge on the English sets; on FLEURS
no number words, or no compound-pair step; soxr's very-high quality, no length
rule) gives differs from the reference.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
from typing import Any

import numpy as np
import pytest
from asr_public_reference import (
    FIXTURE,
    audio_summary,
    by_name,
    corpus_digest,
    data_roots,
    digest,
    fleurs_transcripts,
    real_corpus,
    reference_array,
    reference_env,
    run_reference,
    synthetic_calls,
    synthetic_clips,
    synthetic_english,
    synthetic_fixture,
    synthetic_fleurs,
    synthetic_inputs,
)

from pipette_scores.dataset_catalog import _asr_audio, asr_datasets
from pipette_scores.scoring.asr.align import Aligner
from pipette_scores.scoring.asr.normalize import OPEN_ASR_LANGUAGE_CODES, NormalizerRecipe, normalize
from pipette_scores.scoring.asr.wer import AsrReference, AsrScore, AsrScoringSpec, score_asr
from pipette_scores.types import EvalId

OPEN_ASR = AsrScoringSpec(NormalizerRecipe.OPEN_ASR_ENGLISH, Aligner.KALDIALIGN_COMPOUND, open_asr_exclusions=True)


def _score_open_asr(manifest: list[dict[str, Any]], spec: AsrScoringSpec = OPEN_ASR) -> AsrScore:
    references = [
        AsrReference(
            id=str(i),
            reference=row["text"],
            language="eng",
            session=row.get("parent_id"),
            chunk_index=row.get("chunk_index"),
        )
        for i, row in enumerate(manifest)
    ]
    return score_asr(spec, references, {str(i): row["pred_text"] for i, row in enumerate(manifest)})


def _fleurs_spec(eval_id: str) -> AsrScoringSpec:
    dataset = asr_datasets.ASR_DATASETS[EvalId(eval_id)]
    return AsrScoringSpec(dataset.normalizer, dataset.aligner, dataset.open_asr_exclusions)


def _score_fleurs(dataset: dict[str, Any]) -> AsrScore:
    references = [AsrReference(str(i), r, dataset["language"]) for i, (r, _) in enumerate(dataset["pairs"])]
    hypotheses = {str(i): h for i, (_, h) in enumerate(dataset["pairs"])}
    return score_asr(_fleurs_spec(dataset["eval_id"]), references, hypotheses)


def _assert_open_asr_equal(ours: AsrScore, theirs: dict[str, Any]) -> None:
    totals = ours.totals
    assert len(ours.units) == theirs["units"]
    assert {
        "ins": totals.insertions,
        "del": totals.deletions,
        "sub": totals.substitutions,
        "total": totals.errors,
        "ref_len": totals.reference_units,
    } == {k: theirs["totals"][k] for k in ("ins", "del", "sub", "total", "ref_len")}
    rate = ours.corpus_error_rate
    assert rate == theirs["totals"]["err_rate"]
    assert round(100 * rate, 2) == theirs["wer"]
    # Unit by unit, in the leaderboard's order (rows in order; calls sorted by id).
    assert [
        [u.counts.insertions, u.counts.deletions, u.counts.substitutions, u.counts.reference_units] for u in ours.units
    ] == theirs["pairs"]


def _errors_without_merge(manifest: list[dict[str, Any]]) -> int:
    """The leaderboard's alignment of the rows (no chunked calls) without
    merge_compounds: a near miss."""
    from kaldialign import edit_distance

    errors = 0
    for row in manifest:
        reference = normalize(NormalizerRecipe.OPEN_ASR_ENGLISH, row["text"], "eng")
        if reference.strip() in ("", "ignore time segment in scoring"):
            continue
        hypothesis = normalize(NormalizerRecipe.OPEN_ASR_ENGLISH, row["pred_text"], "eng")
        errors += edit_distance(reference.split(), hypothesis.split(), merge_compounds=False)["total"]
    return errors


def _fleurs_totals(ours: AsrScore) -> dict[str, Any]:
    totals = ours.totals
    return {
        "ins": totals.insertions,
        "del": totals.deletions,
        "sub": totals.substitutions,
        "total": totals.errors,
        "ref_len": totals.reference_units,
        "err_rate": ours.corpus_error_rate,
    }


def _assert_fleurs_equal(dataset: dict[str, Any], theirs: dict[str, Any]) -> None:
    recipe = asr_datasets.ASR_DATASETS[EvalId(dataset["eval_id"])].normalizer
    language = dataset["language"]
    assert [normalize(recipe, r, language) for r, _ in dataset["pairs"]] == theirs["references"]
    assert [normalize(recipe, h, language) for _, h in dataset["pairs"]] == theirs["hypotheses"]
    ours = _score_fleurs(dataset)
    assert _fleurs_totals(ours) == theirs["totals"]
    # Pair by pair, in order.
    assert [
        [u.counts.insertions, u.counts.deletions, u.counts.substitutions, u.counts.reference_units] for u in ours.units
    ] == theirs["pairs"]


def _fleurs_errors(dataset: dict[str, Any], spec: AsrScoringSpec) -> int:
    references = [AsrReference(str(i), r, dataset["language"]) for i, (r, _) in enumerate(dataset["pairs"])]
    return score_asr(spec, references, {str(i): h for i, (_, h) in enumerate(dataset["pairs"])}).totals.errors


# ---------------------------------------------------------------------------
# Always: against the reference answers recorded on the synthetic corpus
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert recorded["corpus_sha256"] == corpus_digest(), "the synthetic corpus changed: regenerate the fixture"
    return recorded


def _recorded_totals(ours: AsrScore, recorded: dict[str, Any]) -> None:
    totals = ours.totals
    assert (len(ours.units), totals.insertions, totals.deletions, totals.substitutions, totals.reference_units) == (
        recorded["units"],
        recorded["totals"]["ins"],
        recorded["totals"]["del"],
        recorded["totals"]["sub"],
        recorded["totals"]["ref_len"],
    )
    assert ours.corpus_error_rate == recorded["totals"]["err_rate"]
    assert round(100 * ours.corpus_error_rate, 2) == recorded["wer"]


def test_the_english_corpus_scores_as_the_leaderboard_scored_it(fixture):
    ours = _score_open_asr([{"text": r, "pred_text": h} for r, h in synthetic_english()])
    _recorded_totals(ours, fixture["open_asr_english"])
    assert len(ours.units) == fixture["open_asr_english"]["kept"]


def test_the_chunked_calls_score_as_the_leaderboard_merged_them(fixture):
    _recorded_totals(_score_open_asr(synthetic_calls()), fixture["open_asr_calls"])


@pytest.mark.parametrize("dataset", synthetic_fleurs(), ids=lambda d: d["eval_id"])
def test_each_fleurs_locale_normalizes_and_scores_as_the_leaderboards_multilingual_code_did(dataset, fixture):
    recorded = fixture["fleurs"][dataset["eval_id"]]
    recipe = asr_datasets.ASR_DATASETS[EvalId(dataset["eval_id"])].normalizer
    language = dataset["language"]
    assert digest([normalize(recipe, r, language) for r, _ in dataset["pairs"]]) == recorded["references_sha256"]
    assert digest([normalize(recipe, h, language) for _, h in dataset["pairs"]]) == recorded["hypotheses_sha256"]
    assert _fleurs_totals(_score_fleurs(dataset)) == recorded["totals"]


def test_the_negative_controls_miss_the_recorded_answers(fixture, monkeypatch):
    """Near-miss methods do not reproduce the leaderboard: on its English sets
    without merge_compounds; on FLEURS without the number words (every non-English
    locale), or without the compound-pair step (any locale's corpus, whose
    hypotheses cut neighbouring words at other points)."""
    manifest = [{"text": r, "pred_text": h} for r, h in synthetic_english()]
    assert _errors_without_merge(manifest) != fixture["open_asr_english"]["totals"]["total"]

    fleurs = {d["eval_id"]: d for d in synthetic_fleurs()}
    no_pairs = AsrScoringSpec(NormalizerRecipe.OPEN_ASR_MULTILINGUAL, Aligner.KALDIALIGN_COMPOUND)
    for eval_id, dataset in fleurs.items():
        assert _fleurs_errors(dataset, no_pairs) != fixture["fleurs"][eval_id]["totals"]["total"], eval_id

    from pipette_scores.scoring.asr.open_asr_normalizer.multilingual import MultilingualNormalizer

    monkeypatch.setattr(MultilingualNormalizer, "_normalize_numbers", lambda self, text, lang: text)
    for eval_id, dataset in fleurs.items():
        if eval_id == "asr_fleurs_en_us":
            continue  # the English normalizer has its own number step
        assert _fleurs_totals(_score_fleurs(dataset)) != fixture["fleurs"][eval_id]["totals"], eval_id


# soxr's "high" quality (librosa's soxr_hq) runs libsoxr's single-precision
# engine, whose SIMD (SSE on x86_64, NEON on arm64) and scalar code round
# differently on each CPU architecture: the same clip, soxr 1.1.0 and numpy give
# float32 samples that differ by up to ~1.2e-7 (4 float32 ulps at these clips'
# peaks) between macOS arm64, Linux x86_64 and Linux aarch64, and the fixture's
# projections by up to ~6e-6. Both are correct; exact equality, on the platform
# the reference runs on, is the live layer's. 1e-4 leaves 16x that, and is
# >200x below the nearest near miss: soxr "very high" moves a projection by
# 2.2e-2 at least, and a wrong length or dtype fails the shape check.
_PROJECTION_TOLERANCE = 1e-4


def _matches_recorded(array: np.ndarray, recorded: dict[str, Any]) -> bool:
    ours = audio_summary(array)
    return (
        ours["dtype"] == recorded["dtype"]
        and ours["shape"] == recorded["shape"]
        and all(
            math.isclose(a, b, rel_tol=0, abs_tol=_PROJECTION_TOLERANCE)
            for a, b in zip(ours["projections"], recorded["projections"], strict=True)
        )
    )


def _near_miss_resample(clip: bytes, *, quality: str = "soxr_hq", length_rule: bool = True) -> np.ndarray:
    """`resample_16k_mono` with one step changed: soxr's quality, or soxr's own
    output length in place of librosa's fix_length to ceil(n * 16000 / sr)."""
    import soundfile as sf
    import soxr

    array, rate = sf.read(io.BytesIO(clip))
    array = array.T
    if array.ndim > 1:
        array = np.mean(array, axis=tuple(range(array.ndim - 1)))
    if rate == 16000:
        return array
    out = soxr.resample(array, rate, 16000, quality=quality)
    if length_rule:
        size = math.ceil(array.shape[-1] * 16000 / rate)
        out = out[:size] if len(out) >= size else np.pad(out, (0, size - len(out)))
    return out


def _rate(clip: bytes) -> int:
    import soundfile as sf

    return sf.info(io.BytesIO(clip)).samplerate


def test_every_synthetic_clip_resamples_as_datasets_audio_16k_did(fixture):
    recorded = fixture["audio"]
    clips = synthetic_inputs()["clips"]
    assert list(recorded) == list(clips) and len(clips) == len(synthetic_clips()) == 13
    for name, clip in clips.items():
        assert _matches_recorded(_asr_audio.resample_16k_mono(clip), recorded[name]), name


def test_the_audio_negative_controls_miss_the_recorded_samples(fixture):
    """soxr at "very high" quality misses every resampled clip; soxr's own output
    length (no librosa length rule) misses the 22,051-frame one."""
    recorded = fixture["audio"]
    clips = synthetic_inputs()["clips"]
    resampled = [name for name, clip in clips.items() if _rate(clip) != 16000]
    assert len(resampled) == 11
    for name in resampled:
        assert not _matches_recorded(_near_miss_resample(clips[name], quality="soxr_vhq"), recorded[name]), name
        assert _matches_recorded(_near_miss_resample(clips[name]), recorded[name]), name
    missed = [
        n for n in resampled if not _matches_recorded(_near_miss_resample(clips[n], length_rule=False), recorded[n])
    ]
    assert missed == ["synthetic:4"]


# ---------------------------------------------------------------------------
# Live: the reference code itself (PIPETTE_ASR_REFERENCE_ENV)
# ---------------------------------------------------------------------------

live = pytest.mark.skipif(
    reference_env() is None,
    reason="set PIPETTE_ASR_REFERENCE_ENV (scripts/setup_asr_reference_env.sh) to run the public references",
)


def test_a_configured_data_dir_must_hold_built_asr_datasets(tmp_path):
    """PIPETTE_SCORES_DATA_DIR set to a missing or empty directory, or to one
    with a stray asr_* directory, fails; unset checks no real data; a root with
    a built dataset (directly, or under datasets/) passes."""
    assert data_roots(None) == [] and data_roots("") == []
    with pytest.raises(ValueError, match="is not a directory"):
        data_roots(str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="holds no built ASR dataset"):
        data_roots(str(tmp_path))
    (tmp_path / "datasets" / "asr_esb_nonexistent" / "2026.10.1").mkdir(parents=True)
    with pytest.raises(ValueError, match="no ASR eval of the catalog"):
        data_roots(str(tmp_path))
    (tmp_path / "datasets" / "asr_esb_nonexistent" / "2026.10.1").rmdir()
    (tmp_path / "datasets" / "asr_esb_nonexistent").rmdir()
    built = tmp_path / "datasets" / "asr_fleurs_nl_nl" / "2026.10.1"
    built.mkdir(parents=True)
    with pytest.raises(ValueError, match="holds no built version"):
        data_roots(str(tmp_path))
    (built / "metadata.json").write_text("{}", encoding="utf-8")
    assert data_roots(str(tmp_path)) == [tmp_path / "datasets"]
    assert data_roots(os.pathsep.join([str(tmp_path / "datasets"), str(tmp_path)])) == [tmp_path / "datasets"] * 2
    with pytest.raises(ValueError, match="is not a directory"):
        data_roots(os.pathsep.join([str(tmp_path), str(tmp_path / "missing")]))


def test_the_answers_must_be_exactly_the_inputs_by_name():
    answers = [{"name": "a", "x": 1}, {"name": "b", "x": 2}]
    assert by_name(["a", "b"], answers, "k") == {"a": answers[0], "b": answers[1]}
    with pytest.raises(AssertionError, match="1 answers for 2 inputs"):
        by_name(["a", "b"], answers[:1], "k")
    with pytest.raises(AssertionError, match="3 answers for 2 inputs"):
        by_name(["a", "b"], [*answers, answers[0]], "k")
    with pytest.raises(AssertionError, match="answers are for"):
        by_name(["a", "b"], answers[::-1], "k")
    with pytest.raises(AssertionError, match="answers are for"):
        by_name(["a", "c"], answers, "k")


@pytest.fixture(scope="module")
def answers() -> dict[str, Any]:
    env = reference_env()
    assert env is not None
    inputs = synthetic_inputs()
    open_asr, fleurs, clips = inputs["open_asr"], inputs["fleurs"], inputs["clips"]
    fleurs.update(fleurs_transcripts())
    roots = data_roots(os.environ.get("PIPETTE_SCORES_DATA_DIR"))
    for root in roots:
        real = real_corpus(root)
        assert real["open_asr"] or real["fleurs"], root
        assert real["clips"], root
        for kind, named in (("open_asr", open_asr), ("fleurs", fleurs), ("clips", clips)):
            assert not set(real[kind]) & set(named), (root, kind)
            named.update(real[kind])
    answer = run_reference(env, open_asr, fleurs, clips)
    return {"answer": answer, "open_asr": open_asr, "fleurs": fleurs, "clips": clips, "roots": roots}


def _pairs(inputs: dict[str, Any], answers: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    assert list(inputs) == list(answers) and inputs
    return [(name, inputs[name], answers[name]) for name in inputs]


@live
def test_live_the_recorded_fixture_is_the_references_answer(answers):
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    expected = synthetic_fixture(answers["answer"])
    assert {k: v for k, v in recorded.items() if k not in ("note", "audio")} == {
        k: v for k, v in expected.items() if k not in ("note", "audio")
    }
    # The audio projections, to rounding (exact equality of the samples is the
    # resample test below).
    assert list(recorded["audio"]) == list(expected["audio"])
    for name, theirs in expected["audio"].items():
        ours = recorded["audio"][name]
        assert (ours["dtype"], ours["shape"]) == (theirs["dtype"], theirs["shape"]), name
        assert all(
            math.isclose(a, b, rel_tol=0, abs_tol=_PROJECTION_TOLERANCE)
            for a, b in zip(ours["projections"], theirs["projections"], strict=True)
        ), name


@live
def test_live_every_configured_data_dir_contributed_real_data(answers):
    for root in answers["roots"]:
        names = [n for kind in ("open_asr", "fleurs", "clips") for n in answers[kind] if n.startswith(f"{root}:")]
        assert names, root
    print("data roots:", answers["roots"])


@live
def test_live_every_english_set_scores_exactly_as_the_leaderboard(answers):
    checked = []
    for name, manifest, theirs in _pairs(answers["open_asr"], answers["answer"]["open_asr"]):
        ours = _score_open_asr(manifest)
        assert len(manifest) - len(ours.excluded_sample_ids) == theirs["kept"], name
        _assert_open_asr_equal(ours, theirs)
        checked.append((name, len(manifest)))
    print("open_asr checked:", checked)


@live
def test_live_without_merge_compounds_the_leaderboard_is_missed(answers):
    """The negative control on the live corpus: every utterance set with a split
    compound differs, so at least the synthetic one must."""
    differs = {
        name: _errors_without_merge(manifest) != theirs["totals"]["total"]
        for name, manifest, theirs in _pairs(answers["open_asr"], answers["answer"]["open_asr"])
        if not any("parent_id" in row for row in manifest)
    }
    assert differs["synthetic"]


@live
def test_live_every_fleurs_set_normalizes_and_scores_exactly_as_the_leaderboards_multilingual_code(answers):
    assert answers["answer"]["versions"]["num2words"] == "0.5.14"
    checked = []
    for name, dataset, theirs in _pairs(answers["fleurs"], answers["answer"]["fleurs"]):
        # The leaderboard's language code (from its config name) is ours.
        assert theirs["language"] == OPEN_ASR_LANGUAGE_CODES[dataset["language"]], name
        _assert_fleurs_equal(dataset, theirs)
        checked.append((name, len(dataset["pairs"])))
    assert sum(name.startswith("fleurs_tsv:") for name, _ in checked) == len(asr_datasets.FLEURS_LOCALES)
    print("fleurs checked:", checked)


@live
def test_live_the_vendored_multilingual_excerpts_are_the_leaderboards_source():
    """multilingual.py's FILLER_WORDS / MultilingualNormalizer and
    normalize_compound_pairs are the pinned checkout's text, character for
    character (only the module's header, imports and stand-in are pipette's)."""
    import inspect

    from pipette_scores.scoring.asr.open_asr_normalizer import multilingual

    env = reference_env()
    assert env is not None
    upstream = env / "open_asr_leaderboard" / "normalizer"
    data_utils = (upstream / "data_utils.py").read_text(encoding="utf-8")
    eval_utils = (upstream / "eval_utils.py").read_text(encoding="utf-8")
    ours = inspect.getsource(multilingual)
    filler_block = ours[ours.index("# Language-specific filler words") : ours.index("class MultilingualNormalizer")]
    assert filler_block in data_utils
    assert inspect.getsource(multilingual.MultilingualNormalizer) in data_utils
    assert inspect.getsource(multilingual.normalize_compound_pairs) in eval_utils


@live
def test_live_every_clip_resamples_exactly_as_datasets_audio_16k(answers):
    import soundfile as sf

    assert answers["answer"]["versions"]["datasets"] == "2.19.0"
    checked = 0
    for name, clip, theirs in _pairs(answers["clips"], answers["answer"]["audio"]):
        expected = reference_array(theirs)
        ours = _asr_audio.resample_16k_mono(clip)
        assert theirs["sampling_rate"] == 16000, name
        assert ours.dtype == expected.dtype and ours.shape == expected.shape, name
        assert np.array_equal(ours, expected), name
        # And the served WAV carries those samples as float32.
        served, rate = sf.read(io.BytesIO(base64.b64decode(_asr_audio.encode(clip))), dtype="float32")
        assert rate == 16000 and np.array_equal(served, expected.astype(np.float32)), name
        checked += 1
    print("clips checked:", checked)
