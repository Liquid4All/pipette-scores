"""The ASR evals end to end: builders, integrity, prompt profiles, audio, the samples stream and /score."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json
import pathlib
import re
import shutil
import uuid
from typing import Any
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from pipette_scores import cli, dataset_catalog, scoring
from pipette_scores.api import app as app_module
from pipette_scores.api.app import app
from pipette_scores.dataset_catalog import _asr_audio, _asr_build, _asr_common, asr_datasets, asr_prompt_profiles
from pipette_scores.dataset_catalog.asr_prompt_profiles import AsrClip
from pipette_scores.hashing import samples_digest, short_hash
from pipette_scores.types import ASR_EVAL_IDS, ASR_EVAL_ORDER, ChatMessage, EvalId, EvalSample, SampleCompletion

RATE = 16000


def _wav(seconds: float, *, subtype: str = "PCM_16", rate: int = RATE, channels: int = 1, fmt: str = "WAV") -> bytes:
    n = int(seconds * rate)
    t = np.arange(n, dtype=np.float32) / rate
    tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    data = np.stack([tone] * channels, axis=1)
    buffer = io.BytesIO()
    sf.write(buffer, data, rate, format=fmt, subtype=subtype)
    return buffer.getvalue()


def _source_rows(refs: list[str], *, seconds: float = 1.0, session: str | None = None) -> list[dict[str, Any]]:
    return [
        {
            "audio": _wav(seconds + 0.1 * i),
            "audio_format": "wav",
            "reference": ref,
            "session": session,
            "chunk_index": i if session else None,
            "raw_doc": {"id": f"utt-{i}", "text": ref, **({"parent_chunk_count": len(refs)} if session else {})},
        }
        for i, ref in enumerate(refs)
    ]


def _build(tmp_path: pathlib.Path, eval_id: EvalId, refs: list[str], **kw: Any) -> tuple[str, pathlib.Path]:
    """A fake dataset under tmp_path/datasets, named uniquely (the loaders cache)."""
    name = f"test-{uuid.uuid4().hex[:8]}"
    _asr_build.build(
        asr_datasets.ASR_DATASETS[eval_id], tmp_path / "datasets", name, rows=iter(_source_rows(refs, **kw))
    )
    return name, tmp_path / "datasets" / eval_id.value / name


def _rewrite(target: pathlib.Path, change) -> None:
    """Rewrite test.parquet's rows through ``change`` (metadata.json untouched)."""
    path = target / "test.parquet"
    table = pq.read_table(path)
    rows = change(table.to_pylist())
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path, row_group_size=_asr_build.ROW_GROUP_SIZE)


# ---------------------------------------------------------------------------
# Catalog and builders
# ---------------------------------------------------------------------------


def test_the_catalog_is_the_open_asr_leaderboards_public_english_sets_and_the_fleurs_locales():
    assert len(ASR_EVAL_IDS) == 16
    # The leaderboard's 9 public English sets, in its score_results column order
    # (AMI-Cleaned ... SPGISpeech, URGENT2024, Voice Arena Monsoon, Voxpopuli-AA-Cleaned).
    assert [e.value for e in asr_datasets.OPEN_ASR_EVAL_IDS] == [
        "asr_esb_ami_cleaned",
        "asr_esb_earnings22_cleaned_aa_chunked",
        "asr_esb_gigaspeech_cleaned",
        "asr_esb_librispeech_test_clean",
        "asr_esb_librispeech_test_other",
        "asr_esb_spgispeech",
        "asr_esb_urgent2024",
        "asr_esb_monsoon_en_in",
        "asr_esb_voxpopuli_cleaned_aa",
    ]
    # English and the Open ASR Leaderboard's multilingual FLEURS languages but
    # Armenian.
    assert [e.value.removeprefix("asr_fleurs_") for e in asr_datasets.FLEURS_EVAL_IDS] == [
        "en_us", "es_419", "pt_br", "de_de", "fr_fr", "it_it", "nl_nl",
    ]  # fmt: skip
    # The dropped locales are gone from the ids, not only from the catalog.
    for dropped in (
        "ja_jp",
        "pl_pl",
        "cs_cz",
        "da_dk",
        "el_gr",
        "fi_fi",
        "hu_hu",
        "ro_ro",
        "sv_se",
        "cmn_hans_cn",
        "ko_kr",
    ):
        assert f"asr_fleurs_{dropped}" not in {e.value for e in EvalId}, dropped
    assert list(asr_datasets.ASR_DATASETS) == list(ASR_EVAL_ORDER)
    english = [asr_datasets.ASR_DATASETS[e] for e in asr_datasets.OPEN_ASR_EVAL_IDS]
    fleurs = [asr_datasets.ASR_DATASETS[e] for e in asr_datasets.FLEURS_EVAL_IDS]
    assert all(d.normalizer.value == "open_asr_english" and d.open_asr_exclusions for d in english)
    assert all(d.aligner.value == "kaldialign_compound" for d in english)
    urgent = asr_datasets.ASR_DATASETS[EvalId.ASR_ESB_URGENT2024]
    assert (urgent.repo, urgent.revision, urgent.subset, urgent.split, urgent.expected_samples) == (
        "hf-audio/open-asr-leaderboard",
        "de79e57adf9983993bcc72586d7313460825f5eb",
        "urgent2024",
        "test",
        1000,
    )
    assert "non-commercial" in urgent.license and urgent.expected_source_sha256 == (
        "d34af39602eb850899831527a95a52c139023f3e4f7ee1706245ac4ecff18c34"
    )
    # Every one of the 16 sets has its fingerprint pinned, and only they do.
    pins = [d.expected_source_sha256 for d in asr_datasets.ASR_DATASETS.values()]
    assert len(pins) == 16 and all(re.fullmatch(r"[0-9a-f]{64}", p) for p in pins)
    assert set(asr_datasets.EXPECTED_SOURCE_SHA256) == {e.value for e in ASR_EVAL_ORDER}
    # FLEURS is scored with the leaderboard's multilingual scoring, every locale
    # against FLEURS's own `transcription` references.
    assert all(
        (d.normalizer.value, d.aligner.value, d.open_asr_exclusions, d.reference_column)
        == ("open_asr_multilingual", "kaldialign_compound_pairs", False, "transcription")
        for d in fleurs
    )
    assert {(d.repo, d.revision) for d in fleurs} == {("google/fleurs", "70bb2e84b976b7e960aa89f1c648e09c59f894dd")}


def test_a_build_keeps_order_audio_and_language(tmp_path):
    name, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een", "twee", "drie"])
    meta = json.loads((target / "metadata.json").read_text())
    assert meta["num_samples"] == 3 and meta["scoring"]["aligner"] == "kaldialign_compound_pairs"
    assert meta["limit"] is None and meta["source"]["reference_column"] == "transcription"
    assert meta["audio"]["subtypes"] == {"PCM_16": 3} and meta["audio"]["formats"] == {"WAV": 3}
    assert not (target / "test.parquet.partial").exists()
    samples = _asr_common.load_eval_samples(target)
    assert [s.reference for s in samples] == ["een", "twee", "drie"]
    assert [s.raw_doc["language_name"] for s in samples] == ["Dutch"] * 3
    assert samples[0].raw_doc["language_alpha_2"] == "nl"
    # The id is the position in the source split.
    assert samples[1].id == short_hash("asr_fleurs_nl_nl|1")


def test_a_build_streams_its_row_groups(tmp_path):
    _, target = _build(
        tmp_path, EvalId.ASR_FLEURS_NL_NL, [f"w{i}" for i in range(_asr_build.ROW_GROUP_SIZE + 5)], seconds=0.05
    )
    groups = pq.ParquetFile(target / "test.parquet").metadata
    assert [groups.row_group(i).num_rows for i in range(groups.num_row_groups)] == [_asr_build.ROW_GROUP_SIZE, 5]


def test_a_clip_off_16_khz_mono_is_stored_as_serving_would_bring_it(tmp_path):
    """A 16 kHz mono clip is kept as the source ships it. Any other (MonsoonASR is
    44.1 kHz) is brought to 16 kHz mono at build, as serving would have, so
    serving resamples nothing and its samples are the same on every CPU
    architecture; the row records the source's rate and channels."""
    clips = [_wav(1.0), _wav(0.5, rate=44100), _wav(0.4, rate=48000, channels=2, subtype="FLOAT")]
    rows = [
        {"audio": a, "audio_format": "wav", "reference": f"r{i}", "session": None, "chunk_index": None, "raw_doc": {}}
        for i, a in enumerate(clips)
    ]
    dataset = asr_datasets.ASR_DATASETS[EvalId.ASR_ESB_MONSOON_EN_IN]
    _asr_build.build(dataset, tmp_path / "datasets", "test-rates", rows=iter(rows))
    target = tmp_path / "datasets" / dataset.eval_id.value / "test-rates"
    stored = pq.read_table(target / "test.parquet").to_pylist()
    assert stored[0]["audio"] == clips[0]
    assert [r["audio"] for r in stored[1:]] == [_asr_audio.wav(_asr_audio.resample_16k_mono(c)) for c in clips[1:]]
    assert [(r["sample_rate"], r["channels"], r["num_frames"]) for r in stored] == [
        (16000, 1, 16000),
        (16000, 1, 8000),
        (16000, 1, 6400),
    ]
    sources = [json.loads(r["raw_doc"]) for r in stored]
    assert [(d.get("source_sample_rate"), d.get("source_channels")) for d in sources] == [
        (None, None),
        (44100, 1),
        (48000, 2),
    ]
    # Served, a stored clip is the samples serving brought the source clip to.
    for clip, row in zip(clips, stored, strict=True):
        assert np.array_equal(_asr_audio.to_16k_mono(row["audio"]), _asr_audio.to_16k_mono(clip))
    audio = json.loads((target / "metadata.json").read_text())["audio"]
    assert audio["sample_rates"] in ({16000: 3}, {"16000": 3})
    assert audio["source_sample_rates"] in ({16000: 1, 44100: 1, 48000: 1}, {"16000": 1, "44100": 1, "48000": 1})
    assert audio["channels"] in ({1: 3}, {"1": 3})
    assert audio["subtypes"] == {"FLOAT": 2, "PCM_16": 1}


def test_the_source_fingerprint_ignores_how_the_builder_resampled(tmp_path):
    """The catalog pins the source: a resample that rounds otherwise (another CPU
    architecture) moves content_sha256, never source_sha256, which equals the
    fingerprint of the same clips stored unresampled."""
    clips = [_wav(1.0), _wav(0.5, rate=44100), _wav(0.4, rate=48000, channels=2, subtype="FLOAT")]
    rows = [
        {"audio": a, "audio_format": "wav", "reference": f"r{i}", "session": None, "chunk_index": None, "raw_doc": {}}
        for i, a in enumerate(clips)
    ]
    dataset = asr_datasets.ASR_DATASETS[EvalId.ASR_ESB_MONSOON_EN_IN]
    meta = _asr_build.build(dataset, tmp_path / "datasets", "test-source", rows=iter(rows))
    stored = pq.read_table(tmp_path / "datasets" / dataset.eval_id.value / "test-source" / "test.parquet").to_pylist()
    assert meta["source_sha256"] == _asr_common.source_sha256(stored) != meta["content_sha256"]
    other_arch = [{**r, "audio_sha256": "0" * 64} if i else r for i, r in enumerate(stored)]
    assert _asr_common.source_sha256(other_arch) == meta["source_sha256"]
    assert _asr_common.content_sha256(other_arch) != meta["content_sha256"]
    # As the builder before build-time resampling stored them: the source bytes, rate and raw_doc.
    unresampled = [
        {
            **r,
            "audio_sha256": hashlib.sha256(c).hexdigest(),
            "sample_rate": sf.info(io.BytesIO(c)).samplerate,
            "channels": sf.info(io.BytesIO(c)).channels,
            "num_frames": sf.info(io.BytesIO(c)).frames,
            "raw_doc": json.dumps(
                {k: v for k, v in json.loads(r["raw_doc"]).items() if k not in _asr_common.SOURCE_KEYS},
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        for r, c in zip(stored, clips, strict=True)
    ]
    assert _asr_common.content_sha256(unresampled) == meta["source_sha256"]


def test_a_full_canonical_build_must_match_the_catalog(tmp_path):
    dataset = asr_datasets.ASR_DATASETS[EvalId.ASR_FLEURS_NL_NL]
    canonical = asr_datasets.DATASET_VERSION
    target = tmp_path / "datasets" / dataset.eval_id.value / canonical
    # The pinned 364 clips, not 2.
    with pytest.raises(SystemExit, match="expected 364"):
        _asr_build.build(dataset, tmp_path / "datasets", canonical, rows=iter(_source_rows(["een", "twee"])))
    assert not (target / "test.parquet").exists()
    # The right count but another fingerprint.
    two = dataclasses.replace(dataset, expected_samples=2)
    with pytest.raises(SystemExit, match="source_sha256 .* != expected"):
        _asr_build.build(two, tmp_path / "datasets", canonical, rows=iter(_source_rows(["een", "twee"])))
    assert not (target / "test.parquet").exists() and not (target / "test.parquet.partial").exists()


def test_a_smoke_build_may_not_take_the_canonical_version_and_is_never_served(tmp_path):
    dataset = asr_datasets.ASR_DATASETS[EvalId.ASR_FLEURS_NL_NL]
    with pytest.raises(SystemExit, match="canonical version"):
        _asr_build.build(
            dataset, tmp_path / "datasets", asr_datasets.DATASET_VERSION, limit=1, rows=iter(_source_rows(["a"]))
        )
    _asr_build.build(dataset, tmp_path / "datasets", "smoke", limit=1, rows=iter(_source_rows(["a", "b"])))
    with pytest.raises(_asr_common.AsrDatasetError, match="smoke build"):
        _asr_common.load_eval_samples(tmp_path / "datasets" / dataset.eval_id.value / "smoke")


def test_the_builder_refuses_an_output_inside_the_checkout(tmp_path):
    checkout = tmp_path / "repo"
    (checkout / ".git").mkdir(parents=True)
    with pytest.raises(SystemExit, match="inside the repository checkout"):
        _asr_build.refuse_inside_checkout(checkout / "datasets", checkout)
    _asr_build.refuse_inside_checkout(tmp_path / "elsewhere", checkout)
    # The real checkout's datasets/ (the old documented --out) is refused too.
    with pytest.raises(SystemExit, match="inside the repository checkout"):
        _asr_build.build(asr_datasets.ASR_DATASETS[EvalId.ASR_FLEURS_NL_NL], _asr_build._CHECKOUT / "datasets", "x")


# ---------------------------------------------------------------------------
# Integrity: the fingerprint, the catalog, the order, the served bytes
# ---------------------------------------------------------------------------


def test_a_tampered_digest_is_refused_at_load(tmp_path):
    _, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een"])
    meta = json.loads((target / "metadata.json").read_text())
    meta["content_sha256"] = "0" * 64
    (target / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(_asr_common.AsrDatasetError, match="content_sha256"):
        _asr_common.load_eval_samples(target)


def test_a_dataset_without_a_digest_is_refused(tmp_path):
    _, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een"])
    meta = json.loads((target / "metadata.json").read_text())
    del meta["content_sha256"]
    (target / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(_asr_common.AsrDatasetError, match="no content_sha256"):
        _asr_common.load_eval_samples(target)


def test_a_recipe_that_is_not_the_catalogs_is_refused(tmp_path):
    _, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een"])
    meta = json.loads((target / "metadata.json").read_text())
    meta["scoring"]["normalizer"] = "open_asr_english"
    (target / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(_asr_common.AsrDatasetError, match="scoring recipe"):
        _asr_common.load_eval_samples(target)


def test_the_canonical_version_must_carry_the_catalogs_fingerprint(tmp_path):
    _, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een"])
    canonical = target.with_name(asr_datasets.DATASET_VERSION)
    target.rename(canonical)
    meta = json.loads((canonical / "metadata.json").read_text())
    (canonical / "metadata.json").write_text(json.dumps(meta | {"dataset": asr_datasets.DATASET_VERSION}))
    with pytest.raises(_asr_common.AsrDatasetError, match="expected_source_sha256"):
        _asr_common.load_eval_samples(canonical)


def test_rows_out_of_position_are_refused(tmp_path):
    """Swapping two whole rows keeps the (order-free) digest, but not the order."""
    _, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een", "twee", "drie"])
    _rewrite(target, lambda rows: [rows[1], rows[0], rows[2]])
    with pytest.raises(_asr_common.AsrDatasetError, match="must be its position"):
        _asr_common.load_eval_samples(target)


def test_a_row_not_at_16_khz_mono_is_refused(tmp_path):
    """A build from before resampling moved to the builder: serving would resample it."""
    _, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een"])
    _rewrite(target, lambda rows: [{**r, "sample_rate": 44100} for r in rows])
    meta = json.loads((target / "metadata.json").read_text())
    meta["content_sha256"] = _asr_common.content_sha256(pq.read_table(target / "test.parquet").to_pylist())
    (target / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(_asr_common.AsrDatasetError, match="44100 Hz x 1"):
        _asr_common.load_eval_samples(target)


def test_swapped_audio_is_refused_when_served(tmp_path):
    """The columns (and so the digest) are intact; the served bytes are not."""
    name, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een", "twee"])

    def swap(rows):
        rows[0]["audio"], rows[1]["audio"] = rows[1]["audio"], rows[0]["audio"]
        return rows

    _rewrite(target, swap)
    _asr_common.load_eval_samples(target)  # scoring reads no audio
    with pytest.raises(_asr_common.AsrDatasetError, match="audio does not match"):
        list(_asr_common.iter_prompt_samples(EvalId.ASR_FLEURS_NL_NL, target, "lfm3"))


# ---------------------------------------------------------------------------
# Prompt profiles (each model's public usage guidance; asr_prompt_profiles.py)
# ---------------------------------------------------------------------------

AUDIO = {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}}


def _clip(eval_id: EvalId, language: str, name: str, alpha_2: str | None) -> AsrClip:
    return AsrClip(eval_id=eval_id, language=language, language_name=name, language_alpha_2=alpha_2)


ENGLISH_AMI = _clip(EvalId.ASR_ESB_AMI_CLEANED, "eng", "English", "en")
GERMAN = _clip(EvalId.ASR_FLEURS_DE_DE, "deu", "German", "de")
# No FLEURS locale of the catalog lies outside Voxtral's hinted languages; a
# clip that did (here a Polish one under a catalog id) is left to detection.
POLISH = _clip(EvalId.ASR_FLEURS_NL_NL, "pol", "Polish", "pl")


def test_gemma4_asks_with_googles_prompt_then_the_clip():
    [user] = asr_prompt_profiles.PROFILES["gemma4"].messages(ENGLISH_AMI, AUDIO)
    assert user["role"] == "user" and user["content"][1] == AUDIO
    assert user["content"][0]["text"] == (
        "Transcribe the following speech segment in English into English text.\n\n"
        "Follow these specific instructions for formatting the answer:\n"
        "* Only output the transcription, with no newlines.\n"
        "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, "
        "and write 3 instead of three."
    )
    [user] = asr_prompt_profiles.PROFILES["gemma4"].messages(GERMAN, AUDIO)
    assert user["content"][0]["text"].startswith("Transcribe the following speech segment in German into German text.")


def test_qwen25_omni_has_its_system_turn_and_the_clip_first():
    system, user = asr_prompt_profiles.PROFILES["qwen25_omni"].messages(GERMAN, AUDIO)
    assert system == {"role": "system", "content": "You are a speech recognition model."}
    assert user["content"] == [
        AUDIO,
        {"type": "text", "text": "Transcribe the German audio into text without any punctuation marks."},
    ]


@pytest.mark.parametrize(
    "clip, language",
    [(ENGLISH_AMI, "en"), (_clip(EvalId.ASR_FLEURS_NL_NL, "nld", "Dutch", "nl"), "nl"), (GERMAN, "de"), (POLISH, "")],
)
def test_voxtral_names_the_language_only_where_mistral_reports_it(clip, language):
    profile = asr_prompt_profiles.PROFILES["voxtral"]
    assert profile.messages(clip, AUDIO) == [{"role": "user", "content": [AUDIO]}]
    assert profile.chat_template_kwargs(clip) == {"transcription": True, "language": language}


_GENERAL = "Please listen to the audio carefully and transcribe it with high precision."
_LOWERCASE = (
    "Please listen to the audio carefully and transcribe it with high precision. "
    "Please output in lower case without punctuation."
)
# The FLEURS prompts, spelled out here independently of the profile.
_FLEURS_PROMPTS = {
    EvalId.ASR_FLEURS_DE_DE: (
        "deu",
        "German",
        "Bitte hören Sie sich die Audioaufnahme aufmerksam an und transkribieren Sie sie mit hoher Präzision. "
        "Transkribieren Sie sie wortgetreu in der Originalsprache (Deutsch); übersetzen Sie sie nicht.",
    ),
    EvalId.ASR_FLEURS_ES_419: (
        "spa",
        "Spanish",
        "Por favor, escuche atentamente el audio y transcríbalo con gran precisión. "
        "Transcríbalo palabra por palabra en el idioma original (español); no lo traduzca.",
    ),
    EvalId.ASR_FLEURS_PT_BR: (
        "por",
        "Portuguese",
        "Por favor, ouça o áudio com atenção e transcreva-o com alta precisão. "
        "Transcreva-o palavra por palavra no idioma original (português); não o traduza.",
    ),
    EvalId.ASR_FLEURS_FR_FR: (
        "fra",
        "French",
        "Veuillez écouter attentivement l'audio et le transcrire avec précision. "
        "Transcrivez-le mot pour mot dans la langue d'origine (français) ; ne le traduisez pas.",
    ),
    EvalId.ASR_FLEURS_IT_IT: (
        "ita",
        "Italian",
        "Per favore, ascolta attentamente l'audio e trascrivilo con alta precisione. "
        "Trascrivilo parola per parola nella lingua originale (italiano); non tradurlo.",
    ),
    EvalId.ASR_FLEURS_NL_NL: (
        "nld",
        "Dutch",
        "Luister aandachtig naar de audio en transcribeer deze met hoge precisie. "
        "Transcribeer het woordelijk in de oorspronkelijke taal (Nederlands); vertaal het niet.",
    ),
}


@pytest.mark.parametrize(
    "eval_id, language, name, prompt",
    [
        (
            EvalId.ASR_ESB_SPGISPEECH,
            "eng",
            "English",
            "Please listen to the financial domain audio and transcribe it accurately.",
        ),
        (EvalId.ASR_ESB_GIGASPEECH_CLEANED, "eng", "English", _LOWERCASE),
        (EvalId.ASR_ESB_VOXPOPULI_CLEANED_AA, "eng", "English", _LOWERCASE),
        (EvalId.ASR_ESB_LIBRISPEECH_TEST_CLEAN, "eng", "English", _LOWERCASE),
        (EvalId.ASR_ESB_LIBRISPEECH_TEST_OTHER, "eng", "English", _LOWERCASE),
        (EvalId.ASR_ESB_AMI_CLEANED, "eng", "English", _GENERAL),
        (EvalId.ASR_ESB_MONSOON_EN_IN, "eng", "English", _GENERAL),
        (EvalId.ASR_ESB_EARNINGS22_CLEANED_AA_CHUNKED, "eng", "English", _GENERAL),
        (EvalId.ASR_ESB_URGENT2024, "eng", "English", _GENERAL),
        # FLEURS English keeps OpenBMB's English prompt, with no added clause.
        (EvalId.ASR_FLEURS_EN_US, "eng", "English", _GENERAL),
        *((e, language, name, prompt) for e, (language, name, prompt) in _FLEURS_PROMPTS.items()),
    ],
)
def test_minicpm_o_takes_openbmbs_prompt_for_the_dataset(eval_id, language, name, prompt):
    clip = _clip(eval_id, language, name, None)
    [user] = asr_prompt_profiles.PROFILES["minicpm_o"].messages(clip, AUDIO)
    # OmniEvalKit's "<prompt>\n{audio}".
    assert user["content"] == [{"type": "text", "text": prompt + "\n"}, AUDIO]


def test_minicpm_o_asks_each_fleurs_language_in_that_language():
    """Every non-English FLEURS locale has its own prompt, none of them English:
    the English extension it replaced named the language in English, which the
    model then answered in."""
    non_english = [e for e in asr_datasets.FLEURS_EVAL_IDS if asr_datasets.ASR_DATASETS[e].language != "eng"]
    assert set(asr_prompt_profiles.MINICPM_FLEURS) == set(non_english) == set(_FLEURS_PROMPTS)
    for eval_id, (language, name, prompt) in _FLEURS_PROMPTS.items():
        sent = asr_prompt_profiles.minicpm_o_prompt(_clip(eval_id, language, name, None))
        assert sent == prompt and "Please" not in sent and "transcribe the content" not in sent, eval_id
    # French opens with OmniEvalKit's commonvoice_fr prompt, verbatim.
    assert asr_prompt_profiles.MINICPM_FLEURS[EvalId.ASR_FLEURS_FR_FR].startswith(
        "Veuillez écouter attentivement l'audio et le transcrire avec précision. "
    )
    # Negative control: a non-English clip with no prompt is refused, never
    # sent the English one.
    with pytest.raises(ValueError, match="no Polish prompt"):
        asr_prompt_profiles.minicpm_o_prompt(_clip(EvalId.ASR_ESB_AMI_CLEANED, "pol", "Polish", "pl"))


def test_the_lfm3_profile_names_the_language_in_the_system_turn():
    system, user = asr_prompt_profiles.PROFILES["lfm3"].messages(GERMAN, AUDIO)
    assert system == {"role": "system", "content": "Perform ASR in German."}
    assert user == {"role": "user", "content": [AUDIO]}
    system, _ = asr_prompt_profiles.PROFILES["lfm3"].messages(ENGLISH_AMI, AUDIO)
    assert system == {"role": "system", "content": "Perform ASR in English."}


@pytest.mark.parametrize("clip", [ENGLISH_AMI, GERMAN])
def test_lfm25_audio_always_asks_with_its_cards_fixed_prompt(clip):
    """The card's one prompt, "Perform ASR.", whatever the clip's language, with
    no language named (LFM3, on the same clip, names it: the negative control)."""
    system, user = asr_prompt_profiles.PROFILES["lfm25_audio"].messages(clip, AUDIO)
    assert system == {"role": "system", "content": "Perform ASR."}
    assert user == {"role": "user", "content": [AUDIO]}
    lfm3_system, _ = asr_prompt_profiles.PROFILES["lfm3"].messages(clip, AUDIO)
    assert lfm3_system != system


def test_every_profile_names_its_public_source():
    for name, profile in asr_prompt_profiles.PROFILES.items():
        assert profile.reference, name
        assert "@" in profile.reference or name == "lfm3", name


@pytest.mark.parametrize(
    "profile, raw, cleaned",
    [
        ("qwen25_omni", "system\nyou are\nassistant\n hello world ", "hello world"),
        # OpenBMB's text between <|tts_bos|> and <|tts_eos|>, stripped.
        ("minicpm_o", "<|tts_bos|> Hello there.<|tts_eos|> ignored", "Hello there."),
        ("minicpm_o", "Hello there.", "Hello there."),
        ("lfm25_audio", "Hello.<|im_end|>", "Hello."),
        ("lfm3", "Hello.<|im_end|>", "Hello.<|im_end|>"),
        ("lfm3", " Hello. ", " Hello. "),
        ("gemma4", " Hello. ", " Hello. "),
        ("voxtral", " Hello. ", " Hello. "),
    ],
)
def test_each_profile_cleans_the_answer_as_its_source_does(profile, raw, cleaned):
    assert asr_prompt_profiles.PROFILES[profile].postprocess(raw) == cleaned


# ---------------------------------------------------------------------------
# Audio: one resample to 16 kHz mono for every profile
# ---------------------------------------------------------------------------


def _decode(b64: str) -> tuple[np.ndarray, Any]:
    wav = base64.b64decode(b64)
    data, _ = sf.read(io.BytesIO(wav), dtype="float32")
    return data, sf.info(io.BytesIO(wav))


@pytest.mark.parametrize("profile", sorted(asr_prompt_profiles.PROFILES))
def test_every_profile_sends_the_same_16_khz_mono_clip(profile):
    p = asr_prompt_profiles.PROFILES[profile]
    cap = 480_000 if profile == "gemma4" else None
    assert p.audio_max_samples == cap
    # A 16 kHz clip is not resampled: its samples arrive unchanged (Gemma 4's cut at 30 s),
    # as float where they are finer than 16 bits ...
    clip = _wav(31.0, subtype="FLOAT")
    source, _ = sf.read(io.BytesIO(clip), dtype="float32")
    sent, info = _decode(_asr_common.encode_for(clip, p))
    assert (info.subtype, info.frames, info.samplerate, info.channels) == ("FLOAT", cap or len(source), RATE, 1)
    assert np.array_equal(sent, source[: cap or len(source)])
    # ... and as 16-bit PCM where they are 16-bit values, the same samples.
    clip = _wav(31.0, subtype="PCM_16")
    source, _ = sf.read(io.BytesIO(clip), dtype="float32")
    sent, info = _decode(_asr_common.encode_for(clip, p))
    assert (info.subtype, info.frames, info.samplerate, info.channels) == ("PCM_16", cap or len(source), RATE, 1)
    assert np.array_equal(sent, source[: cap or len(source)])
    # A 44.1 kHz stereo clip comes at 16 kHz mono, ceil(n * 16000 / 44100) samples,
    # as float: resampling leaves 16-bit values.
    sent, info = _decode(_asr_common.encode_for(_wav(2.0, rate=44100, channels=2), p))
    assert (info.samplerate, info.channels, info.subtype, info.frames) == (RATE, 1, "FLOAT", int(np.ceil(2.0 * RATE)))
    # 440 Hz survives the resampling.
    spectrum = np.abs(np.fft.rfft(sent))
    assert abs(np.argmax(spectrum) * RATE / len(sent) - 440) < 2


def test_a_clip_of_16_bit_values_is_served_as_16_bit_pcm_exactly():
    """Any float array whose every sample is k / 32768 for a 16-bit k, a float
    WAV's included (as some sources ship them), goes out as 16-bit PCM and
    decodes to the very same float32 samples; one finer sample keeps it float."""
    values = np.array([-32768, -1, 0, 1, 12345, 32767], dtype=np.int16)
    exact = values / 32768.0
    assert np.array_equal(_asr_audio.pcm16_values(exact), values)
    for source in (exact, exact.astype(np.float32)):
        wav = _asr_audio.wav(source)
        decoded, info = sf.read(io.BytesIO(wav), dtype="float32"), sf.info(io.BytesIO(wav))
        assert info.subtype == "PCM_16" and np.array_equal(decoded[0], exact.astype(np.float32))
    # The float32 a model receives is k * 2**-15 either way (llama.cpp's loader
    # multiplies by 0.000030517578125, soundfile divides by 32768).
    assert np.array_equal(values.astype(np.float32) * np.float32(2.0**-15), exact.astype(np.float32))
    for finer in (np.append(exact, 0.5 / 32768), np.append(exact, 1.0), np.append(exact, -1.0 - 1 / 32768)):
        assert _asr_audio.pcm16_values(finer) is None
        assert sf.info(io.BytesIO(_asr_audio.wav(finer))).subtype == "FLOAT"
    assert np.array_equal(_asr_audio.pcm16_values(np.zeros(0)), np.zeros(0, dtype=np.int16))


def test_the_served_pcm_wav_is_libsndfiles_pcm_16_layout():
    values = (np.random.default_rng(1).standard_normal(16001) * 3000).astype(np.int16)
    ours = _asr_audio.pcm16_wav(values)
    assert _chunks(ours) == [b"fmt ", b"data"]
    buffer = io.BytesIO()
    sf.write(buffer, values[:, np.newaxis], RATE, format="WAV", subtype="PCM_16")
    assert ours == buffer.getvalue()


def test_the_resample_is_soxr_hq_with_librosas_length_rule():
    """22,051 frames at 44.1 kHz are 8000.36 at 16 kHz: librosa's fix_length makes
    it ceil() = 8001 samples, soxr alone gives 8000."""
    import soxr

    clip = (0.1 * np.sin(2 * np.pi * 440 * np.arange(22051) / 44100)).astype(np.float64)
    buffer = io.BytesIO()
    sf.write(buffer, clip, 44100, format="WAV", subtype="DOUBLE")
    out = _asr_audio.resample_16k_mono(buffer.getvalue())
    plain = soxr.resample(clip, 44100, 16000, quality="soxr_hq")
    assert out.dtype == np.float64 and len(out) == 8001 and len(plain) == 8000
    assert np.array_equal(out[:8000], plain) and out[8000] == 0.0
    # Stereo is the mean of the channels, before resampling.
    stereo = np.stack([clip, 0.5 * clip], axis=1)
    buffer = io.BytesIO()
    sf.write(buffer, stereo, 44100, format="WAV", subtype="DOUBLE")
    mixed = _asr_audio.resample_16k_mono(buffer.getvalue())
    assert np.array_equal(mixed[:8000], soxr.resample(stereo.T.mean(axis=0), 44100, 16000, quality="soxr_hq"))


def _chunks(wav: bytes) -> list[bytes]:
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE" and int.from_bytes(wav[4:8], "little") == len(wav) - 8
    ids, i = [], 12
    while i < len(wav):
        size = int.from_bytes(wav[i + 4 : i + 8], "little")
        ids.append(wav[i : i + 4])
        i += 8 + size + (size & 1)
    assert i == len(wav)
    return ids


def _libsndfile_float_wav(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples[:, np.newaxis], RATE, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def test_the_served_wav_is_libsndfiles_float_layout_without_its_peak_chunk():
    samples = np.random.default_rng(0).standard_normal(16001) * 0.3
    ours = _asr_audio.float_wav(samples)
    assert _chunks(ours) == [b"fmt ", b"fact", b"data"]
    # libsndfile's own FLOAT WAV, its PEAK chunk (and the RIFF size) aside, is the same bytes.
    theirs = _libsndfile_float_wav(samples)
    assert _chunks(theirs) == [b"fmt ", b"fact", b"PEAK", b"data"]
    peak = theirs.index(b"PEAK")
    without = theirs[:peak] + theirs[peak + 8 + 16 :]
    assert ours == without[:4] + (len(without) - 8).to_bytes(4, "little") + without[8:]
    decoded, rate = sf.read(io.BytesIO(ours), dtype="float32")
    assert rate == RATE and np.array_equal(decoded, samples.astype(np.float32))


def test_a_clip_is_served_as_the_same_bytes_on_every_call():
    """The served WAV depends on the samples alone. The negative control:
    libsndfile's FLOAT WAV stamps its PEAK chunk with the time, so two writes
    in different seconds differ."""
    import time

    clip = _wav(0.7, rate=44100, channels=2)
    mono = _asr_audio.resample_16k_mono(clip)
    first, libsndfile_first = _asr_audio.encode(clip), _libsndfile_float_wav(mono)
    second_of_first = int(time.time())
    while int(time.time()) == second_of_first:
        time.sleep(0.01)
    second, libsndfile_second = _asr_audio.encode(clip), _libsndfile_float_wav(mono)
    assert first == second
    assert libsndfile_first != libsndfile_second


# ---------------------------------------------------------------------------
# API: the samples stream and its digest
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_catalog, "_DEFAULT_ROOT", tmp_path)
    monkeypatch.setattr(app_module, "_DIGESTS", {})
    return TestClient(app)


def _samples(resp) -> list[EvalSample]:
    return [EvalSample.model_validate(s) for s in resp.json()["samples"]]


def test_an_asr_set_is_streamed_whole_with_its_digest(client, tmp_path):
    name, _ = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een", "twee", "drie"])
    resp = client.get(f"/evals/asr_fleurs_nl_nl/datasets/{name}/samples?prompt_profile=voxtral")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/json"
    body = resp.json()
    assert set(body) == {"samples"}
    samples = _samples(resp)
    assert [s.id for s in samples] == [short_hash(f"asr_fleurs_nl_nl|{i}") for i in range(3)]
    sample = body["samples"][1]
    assert sample["chat_template_kwargs"] == {"transcription": True, "language": "nl"}
    [part] = sample["messages"][0]["content"]
    _, info = _decode(part["input_audio"]["data"])
    # The test clips are 16-bit PCM at 16 kHz: served as such, bit for bit.
    assert part["input_audio"]["format"] == "wav" and info.subtype == "PCM_16" and info.frames == int(1.1 * RATE)
    # The header is the body's digest, and the digest route says the same.
    digest = resp.headers["x-samples-digest"]
    assert digest == samples_digest(samples)
    route = client.get(f"/evals/asr_fleurs_nl_nl/datasets/{name}/samples/digest?prompt_profile=voxtral").json()
    assert route == {"digest": digest, "sample_count": 3}
    # A profile without template arguments leaves the field out of the wire shape.
    resp = client.get(f"/evals/asr_fleurs_nl_nl/datasets/{name}/samples?prompt_profile=lfm3")
    assert "chat_template_kwargs" not in resp.json()["samples"][0]


def test_a_build_records_where_the_set_may_be_kept_and_the_text_moves_no_pin(client, tmp_path):
    """metadata.json's `redistribution` is the builder's generic use policy (a deployment
    records where it keeps the sets itself), and no fingerprint reads it: the same set
    reworded loads and serves the same samples digest."""
    name, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een", "twee"])
    meta = json.loads((target / "metadata.json").read_text())
    assert meta["redistribution"] == _asr_build.REDISTRIBUTION
    assert "never commit it to a public repository" in meta["redistribution"]
    assert "docs/dataset-licenses.md" not in meta["redistribution"]
    digest = "/evals/asr_fleurs_nl_nl/datasets/{}/samples/digest?prompt_profile=voxtral"
    before = client.get(digest.format(name))
    assert before.status_code == 200, before.text
    # A copy under another name, since the loaders cache by directory.
    reworded = target.parent / f"{name}-reworded"
    shutil.copytree(target, reworded)
    meta["redistribution"] = "internal use only: reworded"
    (reworded / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    after = client.get(digest.format(reworded.name))
    assert after.status_code == 200, after.text
    assert after.json() == before.json()


def test_the_body_is_the_canonical_form_its_digest_hashes(client, tmp_path):
    """A client can hash the stream's samples without re-encoding them."""
    import hashlib

    name, _ = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een", "twee"])
    resp = client.get(f"/evals/asr_fleurs_nl_nl/datasets/{name}/samples?prompt_profile=gemma4")
    raw = resp.content
    assert raw.startswith(b'{"samples":[{') and raw.endswith(b"}]}")
    lines = raw[len(b'{"samples":[') : -len(b"]}")].split(b',{"id":')
    lines = [lines[0]] + [b'{"id":' + rest for rest in lines[1:]]
    assert len(lines) == 2
    assert (
        resp.headers["x-samples-digest"] == "sha256:" + hashlib.sha256(b"".join(x + b"\n" for x in lines)).hexdigest()
    )


def test_one_pass_gives_every_profiles_digest(tmp_path):
    """prompt_digests encodes each clip once per cut and hashes it into every
    profile; each must be the digest of that profile's own stream, Gemma 4's 30 s
    cut included (a 31 s clip)."""
    rows = _source_rows(["een", "twee"], seconds=1.0)
    rows.append({**rows[0], "audio": _wav(31.0), "raw_doc": {"id": "long"}})
    name = f"test-{uuid.uuid4().hex[:8]}"
    _asr_build.build(asr_datasets.ASR_DATASETS[EvalId.ASR_FLEURS_NL_NL], tmp_path / "datasets", name, rows=iter(rows))
    target = tmp_path / "datasets" / "asr_fleurs_nl_nl" / name
    digests = _asr_common.prompt_digests(EvalId.ASR_FLEURS_NL_NL, target)
    assert set(digests) == set(asr_prompt_profiles.PROFILES)
    for profile, (digest, count) in digests.items():
        streamed = list(_asr_common.iter_prompt_samples(EvalId.ASR_FLEURS_NL_NL, target, profile))
        assert (digest, count) == (samples_digest(streamed), 3), profile
    # Each profile's samples are its own: six profiles, six digests.
    assert len({d for d, _ in digests.values()}) == len(digests)


def test_a_cold_digest_learns_every_profile_at_once(client, tmp_path, monkeypatch):
    name, _ = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een"])
    calls = []
    real = dataset_catalog.asr_prompt_digests

    def spy(*args, **kw):
        calls.append(args)
        return real(*args, **kw)

    monkeypatch.setattr(dataset_catalog, "asr_prompt_digests", spy)
    for profile in asr_prompt_profiles.PROFILES:
        resp = client.get(f"/evals/asr_fleurs_nl_nl/datasets/{name}/samples/digest?prompt_profile={profile}")
        assert resp.status_code == 200, (profile, resp.text)
    assert len(calls) == 1


@pytest.mark.parametrize("route", ["samples", "samples/digest"])
@pytest.mark.parametrize(
    "query, status",
    [
        ("", 400),
        ("prompt_profile=nope", 400),
        ("prompt_profile=lfm3&offset=0&limit=1", 400),
    ],
)
def test_an_asr_samples_request_needs_a_profile_and_is_not_paged(client, tmp_path, route, query, status):
    name, _ = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een"])
    resp = client.get(f"/evals/asr_fleurs_nl_nl/datasets/{name}/{route}?{query}")
    assert resp.status_code == status, resp.text


def test_a_missing_dataset_is_a_404(client, tmp_path):
    _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een"])
    for route in ("samples", "samples/digest"):
        assert client.get(f"/evals/asr_fleurs_nl_nl/datasets/nope/{route}?prompt_profile=lfm3").status_code == 404


def test_a_dataset_that_fails_its_checks_is_a_500_before_any_byte(client, tmp_path):
    name, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een", "twee"])
    _rewrite(target, lambda rows: [rows[1], rows[0]])
    resp = client.get(f"/evals/asr_fleurs_nl_nl/datasets/{name}/samples?prompt_profile=lfm3")
    assert resp.status_code == 500 and "position" in resp.json()["detail"]


def test_a_stream_that_strays_from_its_digest_is_cut_off(client, tmp_path, monkeypatch):
    """The header went out first, so a body that would not hash to it never
    ends: the client is left with JSON it cannot parse."""
    name, _ = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een", "twee"])
    key = ("asr_fleurs_nl_nl", name, "lfm3")
    app_module._DIGESTS[key] = ("sha256:" + "0" * 64, 2)
    with pytest.raises(RuntimeError, match="other samples than its digest"):
        client.get(f"/evals/asr_fleurs_nl_nl/datasets/{name}/samples?prompt_profile=lfm3")


def test_a_44_1_khz_or_stereo_clip_is_served_at_16_khz_mono_to_every_profile(client, tmp_path):
    rows = [
        {"audio": a, "audio_format": "wav", "reference": "r", "session": None, "chunk_index": None, "raw_doc": {}}
        for a in (_wav(0.5, rate=44100), _wav(0.5, rate=16000, channels=2))
    ]
    _asr_build.build(
        asr_datasets.ASR_DATASETS[EvalId.ASR_ESB_MONSOON_EN_IN], tmp_path / "datasets", "rates", rows=iter(rows)
    )
    url = "/evals/asr_esb_monsoon_en_in/datasets/rates/samples?prompt_profile={}"
    body = client.get(url.format("voxtral")).json()
    [part] = body["samples"][0]["messages"][0]["content"]
    _, info = _decode(part["input_audio"]["data"])
    assert (info.samplerate, info.channels, info.frames, info.subtype) == (RATE, 1, 8000, "FLOAT")
    for profile in asr_prompt_profiles.PROFILES:
        resp = client.get(url.format(profile))
        assert resp.status_code == 200, (profile, resp.text)
        parts = [
            p for m in resp.json()["samples"][1]["messages"] if isinstance(m["content"], list) for p in m["content"]
        ]
        [audio] = [p for p in parts if p["type"] == "input_audio"]
        _, info = _decode(audio["input_audio"]["data"])
        # Two identical 16-bit channels average to 16-bit values: PCM again.
        assert (info.samplerate, info.channels, info.subtype) == (RATE, 1, "PCM_16")


def test_a_text_eval_keeps_its_unpaged_shape(client, tmp_path):
    prompts = [EvalSample(id=str(i), messages=[ChatMessage(role="user", content="Q?")]) for i in range(3)]
    with patch("pipette_scores.api.app._load_prompts_cached", return_value=prompts):
        whole = client.get("/evals/ifstruct/datasets/test/samples").json()
    assert set(whole) == {"samples"} and len(whole["samples"]) == 3
    assert "chat_template_kwargs" not in whole["samples"][0]


# ---------------------------------------------------------------------------
# API: /score
# ---------------------------------------------------------------------------


def _score(client: TestClient, eval_id: EvalId, name: str, completions: list[dict[str, Any]], **extra: Any):
    body = {"eval_id": eval_id.value, "dataset_name": name, "completions": completions, **extra}
    return client.post("/score", json=body)


def test_score_reports_the_corpus_rate_and_each_samples_alignment(client, tmp_path):
    name, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een twee drie vier", "vijf zes"])
    ids = [s.id for s in _asr_common.load_eval_samples(target)]
    resp = _score(
        client,
        EvalId.ASR_FLEURS_NL_NL,
        name,
        [
            # The profile keeps the text after the last "\nassistant\n".
            {"id": ids[0], "completion": "system\nx\nassistant\nEen, twee, DRIE!", "stop_reason": "eos"},
            {"id": ids[1], "completion": "vijf zeven", "stop_reason": "truncated"},
        ],
        prompt_profile="qwen25_omni",
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    ctx = out["context"]
    # One deletion (vier) and one substitution (zeven) over six reference words.
    assert ctx["error_unit"] == "word" and ctx["units_scored"] == 2 and ctx["samples_total"] == 2
    assert ctx["error_rate"] == pytest.approx(2 / 6) and ctx["result"] == pytest.approx(100 * 2 / 6)
    assert ctx["stop_reasons"] == {"eos": 1, "truncated": 1}
    assert not {"complete", "samples_missing", "samples_failed", "units_unscored"} & set(ctx)
    first, second = out["scored_samples"]
    assert first["measures"]["deletions"] == 1 and first["measures"]["hypothesis"] == "een twee drie"
    assert second["measures"]["substitutions"] == 1 and not second["is_correct"]


def test_scored_samples_keep_their_text_and_an_empty_audio_part_in_its_place(client, tmp_path):
    name, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een"])
    [sample] = _asr_common.load_eval_samples(target)
    [served] = client.get(f"/evals/asr_fleurs_nl_nl/datasets/{name}/samples?prompt_profile=qwen25_omni").json()[
        "samples"
    ]
    out = _score(
        client, EvalId.ASR_FLEURS_NL_NL, name, [{"id": sample.id, "completion": "een"}], prompt_profile="qwen25_omni"
    ).json()
    [scored] = out["scored_samples"]
    # The turns and parts as served, in order; the audio part keeps its format, not its data.
    assert scored["messages"] == [
        {"role": "system", "content": "You are a speech recognition model."},
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}},
                {"type": "text", "text": "Transcribe the Dutch audio into text without any punctuation marks."},
            ],
        },
    ]
    audio = served["messages"][1]["content"][0]
    assert audio["type"] == "input_audio" and audio["input_audio"]["data"]  # served with its clip
    assert [m["role"] for m in served["messages"]] == [m["role"] for m in scored["messages"]]
    assert [[p["type"] for p in m["content"]] for m in served["messages"][1:]] == [
        [p["type"] for p in m["content"]] for m in scored["messages"][1:]
    ]


def test_scored_samples_never_carry_an_audio_payload(client, tmp_path):
    """Whatever the prompts hold, the response empties each audio part."""
    name, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een"])
    [sample] = _asr_common.load_eval_samples(target)
    prompt = EvalSample(
        id=sample.id,
        messages=[ChatMessage(role="user", content=[AUDIO, {"type": "text", "text": "Transcribe."}])],
    )
    with patch("pipette_scores.api.app._load_prompts_cached", return_value=[prompt]):
        resp = _score(
            client, EvalId.ASR_FLEURS_NL_NL, name, [{"id": sample.id, "completion": "een"}], prompt_profile="lfm3"
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["scored_samples"][0]["messages"][0]["content"] == [
        {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}},
        {"type": "text", "text": "Transcribe."},
    ]
    assert prompt.messages[0].content[0] == AUDIO  # the cached prompt is untouched


def test_score_shows_the_template_arguments_each_sample_was_sent_with(client, tmp_path):
    name, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een"])
    [sample] = _asr_common.load_eval_samples(target)
    out = _score(
        client, EvalId.ASR_FLEURS_NL_NL, name, [{"id": sample.id, "completion": "een"}], prompt_profile="voxtral"
    ).json()
    assert out["scored_samples"][0]["chat_template_kwargs"] == {"transcription": True, "language": "nl"}
    out = _score(
        client, EvalId.ASR_FLEURS_NL_NL, name, [{"id": sample.id, "completion": "een"}], prompt_profile="lfm3"
    ).json()
    assert "chat_template_kwargs" not in out["scored_samples"][0]


def test_score_drops_open_asr_references_that_normalize_away(client, tmp_path):
    name, target = _build(
        tmp_path, EvalId.ASR_ESB_VOXPOPULI_CLEANED_AA, ["hello world", "<unk>", "ignore time segment in scoring"]
    )
    ids = [s.id for s in _asr_common.load_eval_samples(target)]
    out = _score(
        client,
        EvalId.ASR_ESB_VOXPOPULI_CLEANED_AA,
        name,
        [{"id": i, "completion": "hello world"} for i in ids],
        prompt_profile="lfm25_audio",
    ).json()
    ctx = out["context"]
    assert (ctx["samples_excluded"], ctx["units_scored"], ctx["error_rate"]) == (2, 1, 0.0)
    assert [s["measures"] for s in out["scored_samples"][1:]] == [{"excluded": True}, {"excluded": True}]


def _earnings22(tmp_path: pathlib.Path) -> tuple[str, list[str]]:
    call = "Revenue grew five percent this quarter."
    name, target = _build(tmp_path, EvalId.ASR_ESB_EARNINGS22_CLEANED_AA_CHUNKED, [call] * 3, session="call-1")
    return name, [s.id for s in _asr_common.load_eval_samples(target)]


def test_an_earnings22_call_scores_once_and_its_measures_sum_to_the_totals(client, tmp_path):
    name, ids = _earnings22(tmp_path)
    out = _score(
        client,
        EvalId.ASR_ESB_EARNINGS22_CLEANED_AA_CHUNKED,
        name,
        # Out of order, one chunk blank: joined in chunk order, empties skipped.
        [
            {"id": ids[2], "completion": "this quarter. "},
            {"id": ids[0], "completion": " Revenue grew"},
            {"id": ids[1], "completion": " 5% "},
        ],
        prompt_profile="gemma4",
    ).json()
    ctx = out["context"]
    assert (ctx["units_scored"], ctx["error_rate"], ctx["reference_units"]) == (1, 0.0, 5)
    measures = {s["id"]: s["measures"] for s in out["scored_samples"]}
    # The call's counts and texts once, on its first chunk; the others name the call only.
    assert measures[ids[0]]["unit"] == "call-1" and measures[ids[0]]["reference_units"] == 5
    assert measures[ids[1]] == measures[ids[2]] == {"unit": "call-1"}
    for key in ("hits", "substitutions", "deletions", "insertions", "reference_units"):
        assert sum(m.get(key, 0) for m in measures.values()) == ctx[key]


@pytest.mark.parametrize(
    "answers, names",
    [
        # One sample unanswered.
        ([("een twee", None), ("drie", None)], ["unanswered: {2}"]),
        # One generation failed.
        ([("een twee", None), ("", "failure"), ("vier", "eos")], ["failed: {1}"]),
        # Both, as the old incomplete result had them.
        ([("een twee", None), ("", "failure")], ["unanswered: {2}", "failed: {1}"]),
    ],
)
def test_a_run_without_an_answer_to_every_sample_is_refused(client, tmp_path, answers, names):
    name, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een twee", "drie", "vier"])
    ids = [s.id for s in _asr_common.load_eval_samples(target)]
    completions = [
        {"id": ids[i], "completion": text, **({"stop_reason": reason} if reason else {})}
        for i, (text, reason) in enumerate(answers)
    ]
    with patch("pipette_scores.api.app.scoring.score") as score:
        resp = _score(client, EvalId.ASR_FLEURS_NL_NL, name, completions, prompt_profile="lfm3")
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail.startswith("an ASR eval is scored only with an answer to every sample; ")
    for label in names:
        assert label.format(*ids) in detail
    score.assert_not_called()
    # Answered in full, the same run scores.
    complete = [{**c, "completion": c["completion"] or "drie", "stop_reason": "eos"} for c in completions]
    complete += [{"id": i, "completion": "vier"} for i in ids[len(complete) :]]
    ok = _score(client, EvalId.ASR_FLEURS_NL_NL, name, complete, prompt_profile="lfm3")
    assert ok.status_code == 200, ok.text
    assert ok.json()["context"]["error_rate"] == pytest.approx(0.0)


def test_the_scorer_itself_refuses_a_failed_or_missing_answer(tmp_path):
    """For any direct caller: a result is an answer to every sample, or none."""
    _, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een", "twee"])
    samples = _asr_common.load_eval_samples(target)
    answered = [SampleCompletion(id=s.id, completion=s.reference) for s in samples]
    failed = [answered[0], SampleCompletion(id=samples[1].id, completion="", stop_reason="failure")]
    with pytest.raises(ValueError, match=f"failed: {samples[1].id}"):
        scoring.score(EvalId.ASR_FLEURS_NL_NL, failed, samples, prompt_profile="lfm3")
    with pytest.raises(ValueError, match=f"unanswered: {samples[1].id}"):
        scoring.score(EvalId.ASR_FLEURS_NL_NL, answered[:1], samples, prompt_profile="lfm3")
    _, context = scoring.score(EvalId.ASR_FLEURS_NL_NL, answered, samples, prompt_profile="lfm3")
    assert context["error_rate"] == 0.0


def test_an_excluded_sample_still_needs_its_answer(client, tmp_path):
    name, target = _build(tmp_path, EvalId.ASR_ESB_VOXPOPULI_CLEANED_AA, ["hello world", "<unk>"])
    ids = [s.id for s in _asr_common.load_eval_samples(target)]
    resp = _score(
        client,
        EvalId.ASR_ESB_VOXPOPULI_CLEANED_AA,
        name,
        [{"id": ids[0], "completion": "hello world"}],
        prompt_profile="lfm25_audio",
    )
    assert resp.status_code == 400 and f"unanswered: {ids[1]}" in resp.json()["detail"]


def test_an_earnings22_call_missing_a_chunk_is_refused(client, tmp_path):
    name, ids = _earnings22(tmp_path)
    resp = _score(
        client,
        EvalId.ASR_ESB_EARNINGS22_CLEANED_AA_CHUNKED,
        name,
        [{"id": ids[0], "completion": "Revenue grew"}, {"id": ids[2], "completion": "this quarter."}],
        prompt_profile="gemma4",
    )
    assert resp.status_code == 400 and f"unanswered: {ids[1]}" in resp.json()["detail"]


def test_ids_twice_or_unknown_are_refused_as_before(client, tmp_path):
    name, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een"])
    [sample] = _asr_common.load_eval_samples(target)
    twice = [{"id": sample.id, "completion": "een"}] * 2
    resp = _score(client, EvalId.ASR_FLEURS_NL_NL, name, twice, prompt_profile="lfm3")
    assert resp.status_code == 400 and "duplicate completion ids" in resp.json()["detail"]
    unknown = [{"id": sample.id, "completion": "een"}, {"id": "nope", "completion": "x"}]
    resp = _score(client, EvalId.ASR_FLEURS_NL_NL, name, unknown, prompt_profile="lfm3")
    assert resp.status_code == 400 and "not in dataset: nope" in resp.json()["detail"]


def test_score_refuses_thinking_and_a_missing_profile(client, tmp_path):
    name, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een"])
    [sample] = _asr_common.load_eval_samples(target)
    completions = [{"id": sample.id, "completion": "een"}]
    assert _score(client, EvalId.ASR_FLEURS_NL_NL, name, completions).status_code == 400
    resp = _score(client, EvalId.ASR_FLEURS_NL_NL, name, completions, prompt_profile="gemma4", enable_thinking=True)
    assert resp.status_code == 400 and "thinking" in resp.json()["detail"]
    ok = _score(client, EvalId.ASR_FLEURS_NL_NL, name, completions, prompt_profile="gemma4", enable_thinking=False)
    assert ok.status_code == 200


def test_score_checks_the_samples_the_run_was_given(client, tmp_path):
    name, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een"])
    [sample] = _asr_common.load_eval_samples(target)
    completions = [{"id": sample.id, "completion": "een"}]
    served = client.get(f"/evals/asr_fleurs_nl_nl/datasets/{name}/samples/digest?prompt_profile=gemma4").json()
    ok = _score(
        client, EvalId.ASR_FLEURS_NL_NL, name, completions, prompt_profile="gemma4", samples_digest=served["digest"]
    )
    assert ok.status_code == 200, ok.text
    # Another profile's samples are other prompts.
    other = client.get(f"/evals/asr_fleurs_nl_nl/datasets/{name}/samples/digest?prompt_profile=lfm3").json()
    moved = _score(
        client, EvalId.ASR_FLEURS_NL_NL, name, completions, prompt_profile="gemma4", samples_digest=other["digest"]
    )
    assert moved.status_code == 409 and served["digest"] in moved.json()["detail"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(monkeypatch, capsys, *argv: str) -> str:
    monkeypatch.setattr("sys.argv", ["pipette-scores", *argv])
    cli.main()
    return capsys.readouterr().out


def test_the_cli_scores_an_asr_run_with_its_profile(tmp_path, monkeypatch, capsys):
    name, target = _build(tmp_path, EvalId.ASR_FLEURS_NL_NL, ["een twee", "drie"])
    ids = [s.id for s in _asr_common.load_eval_samples(target)]
    completions = tmp_path / "completions.jsonl"
    completions.write_text("".join(json.dumps({"id": i, "completion": "een twee<|im_end|>"}) + "\n" for i in ids))
    root = ["--data-dir", str(tmp_path)]
    out = _cli(monkeypatch, capsys, *root, "score", "--eval", "asr_fleurs_nl_nl", "--dataset", name,
               "--prompt-profile", "lfm3", "-f", str(completions), "--no-samples")  # fmt: skip
    # een twee: right; drie -> "een twee": one substitution, one insertion, over 3 words.
    assert "WER" in out and f"{100 * 2 / 3:.4f}%" in out and "ACCURACY" not in out
    with pytest.raises(SystemExit, match="--prompt-profile"):
        _cli(
            monkeypatch, capsys, *root, "score", "--eval", "asr_fleurs_nl_nl", "--dataset", name, "-f", str(completions)
        )
    completions.write_text(json.dumps({"id": ids[0], "completion": "een twee"}) + "\n")
    with pytest.raises(SystemExit, match=f"answer to every sample; unanswered: {ids[1]}"):
        _cli(monkeypatch, capsys, *root, "score", "--eval", "asr_fleurs_nl_nl", "--dataset", name,
             "--prompt-profile", "lfm3", "-f", str(completions))  # fmt: skip
    completions.write_text(json.dumps({"id": "nope", "completion": "x"}) + "\n")
    with pytest.raises(SystemExit, match="not in"):
        _cli(monkeypatch, capsys, *root, "score", "--eval", "asr_fleurs_nl_nl", "--dataset", name,
             "--prompt-profile", "lfm3", "-f", str(completions))  # fmt: skip
    out = _cli(monkeypatch, capsys, *root, "prompts", "--eval", "asr_fleurs_nl_nl", "--dataset", name,
               "--prompt-profile", "qwen25_omni")  # fmt: skip
    first = json.loads(out.splitlines()[0])
    assert first["id"] == ids[0] and first["messages"][0]["content"] == "You are a speech recognition model."
