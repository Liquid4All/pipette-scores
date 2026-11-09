"""How each model family is asked to transcribe a clip: the ASR prompt profiles.

An ASR dataset holds the clip and its reference only; a profile turns a clip into
the request the model's own public usage guidance describes (its model card,
cookbook or official evaluation configuration): the turns and their order, the
prompt wording, the chat-template arguments, the model's audio length limit,
and the cleanup of the answer before scoring. ``AsrProfile.reference`` names
each source at a pinned revision. Where a source gives no wording for a
language, the profile extends it, and says so (``pipette's extension``); every
other step that is pipette's own is labelled the same way.

Every profile receives the same audio: 16 kHz mono, resampled once as the Open
ASR Leaderboard prepares audio for every model (_asr_audio), sent as a WAV that
carries those samples exactly (16-bit PCM where they are 16-bit values, 32-bit
float otherwise).

Generation settings are not here; the benchmark definition carries them. Nor is
what the server must provide (docs/asr-evals.md, "Server prerequisites"):
thinking off (no profile sends ``enable_thinking``), Voxtral's transcription
chat template, LFM3's turn format, and MiniCPM-o's ``use_tts_template`` switch.

Profiles:

- ``gemma4``: no system turn; the user turn is Google's ASR prompt, then the clip
  (the card puts audio after the text), cut to 30 s.
- ``qwen25_omni``: system "You are a speech recognition model."; the user turn
  is the clip, then the cookbook's prompt; the answer is the text after the last
  "\\nassistant\\n", stripped.
- ``voxtral``: the clip alone in transcription mode; the language is named for
  the nine languages Mistral evaluates Voxtral's transcription on, and left to
  automatic detection otherwise.
- ``minicpm_o``: no system turn; the user turn is OpenBMB's evaluation prompt for
  the dataset, written in the clip's language, followed by a newline, then the
  clip; the answer is the text between ``<|tts_bos|>`` and ``<|tts_eos|>``,
  stripped.
- ``lfm25_audio``: system "Perform ASR." (the card's fixed prompt, in every
  language); the clip alone; a trailing ``<|im_end|>`` dropped.
- ``lfm3``: LFM3's native ASR chat format: system "Perform ASR in {language}.",
  the clip alone; the answer as decoded.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from pipette_scores.types import EvalId


@dataclasses.dataclass(frozen=True)
class AsrClip:
    """What a profile reads from a dataset row."""

    eval_id: EvalId
    language: str  # ISO 639-3
    language_name: str  # English name (_asr_common.language_fields)
    language_alpha_2: str | None


Content = list[dict[str, Any]]


def _no_kwargs(_clip: AsrClip) -> dict[str, Any] | None:
    return None


def _as_sent(text: str) -> str:
    return text


@dataclasses.dataclass(frozen=True)
class AsrProfile:
    name: str
    # The public sources this profile follows, at pinned revisions.
    reference: str
    # (clip, the clip's input_audio part) -> chat messages
    messages: Callable[[AsrClip, dict[str, Any]], list[dict[str, Any]]]
    # The model's audio length limit, in 16 kHz samples (None = the whole clip).
    audio_max_samples: int | None = None
    # Per-request chat-template arguments (merged by the client over the
    # profile's server-level ones).
    chat_template_kwargs: Callable[[AsrClip], dict[str, Any] | None] = _no_kwargs
    # The cleanup of the model's answer before scoring.
    postprocess: Callable[[str], str] = _as_sent


def _user(*content: dict[str, Any]) -> dict[str, Any]:
    return {"role": "user", "content": list(content)}


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


# --- Gemma 4 -----------------------------------------------------------------
# google/gemma-4-E2B-it model card, "Best Practices": §6 "Audio" gives the ASR
# prompt with a {LANGUAGE} placeholder, §4 puts audio after the text, §7 limits
# audio to 30 seconds. The placeholder takes the language's English name.
GEMMA4_ASR_PROMPT = (
    "Transcribe the following speech segment in {source_language} into "
    "{source_language} text.\n\n"
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not "
    "one point seven, and write 3 instead of three."
)
# 30 s at 16 kHz: the card's limit, and the max_length=480_000 (truncation on)
# with which transformers' Gemma4AudioFeatureExtractor cuts longer audio.
GEMMA4_MAX_SAMPLES = 480_000


def _gemma4(clip: AsrClip, audio: dict[str, Any]) -> list[dict[str, Any]]:
    return [_user(_text(GEMMA4_ASR_PROMPT.format(source_language=clip.language_name)), audio)]


# --- Qwen2.5-Omni ------------------------------------------------------------
# QwenLM/Qwen2.5-Omni cookbooks/universal_audio_understanding.ipynb: system "You
# are a speech recognition model.", the audio then the prompt "Transcribe the
# English audio into text without any punctuation marks.". The cookbook words
# its other languages' examples slightly differently ("without including any");
# pipette uses the English wording for every language (pipette's extension).
# The order is the cookbook's messages: the audio part, then the text part, as
# the current Qwen2.5-Omni chat template renders them (the outputs recorded in
# the cookbook's cells, from an earlier template, show the text first).
QWEN25_OMNI_SYSTEM = "You are a speech recognition model."
QWEN25_OMNI_ASR_PROMPT = "Transcribe the {source_language} audio into text without any punctuation marks."


def _qwen25_omni(clip: AsrClip, audio: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": QWEN25_OMNI_SYSTEM},
        _user(audio, _text(QWEN25_OMNI_ASR_PROMPT.format(source_language=clip.language_name))),
    ]


def _qwen25_omni_answer(text: str) -> str:
    # The cookbook decodes the whole sequence (batch_decode of generate's
    # output), so its response is the chat transcript ending
    # "assistant\n<answer>"; the answer is the text after the last
    # "\nassistant\n", stripped. A server that returns the generated text only
    # is unaffected.
    return text.split("\nassistant\n")[-1].strip()


# --- Voxtral -----------------------------------------------------------------
# mistralai/Voxtral-Mini-3B-2507 model card, "Transcription": the dedicated
# transcription mode (mistral-common TranscriptionRequest; transformers
# apply_transcription_request), with an optional ISO 639-1 `language` that
# "will improve language adherence" (mistral-common
# protocol/transcription/request.py); without one the model detects the
# language. The hint is given for the languages Mistral reports Voxtral's
# transcription on: FLEURS in the Voxtral technical report (arXiv 2507.13264,
# App. A.1, Table 4: Arabic, Dutch, English, French, German, Hindi, Italian,
# Portuguese, Spanish).
VOXTRAL_EXPLICIT_LANGUAGES = frozenset({"ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt"})


def _voxtral(_clip: AsrClip, audio: dict[str, Any]) -> list[dict[str, Any]]:
    return [_user(audio)]


def _voxtral_kwargs(clip: AsrClip) -> dict[str, Any]:
    alpha_2 = clip.language_alpha_2
    return {
        "transcription": True,
        "language": alpha_2 if alpha_2 in VOXTRAL_EXPLICIT_LANGUAGES else "",
    }


# --- MiniCPM-o 4.5 -----------------------------------------------------------
# OpenBMB's evaluation prompts: OmniEvalKit o_e_Kit/configs/generation_configs.json
# `asr_configs`, each "<prompt>\n{audio}" (the prompt, a newline, then the clip).
MINICPM_GENERAL = "Please listen to the audio carefully and transcribe it with high precision."
MINICPM_LOWERCASE = (
    "Please listen to the audio carefully and transcribe it with high precision. "
    "Please output in lower case without punctuation."
)
MINICPM_SPGISPEECH = "Please listen to the financial domain audio and transcribe it accurately."
# The FLEURS languages other than English. MiniCPM-o picks the language of its
# answer from the language of the instruction: the openbmb/MiniCPM-o-4_5 model
# card (@ d59117d19ddb8a28a62f1b5f4e3b34ac48d510a4, audio understanding) gives
# one English prompt for English ASR and for Chinese-to-English translation
# alike, and the MiniCPM-o 2.6 code (openbmb/MiniCPM-o-2_6 modeling_minicpmo.py
# @ 06849bfd36da94da1f5a88fa17e8ba63f08a4c4d, `get_sys_prompt`) says the model
# can "select the response language based on the question language". So an
# English instruction pulls the answer towards English. OpenBMB writes the
# instruction in the target language: OmniEvalKit's one European-language
# config, `commonvoice_fr`, is "Veuillez écouter attentivement l'audio et le
# transcrire avec précision.". Each prompt here is that instruction in the
# clip's language, then a clause asking for a verbatim transcript in that
# language and no translation. The French first sentence is OmniEvalKit's,
# verbatim; the German, Spanish, Portuguese, Italian and Dutch translations and
# every "do not translate" clause are pipette's extension. The scheme was chosen
# among six candidate prompts on the FLEURS validation split, by a rule fixed
# before any was run; the test split was not used (docs/asr-evals.md, "How
# MiniCPM-o's FLEURS prompts were chosen"). The non-French strings await review
# by native speakers.
MINICPM_FLEURS: dict[EvalId, str] = {
    EvalId.ASR_FLEURS_ES_419: (
        "Por favor, escuche atentamente el audio y transcríbalo con gran precisión. "
        "Transcríbalo palabra por palabra en el idioma original (español); no lo traduzca."
    ),
    EvalId.ASR_FLEURS_PT_BR: (
        "Por favor, ouça o áudio com atenção e transcreva-o com alta precisão. "
        "Transcreva-o palavra por palavra no idioma original (português); não o traduza."
    ),
    EvalId.ASR_FLEURS_DE_DE: (
        "Bitte hören Sie sich die Audioaufnahme aufmerksam an und transkribieren Sie sie mit hoher Präzision. "
        "Transkribieren Sie sie wortgetreu in der Originalsprache (Deutsch); übersetzen Sie sie nicht."
    ),
    EvalId.ASR_FLEURS_FR_FR: (
        "Veuillez écouter attentivement l'audio et le transcrire avec précision. "
        "Transcrivez-le mot pour mot dans la langue d'origine (français) ; ne le traduisez pas."
    ),
    EvalId.ASR_FLEURS_IT_IT: (
        "Per favore, ascolta attentamente l'audio e trascrivilo con alta precisione. "
        "Trascrivilo parola per parola nella lingua originale (italiano); non tradurlo."
    ),
    EvalId.ASR_FLEURS_NL_NL: (
        "Luister aandachtig naar de audio en transcribeer deze met hoge precisie. "
        "Transcribeer het woordelijk in de oorspronkelijke taal (Nederlands); vertaal het niet."
    ),
}
# OmniEvalKit's english configs: LibriSpeech, GigaSpeech and VoxPopuli take the
# lowercase prompt, SPGISpeech the financial one, the others (FLEURS English
# included: its `fleurs_en`, and its default) the general one.
_MINICPM_LOWERCASE_SETS = frozenset(
    {
        EvalId.ASR_ESB_GIGASPEECH_CLEANED,
        EvalId.ASR_ESB_VOXPOPULI_CLEANED_AA,
        EvalId.ASR_ESB_LIBRISPEECH_TEST_CLEAN,
        EvalId.ASR_ESB_LIBRISPEECH_TEST_OTHER,
    }
)


def minicpm_o_prompt(clip: AsrClip) -> str:
    if clip.eval_id == EvalId.ASR_ESB_SPGISPEECH:
        return MINICPM_SPGISPEECH
    if clip.eval_id in _MINICPM_LOWERCASE_SETS:
        return MINICPM_LOWERCASE
    if clip.language == "eng":
        return MINICPM_GENERAL
    try:
        return MINICPM_FLEURS[clip.eval_id]
    except KeyError:
        raise ValueError(f"minicpm_o has no {clip.language_name} prompt for {clip.eval_id.value}") from None


def _minicpm_o(clip: AsrClip, audio: dict[str, Any]) -> list[dict[str, Any]]:
    # A chat-completions server concatenates a turn's content parts, so the
    # newline OmniEvalKit puts between the prompt and {audio} rides on the text.
    return [_user(_text(minicpm_o_prompt(clip) + "\n"), audio)]


def _minicpm_o_answer(text: str) -> str:
    # openbmb/MiniCPM-o-4_5 modeling_minicpmo.py: the spoken answer is the text
    # after the last <|tts_bos|> and before <|tts_eos|> (`_generate_mel_spec`),
    # and chat() cuts its answer at <|tts_eos|>. The TTS chat template
    # (use_tts_template) opens the assistant turn with <|tts_bos|>, which a
    # server may return with the answer; absent, the split is a no-op.
    return text.split("<|tts_bos|>")[-1].split("<|tts_eos|>")[0].strip()


# --- LFM2.5-Audio and LFM3 ---------------------------------------------------
# LiquidAI/LFM2.5-Audio-1.5B: the card points to the liquid-audio package for
# ASR, whose README ("ASR") uses sequential generation with "the fixed system
# prompt `Perform ASR.`" and the audio as the user turn. The card lists English
# as the supported language and gives no other prompt, so the profile sends
# that fixed prompt whatever the clip's language (a FLEURS cell in another
# language runs, and is marked as outside the card's declared languages).
LFM25_AUDIO_ASR_SYSTEM = "Perform ASR."
# LFM3's native ASR chat format (no public card yet).
LFM_ASR_SYSTEM = "Perform ASR in {source_language}."


def _lfm25_audio(_clip: AsrClip, audio: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "system", "content": LFM25_AUDIO_ASR_SYSTEM}, _user(audio)]


def _lfm3(clip: AsrClip, audio: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "system", "content": LFM_ASR_SYSTEM.format(source_language=clip.language_name)}, _user(audio)]


def _lfm25_audio_answer(text: str) -> str:
    # The end-of-turn token, should a server return it with the text.
    return text.removesuffix("<|im_end|>")


PROFILES: dict[str, AsrProfile] = {
    p.name: p
    for p in (
        AsrProfile(
            "gemma4",
            "google/gemma-4-E2B-it model card @ 3e22461f65e89153144f8adb70e3b8c2cc9845a7, Best Practices "
            "§4 (audio after text), §6 (ASR prompt), §7 (30 s audio)",
            _gemma4,
            audio_max_samples=GEMMA4_MAX_SAMPLES,
        ),
        AsrProfile(
            "qwen25_omni",
            "QwenLM/Qwen2.5-Omni @ d8a31ca56c0456b6edfcbcbf4bdbb6ae2200ef42 "
            "cookbooks/universal_audio_understanding.ipynb (speech recognition)",
            _qwen25_omni,
            postprocess=_qwen25_omni_answer,
        ),
        AsrProfile(
            "voxtral",
            "mistralai/Voxtral-Mini-3B-2507 model card @ 3060fe34b35ba5d44202ce9ff3c097642914f8f3 "
            "(Transcription); mistral-common @ b54161726d8cde801c976a2bad48c23777551927 "
            "protocol/transcription/request.py; Voxtral technical report arXiv 2507.13264 App. A.1 Table 4",
            _voxtral,
            chat_template_kwargs=_voxtral_kwargs,
        ),
        AsrProfile(
            "minicpm_o",
            "OpenBMB/OmniEvalKit @ 1adac0577258d539efc03f16d8a0f4b3f7df6c19 "
            "o_e_Kit/configs/generation_configs.json (asr_configs); openbmb/MiniCPM-o-4_5 "
            "@ d59117d19ddb8a28a62f1b5f4e3b34ac48d510a4 model card (audio understanding) and "
            "modeling_minicpmo.py (chat, _generate_mel_spec); openbmb/MiniCPM-o-2_6 "
            "@ 06849bfd36da94da1f5a88fa17e8ba63f08a4c4d modeling_minicpmo.py (get_sys_prompt)",
            _minicpm_o,
            postprocess=_minicpm_o_answer,
        ),
        AsrProfile(
            "lfm25_audio",
            "LiquidAI/LFM2.5-Audio-1.5B model card @ c362a0625dfe45aa588dce5f0ada28a7e5707628; "
            "Liquid4All/liquid-audio @ 19e65845923a7f136442c95137884ec61eb386aa README (ASR)",
            _lfm25_audio,
            postprocess=_lfm25_audio_answer,
        ),
        AsrProfile(
            "lfm3",
            "LFM3's native ASR chat format",
            _lfm3,
        ),
    )
}


def profile(name: str) -> AsrProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"Unknown ASR prompt profile: {name!r} (known: {', '.join(PROFILES)})") from None
