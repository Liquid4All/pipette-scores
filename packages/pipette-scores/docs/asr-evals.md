# ASR evals: the Open ASR Leaderboard (English) and FLEURS

The ASR evals score speech-recognition quality with the two public protocols of
the Open ASR Leaderboard (huggingface/open_asr_leaderboard; Srivastav et al.,
"Open ASR Leaderboard", arXiv 2510.06961):

- **English**: its nine public English test sets, its English normalizer, its
  row filter, its long-form merge and its kaldialign word error rate.
- **FLEURS** (google/fleurs; Conneau et al., arXiv 2205.12446): the
  leaderboard's multilingual scoring (normalizer, number handling, alignment)
  applied to FLEURS's own `transcription` references. Its multilingual
  normalizer spells every number out in the clip's language on both sides, so a
  transcript is not charged for writing "twenty" where the reference has "20"
  (or the reverse); then its compound-pair step and its kaldialign word error
  rate. The leaderboard's published FLEURS rows score against its own cleaned
  copy of the references, so its numbers can differ slightly from pipette's.

Each model is asked as its own public usage guidance says (§2). Where pipette
departs from either protocol, §4 lists it with the reason. Every source below
is public and pinned. The leaderboard code is cited at
huggingface/open_asr_leaderboard @ `62a457c5c32d0de4df77e9470308aae7d45b055f`,
its Space (hf-audio/open_asr_leaderboard) at `960d57b6cf65ea8dbb4c628ea87e60458d75018d`
for the English average and at `9b2dcd1f6318e601432eeb73a2563848fc0a20b3` for
the multilingual tab (`init.py` version "21-10-2026", whose FLEURS columns are
`hf-audio/multilingual_evals` @ `3589b3c077cb16c935e52bdd15c7fb0a3f175f7a`).

## 1. Datasets

One eval per dataset, 16 in all (9 English sets, 7 FLEURS locales), version
`2026.10.1`. Sources, pins, clip
counts, licenses and fingerprints are in
`pipette_scores/dataset_catalog/asr_datasets.py`.

**The leaderboard's English sets.** The Space's English average, as of its
version "09-10-2026: Add URGENT24 to avg English WER" (`init.py`
`default_datasets`), is taken over nine public sets and two private aggregates
that no third party can run. pipette evaluates the nine public sets, in the
leaderboard's column order (`normalizer/eval_utils.score_results`, the
`DATASET_CONFIGS` of its job scripts). The EvalIds keep the `asr_esb_` prefix of
the leaderboard's lineage (the ESB benchmark, arXiv 2210.13352, whose dataset
repo this was) so that consumers' ids stay stable; the method is the
leaderboard's.

| EvalId | Source @ revision | Clips |
|---|---|---|
| `asr_esb_ami_cleaned` | `hf-audio/open-asr-leaderboard` `ami_cleaned`/test @ `de79e57a` | 7,805 (7,715 scored after the row filter) |
| `asr_esb_earnings22_cleaned_aa_chunked` | `ArtificialAnalysis/Earnings22-Cleaned-AA-chunked` @ `27f7d005` (audio, chunk manifest) + `ArtificialAnalysis/Earnings22-Cleaned-AA` @ `5999936e` (call transcripts) | 341 chunks of 6 calls |
| `asr_esb_gigaspeech_cleaned` | as AMI, `gigaspeech_cleaned`/test | 18,768 (18,757 scored) |
| `asr_esb_librispeech_test_clean` | as AMI, `librispeech`/test.clean | 2,620 |
| `asr_esb_librispeech_test_other` | as AMI, `librispeech`/test.other | 2,939 |
| `asr_esb_spgispeech` | as AMI, `spgispeech`/test | 39,341 |
| `asr_esb_urgent2024` | as AMI, `urgent2024`/test (the noisy URGENT 2024 set; `urgent2024_clean` is outside the leaderboard average and not evaluated) | 1,000 |
| `asr_esb_monsoon_en_in` | `VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN` @ `bc1da7b4` (the leaderboard's `VoiceArena/Monsoon_en_IN_test` redirects to it) | 2,102 |
| `asr_esb_voxpopuli_cleaned_aa` | as AMI, `voxpopuli_cleaned_aa`/test | 628 |

`hf-audio/open-asr-leaderboard` is pinned at `de79e57adf9983993bcc72586d7313460825f5eb`
("Add urgent2024 (#13)"); its other configs are byte-identical to those at the
earlier `b6bdcd0b` (that diff touches only `README.md` and adds the two URGENT
directories), so the fingerprints recorded there hold.

**FLEURS.** `google/fleurs` `<locale>`/test @ `70bb2e84b976b7e960aa89f1c648e09c59f894dd`
(main since 2026-05-15), 7 locales: `en_us`, `es_419`, `pt_br`, `de_de`,
`fr_fr`, `it_it`, `nl_nl` (`FLEURS_LOCALES`, 364 to 919 clips each). These are
English and the languages the leaderboard runs its multilingual FLEURS
evaluation on (`api/run_api_ml.sh` `DATASET_CONFIGS`: German, French, Italian,
Spanish, Portuguese, Dutch and Armenian), without Armenian (§4). FLEURS ships
Spanish and Portuguese as `es_419` and `pt_br` only.

The **references** are FLEURS's own `transcription` column (Google's
normalized transcript, as the dataset ships it). The leaderboard scores its
published FLEURS rows against a copy of FLEURS whose references it cleaned
itself, starting from FLEURS's `raw_transcription`; that preparation is not
published, so pipette scores against FLEURS's `transcription`, with the same
scoring code. The two reference texts normalize almost alike, so a model's
numbers can differ slightly from what the leaderboard would report for the same
transcripts.

**Licenses** (as each dataset card, or the leaderboard's About table, states
them). Every set is for internal evaluation only here: pipette uses the sets
only for internal, non-commercial benchmarking of its models, keeps them
org-private, and never redistributes the built data
(the repository's `docs/dataset-licenses.md`). SPGISpeech and URGENT24 are licensed for
non-commercial use only, and that is their only use here:

| Set | License |
|---|---|
| AMI | CC-BY-4.0 |
| Earnings22-Cleaned-AA(-chunked) | Apache-2.0 (Artificial Analysis cards; Earnings-22 itself CC-BY-SA-4.0) |
| GigaSpeech | Apache-2.0, access gated behind the SpeechColab terms |
| LibriSpeech | CC-BY-4.0 |
| SPGISpeech | Kensho user agreement, **non-commercial** |
| URGENT24 | mixed, **non-commercial**: its speech is CC0 / CC-BY-4.0 / ODC-BY / Creative Commons, but the overlaid noise includes WHAM! (CC-BY-NC-4.0) and TUT Urban Acoustic Scenes 2018 (non-commercial, research only) |
| Voice Arena Monsoon | CC-BY-4.0 |
| VoxPopuli-Cleaned-AA | Apache-2.0 (Artificial Analysis card; VoxPopuli itself CC0) |
| FLEURS | CC-BY-4.0 |

**Built** by `scripts/build_asr_datasets.py` into
`<out>/<eval_id>/2026.10.1/{test.parquet, metadata.json}`, reading each source as
the leaderboard loads it (`normalizer/data_utils.load_data`; for Earnings22
`load_chunked_data`: the chunk manifest and WAVs, each chunk carrying its call's
`transcript` as read by `get_text`, its `parent_id` and `chunk_index`), at the
catalog's revision and in the split's row order. A clip's position, `doc_id`, is
its row in the source split, and the sample id is `sha256("<eval_id>|<doc_id>")[:12]`.
References: the leaderboard sets' `text` (Earnings22: its call's `transcript`);
FLEURS `transcription`; all stripped. The leaderboard's row filter
is applied when scoring (§3). The build streams rows to the parquet one row
group (64 clips) at a time.

The builder **refuses an output inside this repository's checkout**, and
`datasets/asr_*/` is in `.gitignore` and `.dockerignore` as a backstop.

**Audio is stored byte for byte** as the source ships it when it is already
16 kHz mono; any other clip is resampled once by the builder and stored as a
16 kHz mono WAV (below). Each clip's format, sample rate, channels and length
are recorded, and the source's rate and channels go in `raw_doc`. FLEURS and VoxPopuli ship 16 kHz 32-bit float WAV,
Earnings22 16 kHz 16-bit WAV, LibriSpeech 16 kHz 16-bit FLAC, MonsoonASR
**44.1 kHz** 16-bit WAV, and URGENT24 16-bit WAV at **mixed rates** (16, 22.05,
24, 32, 44.1 and 48 kHz). Some clips run past 30 s (VoxPopuli 3 of 628, up to
38.4 s; LibriSpeech test.clean 9 of 2,620, up to 35.0 s), which is where Gemma
4's 30 s limit applies.

**Integrity.** `metadata.json` holds the source pins, the scoring recipe, an
audio summary, the license and `content_sha256`: the audio-inclusive fingerprint
(the same meaning as the VLM sets' field), a digest of every column with the
audio through each row's `audio_sha256`. The server recomputes it at load
without reading audio, and re-hashes every clip it serves. Before a dataset is
scored or served the server also checks that row `i` is `doc_id` `i` with that
position's sample id (so a sample's id says where it stands in the stream),
that every clip is 16 kHz mono, that the scoring recipe, the language and, for
`2026.10.1`, the clip count and source fingerprint are the catalog's, and that
it is no `--limit` smoke build. A full
build of `2026.10.1` must match the catalog before its file lands.
`scripts/verify_asr_datasets.py` runs all of it offline, plus every clip's hash.

**Split sets.** Some storage backends cap the size of one object (git-LFS: 2 GB
on most GitHub plans), and a set can be larger (SPGISpeech is 9.6 GB). `scripts/split_asr_dataset.py`
cuts a `test.parquet` over 1.9 GB into ordered parts `test-00000-of-0000N.parquet`,
…, between row groups, with `metadata.json` copied unchanged. The server reads
the parts as one table (`_asr_common.parquet_files`) and refuses both layouts at
once, a missing or misnumbered part, or a stray `test-*.parquet`. The
fingerprint is a digest of the rows, not the files, so the parts carry the
build's (`tests/test_asr_parts.py`).

**The catalog pins the input, not the output.** A clip already at 16 kHz mono
is stored as shipped; any other (MonsoonASR, most of URGENT24) is resampled
once, by the builder, and stored as the 16 kHz mono WAV serving would write,
with its source bytes' sha256, format, rate, channels and length in the row's
`raw_doc`. Serving resamples nothing. soxr rounds per CPU architecture (§4), so
the stored bytes of those two sets, and their `content_sha256`, depend on the
machine that built them. The catalog's pin is therefore `source_sha256`
(`metadata.json` records it too): the same digest over the rows as the source
shipped them, a resampled clip counted by its source fields. Any machine
reproduces it, and for the other 14 sets it equals `content_sha256`. Pins are
recorded for all 16 sets, and both the builder and the server enforce them;
`content_sha256` keeps guarding the stored files against `metadata.json`.

## 2. Requests: the prompt profiles

An ASR dataset holds no prompt. A **prompt profile**
(`dataset_catalog/asr_prompt_profiles.py`) turns each clip into the request the
model's own public guidance describes: the turns, the text, the template
arguments, the model's audio limit, and the cleanup of its answer. Where the
source gives no wording for a language, the profile extends it, labelled
"pipette's extension".

**Audio, the same for every profile.** Each clip is brought to 16 kHz mono once,
as the leaderboard prepares audio for every model it runs:
`data_utils.prepare_data` casts the audio column to
`datasets.Audio(sampling_rate=16000)`. How that cast decodes depends on the
installed `datasets`: up to 3.x soundfile and librosa, from 4.0 torchcodec
(FFmpeg). The leaderboard pins it in one place, its API-provider image
(`api/Dockerfile`: `datasets==2.19.0`, "to avoid torchcodec dependency"); its GPU
model jobs install `datasets` unpinned (`requirements/requirements_jobs.txt`), so
on a current install they decode through torchcodec. pipette follows the API
image's path, the leaderboard's only pinned one, which is reproducible from
pinned Python packages without FFmpeg. The choice matters only for the clips
that are resampled, MonsoonASR's (44.1 kHz) and URGENT24's non-16 kHz ones; every
other set ships 16 kHz audio, which both paths pass through unchanged (§4).
The path: `datasets` 2.19.0 `Audio.decode_example` reads the
clip with soundfile (float64), averages the channels (`librosa.to_mono`) and
calls `librosa.resample(array, orig_sr=sr, target_sr=16000)`, whose default
`soxr_hq` runs soxr at high quality and then pads or cuts the result to
`ceil(n * 16000 / sr)` samples. `_asr_audio.resample_16k_mono` is that sequence
(with `soundfile==0.14.0` and `soxr==1.1.0`, without depending on librosa), and
matches `datasets` 2.19.0 sample for sample (§5). The clip is then cut to the
model's audio limit where its card states one (Gemma 4: 30 s) and served as a
WAV that carries those samples exactly: **16-bit PCM** when every sample is a
16-bit value (k / 32768 for an integer k), **32-bit float** otherwise. The
leaderboard's `datasets` path decodes every clip to float, and the float
samples are what must reach the model unchanged; a 16 kHz mono 16-bit source
(LibriSpeech, Earnings22, AMI, …) decodes to 16-bit values, and so do some float
sources (VoxPopuli's), so for those clips a 16-bit WAV is the same samples at
half the bytes: soundfile reads it as k / 32768, llama.cpp's audio loader
(miniaudio's `ma_pcm_s16_to_f32`) as k × 0.000030517578125, the same float32.
A resampled clip (MonsoonASR, most of URGENT24) or a float source with finer
values (FLEURS) stays 32-bit float. pipette writes either WAV itself (PCM:
`fmt `, `data`; float: `fmt `, `fact`, `data`, without the timestamped PEAK
chunk libsndfile's float WAVs add), so a clip's served bytes are the same on
every call.

| Profile | Public source | Turns | Answer cleaned |
|---|---|---|---|
| `gemma4` | google/gemma-4-E2B-it model card @ `3e22461f65e89153144f8adb70e3b8c2cc9845a7`, "Best Practices" §4 (audio after text), §6 (ASR prompt), §7 (audio up to 30 s) | user: "Transcribe the following speech segment in {language} into {language} text.\n\nFollow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.", then the clip, cut at 30 s (480,000 samples; transformers' `Gemma4AudioFeatureExtractor` truncates at the same `max_length`) | as is |
| `qwen25_omni` | QwenLM/Qwen2.5-Omni @ `d8a31ca56c0456b6edfcbcbf4bdbb6ae2200ef42` `cookbooks/universal_audio_understanding.ipynb` | system "You are a speech recognition model."; user: the clip, then "Transcribe the {language} audio into text without any punctuation marks." (the cookbook's message order, audio then text) | the text after the last `\nassistant\n` (the cookbook decodes prompt and answer together), stripped |
| `voxtral` | mistralai/Voxtral-Mini-3B-2507 model card @ `3060fe34b35ba5d44202ce9ff3c097642914f8f3` ("Transcription"); mistral-common @ `b54161726d8cde801c976a2bad48c23777551927` `protocol/transcription/request.py`; Voxtral technical report, arXiv 2507.13264, App. A.1 Table 4 | user: the clip alone, in the dedicated transcription mode: `chat_template_kwargs` `{"transcription": true, "language": <code or "">}` | as is |
| `minicpm_o` | OpenBMB/OmniEvalKit @ `1adac0577258d539efc03f16d8a0f4b3f7df6c19` `o_e_Kit/configs/generation_configs.json` (`asr_configs`); openbmb/MiniCPM-o-4_5 @ `d59117d19ddb8a28a62f1b5f4e3b34ac48d510a4` model card (audio understanding) and `modeling_minicpmo.py`; openbmb/MiniCPM-o-2_6 @ `06849bfd36da94da1f5a88fa17e8ba63f08a4c4d` `modeling_minicpmo.py` (`get_sys_prompt`) | user: OpenBMB's evaluation prompt for the dataset, in the clip's language (below), and the newline of its `"<prompt>\n{audio}"`, then the clip | the text after the last `<|tts_bos|>` and before `<|tts_eos|>` (`_generate_mel_spec`; `chat()` cuts at `<|tts_eos|>`), stripped |
| `lfm25_audio` | LiquidAI/LFM2.5-Audio-1.5B model card @ `c362a0625dfe45aa588dce5f0ada28a7e5707628`, which points to Liquid4All/liquid-audio @ `19e65845923a7f136442c95137884ec61eb386aa` README "ASR" | system "Perform ASR." (the README's fixed ASR prompt), whatever the clip; user: the clip. The card lists English as the supported language and gives no prompt for another; LFM2.5-Audio runs every dataset, and a cell in another language is marked as outside the card's declared languages (pipette-mgmt's catalog entries carry the card's languages) | a trailing `<|im_end|>` dropped |
| `lfm3` | LFM3's native ASR chat format: a placeholder for an unreleased model (no public card yet), to be replaced from its card before release | system "Perform ASR in {language}."; user: the clip | as is |

`{language}` is the language's English name, pycountry's ISO 639-3 name,
fixed at build time.

**Qwen2.5-Omni**: the cookbook's English prompt is used verbatim; its other
languages' examples are worded slightly differently ("without including any
punctuation marks"; a Chinese-language prompt for Chinese), and pipette uses the
English wording for every language (pipette's extension). The user turn follows
the cookbook's message order, the audio part then the text part, which the
current Qwen2.5-Omni chat template renders in that order. (The outputs recorded
in the cookbook's cells show the text before the audio, as an earlier version of
the template rendered the same messages; pipette follows the messages.)

**Voxtral's language hint** is given for the nine languages Mistral reports
Voxtral's transcription on, FLEURS in the technical report (Arabic, Dutch,
English, French, German, Hindi, Italian, Portuguese, Spanish), which include
every language evaluated here, and left empty, so that the model detects the
language (card: "By default, Voxtral automatically predicts the source audio
language"), for any other. mistral-common's
`language` is an optional ISO 639-1 code that "will improve language adherence".

**MiniCPM-o's prompts** (OmniEvalKit `asr_configs`): SPGISpeech "Please listen
to the financial domain audio and transcribe it accurately."; GigaSpeech,
VoxPopuli and LibriSpeech "Please listen to the audio carefully and transcribe it
with high precision. Please output in lower case without punctuation."; FLEURS
English and the English sets OmniEvalKit does not configure (AMI, Earnings22,
Monsoon, URGENT24) the general "Please listen to the audio carefully and
transcribe it with high precision." (its `fleurs_en`, and its default),
unchanged.

The other FLEURS languages, which OmniEvalKit does not configure, are asked
**in their own language**. MiniCPM-o picks the language of its answer from the
language of the instruction: the MiniCPM-o 4.5 model card (@ `d59117d1`,
"audio understanding") gives a single English prompt for English ASR and for
Chinese-to-English translation alike, and the MiniCPM-o 2.6 code (@ `06849bfd`,
`get_sys_prompt`) says the model can "select the response language based on the
question language". An English instruction naming the language therefore draws
English translations instead of transcripts. OpenBMB itself writes the
instruction in the target language: the one European language OmniEvalKit
configures, French (`commonvoice_fr`), is asked "Veuillez écouter attentivement
l'audio et le transcrire avec précision.". Each prompt is that instruction in
the clip's language, then a clause asking for a verbatim transcript in the
original language, without translation:

| Locale | Prompt |
|---|---|
| `es_419` | Por favor, escuche atentamente el audio y transcríbalo con gran precisión. Transcríbalo palabra por palabra en el idioma original (español); no lo traduzca. |
| `pt_br` | Por favor, ouça o áudio com atenção e transcreva-o com alta precisão. Transcreva-o palavra por palavra no idioma original (português); não o traduza. |
| `de_de` | Bitte hören Sie sich die Audioaufnahme aufmerksam an und transkribieren Sie sie mit hoher Präzision. Transkribieren Sie sie wortgetreu in der Originalsprache (Deutsch); übersetzen Sie sie nicht. |
| `fr_fr` | Veuillez écouter attentivement l'audio et le transcrire avec précision. Transcrivez-le mot pour mot dans la langue d'origine (français) ; ne le traduisez pas. |
| `it_it` | Per favore, ascolta attentamente l'audio e trascrivilo con alta precisione. Trascrivilo parola per parola nella lingua originale (italiano); non tradurlo. |
| `nl_nl` | Luister aandachtig naar de audio en transcribeer deze met hoge precisie. Transcribeer het woordelijk in de oorspronkelijke taal (Nederlands); vertaal het niet. |

The French first sentence is OmniEvalKit's, verbatim. The German, Spanish,
Portuguese, Italian and Dutch translations of it, and the "do not translate"
clause in every language, are pipette's extension, chosen on the FLEURS
validation split as below. English keeps OpenBMB's prompt without the clause.
**The non-French strings await review by native speakers.**

**How MiniCPM-o's FLEURS prompts were chosen.** OpenBMB publishes no prompt
for these six languages, so pipette had to choose one, and chose it on the
FLEURS **validation** split (`google/fleurs` @ `70bb2e84`, `validation`, every
clip of the six locales), never on the test split the entries score. The
selection rule was written down before any candidate was run:

1. The candidates, each the user turn before the clip, the same in every run:
   - the earlier English extension, "Please listen to the audio snippet
     carefully and transcribe the content in {language}.";
   - the model card's English usage prompt, "Please listen to the audio
     snippet carefully and transcribe the content.";
   - OmniEvalKit's general English prompt, "Please listen to the audio
     carefully and transcribe it with high precision.";
   - that English prompt followed by "Transcribe it verbatim in the original
     language ({language}); do not translate it.";
   - that English prompt rendered in the clip's language (for French,
     OmniEvalKit's `commonvoice_fr` string);
   - the rendered prompt followed by the same clause in the clip's language
     (the table above).
2. Each candidate ran on every validation clip as the catalog entries run
   MiniCPM-o: `openbmb/MiniCPM-o-4_5-gguf` @ `db25077c` (Q8_0 weights, OpenBMB's
   audio GGUF), pipette's ASR build of llama.cpp, three beams, repetition
   penalty 1.2, 128 tokens, thinking off. Every clip was answered.
3. Each was scored with the then-current selection protocol (corpus word error
   rate per locale against google/fleurs' `transcription`), and its answers
   were language-identified with fastText's `lid.176` model.
4. The rule: a candidate is eligible only if at most 2% of its answers, pooled
   over the six locales, are identified as a language other than the clip's;
   the eligible candidate with the lowest mean word error rate over the six
   locales wins.

Only the two candidates carrying the no-translate clause passed the language
filter; the native-language one had the lower mean error rate, and is the
prompt above. Re-scored with the FLEURS protocol of §3 (the leaderboard's
multilingual scoring against `transcription`), the same rule picks the same
prompt: the same two candidates are eligible and the
native-language one has the lower mean error rate. The test split is run once, by the catalog entries, with the
chosen prompt frozen. OpenBMB states that MiniCPM-o's speech modes support only
Chinese and English (MiniCPM-V issues #1056 and #1147), so every non-English
FLEURS result is outside its supported languages.
MiniCPM-o's TTS chat template (`use_tts_template`) opens the assistant turn with
`<|tts_bos|>`, which a server may return with the answer; the cleanup follows
OpenBMB's own split, and is a no-op when it is absent.

### Server prerequisites

What a source sets on the model or server, rather than in the request, pipette's
client must provide:

- **Thinking off, for every family.** The client starts every ASR server with
  reasoning off, so no profile sends `enable_thinking`; `/score` refuses
  `enable_thinking: true` for an ASR eval.
- **Voxtral's transcription mode.** The `transcription` / `language` template
  arguments take effect only under a chat template that renders mistral-common's
  transcription request (`tokens/tokenizers/instruct.py`
  `_encode_instruct_transcription`: the user turn holding the audio, then
  `lang:<code>` when a language is given, then `[TRANSCRIBE]`); a template
  without that mode ignores them, and the model then chats instead of
  transcribing.
- **LFM3's turn format**: its native ASR chat format, an open assistant turn
  with no think block.
- **MiniCPM-o's template switches**: `use_tts_template: true` and
  `enable_thinking: false`, on the server and every request. The prompt-clip
  newline rides in the profile's text, since a server concatenates content parts.
- **Template arguments merge.** The client sends the profile's server-level
  template arguments merged with each sample's `chat_template_kwargs` (the
  sample's win; thinking stays off).

**Generation settings are the benchmark definition's**, not the profile's. The
public recommendations:

| Family | Public recommendation |
|---|---|
| Gemma 4 E2B | card §1: temperature 1.0, top-p 0.95, top-k 64 "across all use cases"; its examples generate up to 512 new tokens; thinking off |
| Qwen2.5-Omni-3B | cookbook: `thinker_do_sample=False` (greedy), `thinker_max_new_tokens=256` |
| Voxtral Mini 3B | card: temperature 0.0 for transcription, `max_new_tokens=500` |
| MiniCPM-o 4.5 | OmniEvalKit `asr_configs`: `num_beams` 3, `sampling` false, `max_tokens` 128; `modeling_minicpmo.py` `prepare_generation_config`: repetition penalty 1.2 with beams, length penalty 1.0 |
| LFM2.5-Audio 1.5B | liquid-audio README: `generate_sequential` with no text temperature (greedy text), `max_new_tokens=512`, its general default; pipette's entries cap a transcript at 256 (on the English sets 512 only let rare repetition loops run longer) |
| LFM3 | greedy (pipette's own) |

## 3. Metrics

Per dataset, from its `metadata.json` `scoring` recipe (checked against the
catalog at load; `scoring/asr/`). The normalizer runs on both the reference and
the hypothesis:

| Datasets | Normalizer | Alignment | Unit |
|---|---|---|---|
| Leaderboard English (9) | the leaderboard's `EnglishTextNormalizer()` (`normalizer/normalizer.py`, vendored byte for byte at `67e8bd6a`, unchanged at `62a457c5`) | `kaldialign.edit_distance(ref.split(), hyp.split(), merge_compounds=True)`, kaldialign 0.12.0 (`eval_utils._align_pair`; the version `requirements_jobs.txt` pins) | word |
| FLEURS (7) | the leaderboard's multilingual scoring normalizer for the clip's language (`eval_utils._normalize(text, <code>)`): for `es`, `pt`, `de`, `fr`, `it`, `nl` its `ml_normalizer`, `MultilingualNormalizer(remove_diacritics=False)` called with the language (`normalizer/data_utils.py`); for `en_us` its English normalizer, as `_normalize` does for `en` | the leaderboard's multilingual alignment (`_align_pair` with `multilingual=True`): `eval_utils.normalize_compound_pairs`, then the same kaldialign alignment | word |

The FLEURS recipe is what the leaderboard's `score_results(multilingual=True,
language=<code>)` (and `scripts/score_bucket_results.py --multilingual`) applies
to its FLEURS runs, the code its multilingual tab is scored with. The
multilingual normalizer lowercases, drops bracketed and parenthesized spans,
turns symbols and punctuation into spaces while keeping diacritics and
combining marks, and collapses whitespace; then it joins space-separated
thousands ("10 000" to "10000") and spells every remaining digit run out with
num2words in the language (`es`, `pt`, `de`, `fr`, `it`, `nl`: "20" becomes
"veinte", "vinte", "zwanzig", "vingt", "venti", "twintig"), on the reference and
the hypothesis alike. A model that writes a number in words is therefore not
charged for it, nor one that writes digits. (The English normalizer goes the
other way, number words to digits, with the same effect.) Its
`normalize_compound_pairs` gives both sides the joined form where a mismatched
stretch is the same characters without its spaces ("zum beispiel" against
"zumbei spiel"). The leaderboard's language codes are its two-letter ones
(`ML_LANG_DATASETS`); its Armenian normalizer is not vendored, and a language
not evaluated here is refused.

Three properties of that code, reproduced as they are: `pt` is num2words'
European Portuguese ("dezasseis", where Brazilian Portuguese writes
"dezesseis"); num2words runs after the lowercase and punctuation passes, so its
own capitals and hyphens stay (German "eine Million", French "vingt-cinq"),
and a hypothesis that writes such a number word itself (lowercased, or split at
the hyphen) is still charged for it; and a parenthesized span is dropped from
both sides, words and all.

The vendored code is `scoring/asr/open_asr_normalizer/multilingual.py`: the
leaderboard's `FILLER_WORDS` and `MultilingualNormalizer` (`data_utils.py`) and
`normalize_compound_pairs` (`eval_utils.py`) at `62a457c5`, unchanged, with
num2words pinned at 0.5.14, the latest release, which the leaderboard's
unpinned `num2words` requirement installs (`scoring/asr/NOTICE.md`). Every
dataset is scored by word error rate.

The rate is the **corpus** rate: summed substitutions, deletions and insertions
over summed reference units (the leaderboard's `align_manifests`, for both
protocols), never the mean of per-sample rates. On the leaderboard's English
sets a sample whose normalized reference is empty or reads "ignore time segment
in scoring" is not scored (`data_utils.is_target_text_in_range`, which
`prepare_data` applies); its multilingual runs drop a reference that is empty or
the ignore marker before normalization (`run_eval_ml.py`), and no FLEURS
reference is either. Earnings22 is scored per call
(`eval_utils.merge_chunked_manifest`): the chunks' stripped answers, empty ones
skipped, joined with one space in `chunk_index` order, against the call's
transcript.

**A result is an answer to every sample, or no result**: `/score` scores an ASR
request only when every sample of the dataset, excluded ones included, has
exactly one completion and none failed (`stop_reason: "failure"`). A request
that leaves a sample unanswered or carries a failed row is refused (400, naming
the unanswered and failed ids), never scored with the gap as an empty
transcript; the scorer refuses it too, for a direct caller. A truncated or
doom-looped generation is an answer and is scored; `stop_reasons` counts them.
A dataset with no reference unit at all has no rate (`error_rate` null).

`/score`'s `context`:

| Key | Meaning |
|---|---|
| `error_unit` | `word` (every dataset; the consumer checks it) |
| `error_rate` | the corpus rate as a ratio; null only when the dataset has no reference unit |
| `result` | `error_rate` × 100 (percent, the leaderboard's unit); null with it |
| `hits`, `substitutions`, `deletions`, `insertions`, `reference_units` | summed over the scored units |
| `units_scored` | scoring units (samples, or Earnings22 calls) |
| `samples_total`, `samples_excluded` | dataset samples; removed by the leaderboard's row filter |
| `stop_reasons` | completions per `stop_reason` (`"unreported"` when none was sent) |
| `prompt_profile`, `scoring` | the profile and the dataset's recipe |

`error_rate` and `result` are unrounded; the leaderboard shows
`round(100 * wer, 2)`. There are no `<eval_id>_blanked_truncated` /
`_pass_at_1_blanked` keys: an error rate has no pass@1.

Each scored sample carries `measures`, **sum-safe**: a unit's counts and
normalized texts are on one sample, so summing a count over the samples gives
the context's total:

| Sample | `measures` |
|---|---|
| scored, owning its unit (an utterance; an Earnings22 call's first chunk) | `unit`, `hits`, `substitutions`, `deletions`, `insertions`, `reference_units`, normalized `reference` and `hypothesis` |
| another chunk of an Earnings22 call | `{"unit": <call>}` |
| removed by the leaderboard's row filter | `{"excluded": true}` |

`is_correct` is "its unit was scored with no error" and false otherwise
(excluded samples included); it is no ASR metric, so never compute an accuracy
from it. Each scored sample's `messages` are the profile's prompt, text only:
each audio part stays in its place as `{"type": "input_audio", "input_audio":
{"data": "", "format": "wav"}}`, so the turn order is auditable, while the clip
itself is the caller's (it was sent with the samples). Each scored sample also
carries the `chat_template_kwargs` it was served with (Voxtral's language hint).

### Suite headlines

A suite headline is computed by the consumer (pipette-mgmt) from the per-dataset
results, as no cross-dataset VLM aggregate is computed here either;
`scoring/asr/suite.py` is the reference definition (`open_asr_mean`,
`fleurs_mean`), tested in `tests/test_asr_suite.py`:

- **Open ASR Leaderboard, English**: the equal-weight mean of the 9 public sets'
  `error_rate` (a macro mean; never summed edits across sets, never weighted by
  size), × 100 for percent, as the leaderboard averages its public columns.
  The leaderboard averages per-set WERs it has already rounded to two
  decimals: its public CSV `avg` is that mean, unrounded, and its console
  "Composite Results" line shows it to two decimals. `leaderboard_result` is
  the console figure: each set's percent WER rounded to two decimals, averaged,
  the mean rounded to two decimals. It is within 0.01 of `result` (each
  rounding contributes up to 0.005).
- **FLEURS**: the equal-weight mean of the 7 locales' `error_rate` (WER), the
  same macro mean, × 100 for percent.

Each needs every one of its datasets exactly once, each with a corpus rate and
scored in words; otherwise there is no headline.

## 4. Where pipette departs from the protocols, and why

**Open ASR Leaderboard (English):**

1. **The average is the public-set mean.** The Space's displayed English
   average also includes two private aggregates ("Private (scripted)", "Private
   (conversational)"), which cannot be reproduced; pipette's headline is the
   mean over the 9 public sets, as `score_results` averages its public columns
   (it averages the per-set WERs rounded to two decimals; pipette's `result`
   averages the unrounded rates, see item 6).
2. **Pinned revisions.** The leaderboard loads its datasets unpinned; pipette
   pins each, so that a number names its data.
3. **Prompts and decoding are each model's own.** The leaderboard runs Gemma 4
   with a prompt of its own ("Write out, word for word, exactly what is said in
   this audio. …") and greedy decoding (256 new tokens), and Voxtral greedy with
   `language="en"` (500 new tokens); it does not run Qwen2.5-Omni, MiniCPM-o or
   LFM2.5-Audio. pipette runs every model with its own published
   recommendation (§2): for Gemma 4 that is Google's ASR prompt and sampling at
   temperature 1.0, top-p 0.95, top-k 64. The card names no seed; pipette's
   entries fix the sampler seed (`parameter_seed = 0`), pipette's own choice so
   that a rerun draws the same way. Sampled numbers are still not comparable
   with the leaderboard's greedy Gemma 4 entry.
4. **Filtered samples are served.** The leaderboard drops a sample whose
   normalized reference is empty or the ignore marker before inference;
   pipette serves it, requires its generation, and leaves it out of the score.
   The same samples are scored.
5. **A partial run is no result.** The leaderboard scores whatever its
   manifest holds and only warns when a chunk of a call is missing; pipette
   refuses to score a run with a missing or failed sample.
6. **Unrounded values.** The rates are kept unrounded; `leaderboard_result`
   gives the leaderboard's console rounding of the headline (§3).
7. **Corner cases.** A dataset with no reference word has no rate (the
   leaderboard reports 0.0, or infinity with errors); a hypothesis on which the
   normalizer raises scores as empty instead of failing the dataset.
8. **Audio decoding.** One resample for every model, the path of the
   leaderboard's API image (`datasets` 2.19.0: soundfile, then librosa's
   `soxr_hq`); its model jobs on `datasets` ≥ 4 decode and resample through
   torchcodec (FFmpeg) instead. Only resampled clips can differ: MonsoonASR's
   and URGENT24's non-16 kHz ones.
9. **Answer cleanup.** Each profile removes its model's own control tokens or
   echoed prompt (§2) before scoring.

**FLEURS (the leaderboard's multilingual scoring):**

1. **7 locales**, pipette's selection: the six languages of the leaderboard's
   multilingual FLEURS runs but Armenian (German, French, Italian, Spanish,
   Portuguese, Dutch), and English. The leaderboard publishes no English FLEURS
   column; `en_us` is scored by the same code path, its multilingual scoring
   with the language `en` (its English normalizer, then the compound-pair
   alignment).
2. **The references are FLEURS's own `transcription`**, not the leaderboard's
   own cleaned copy of them, whose preparation is not published (§1). The
   scoring code is the leaderboard's, unchanged, so the same transcripts can
   score slightly differently from the leaderboard's published rows only
   because of the reference text.
3. **The summary across locales.** The leaderboard shows each language
   separately; pipette's macro WER over the 7 locales is its own summary.
4. **num2words is pinned** (0.5.14), which the leaderboard installs unpinned;
   its other code is pinned as above.
5. **Pinned revision, completeness, prompts and decoding**, as for the
   leaderboard sets above.
6. **MiniCPM-o's non-English prompts are pipette's**, chosen among six
   candidates on the FLEURS validation split by a rule fixed beforehand, so
   the test split scored no candidate (§2).
7. **Every model runs every locale.** A cell in a language its model card does
   not declare for speech recognition is scored and marked, not dropped; the
   catalog entries carry each card's languages (pipette-mgmt's
   `docs/benchmarks.md`, "The languages a card declares").

## 5. Checking against the public references

`tests/test_asr_public_reference.py` checks pipette's numbers against the public
reference implementations themselves:

- the leaderboard's own code (huggingface/open_asr_leaderboard @ `62a457c5`:
  `data_utils` normalizer and row filter, `eval_utils.merge_chunked_manifest`,
  `align_manifests`, `score_manifests`) must give exactly our hits,
  substitutions, deletions, insertions, reference words, corpus rate and rounded
  WER, set by set and unit by unit;
- the same leaderboard code as its multilingual scoring runs it
  (`eval_utils._normalize` with the language, `_align_pair` with
  `multilingual=True`, `align_manifests`), with num2words 0.5.14, must give
  exactly our normalized texts, counts and corpus rate, locale by locale and
  pair by pair, and the vendored excerpts must be its source text;
- `datasets` 2.19.0 `Audio(sampling_rate=16000)` must give exactly our 16 kHz
  mono samples, and the served WAV must carry them;
- negative controls: kaldialign without `merge_compounds` on the English sets,
  and on FLEURS the multilingual normalizer without its number step, or the
  alignment without `normalize_compound_pairs`, must each miss.

The corpus (`tests/asr_public_reference.py`) is a synthetic part (English and
every FLEURS language, built to exercise what the normalizers rewrite, with
hypotheses as perturbed references, numbers among them written the other way,
plus chunked calls), every FLEURS test transcript at the pinned revision
(`transcription`, the reference, and `raw_transcription`), and every reference and some clips of each built dataset
found under `PIPETTE_SCORES_DATA_DIR` (several roots joined with
`:`). When `PIPETTE_SCORES_DATA_DIR` is set, each root must exist and hold at
least one built ASR dataset, or the live tests fail (a mistyped path never
passes silently). Every answer is matched to its input by name, and the number
of answers must equal the number of inputs.

The default test run compares against the reference answers recorded on the
synthetic part (`tests/fixtures/asr_public_reference.json`). For the audio it
records, per synthetic clip, the shape and four fixed ±1 projections of the
samples `datasets` 2.19.0 decodes, compared to within 1e-4 (exact equality is
the live layer's, on the platform the reference runs on). soxr's "high"
quality runs libsoxr's single-precision engine, whose SIMD and scalar code
round differently per CPU architecture: a resampled clip's float32 samples
differ by a few ulps (up to ~1.2e-7) between macOS arm64, Linux x86_64 and
Linux aarch64, its projections by up to ~6e-6, while the near misses move them
by 2.2e-2 or more. So a resampled clip (MonsoonASR, most of URGENT24) is the
same on every call on one architecture, not across architectures: the builder
resamples it once and stores the result, so a served clip never passes through
soxr (§1). Its
negative controls, soxr's "very high"
quality in place of "high" and the resample without librosa's length rule, must
miss. With the reference environment it runs the reference code itself:

    packages/pipette-scores/scripts/setup_asr_reference_env.sh /tmp/asr-reference
    PIPETTE_ASR_REFERENCE_ENV=/tmp/asr-reference \
    PIPETTE_SCORES_DATA_DIR=/data/pipette-scores \
        uv run pytest packages/pipette-scores/tests/test_asr_public_reference.py

and `python packages/pipette-scores/tests/asr_public_reference.py --write-fixture`
re-records the fixture. The environment (a separate virtualenv: `datasets`
2.19.0, librosa, soundfile, soxr, kaldialign 0.12.0, num2words 0.5.14, and a
checkout of the leaderboard repository at the pinned commit) is no dependency
of pipette-scores.

## 6. Serving

`GET /evals/<asr eval>/datasets/2026.10.1/samples?prompt_profile=<profile>`
returns the whole set, as the VLM evals' `/samples` does, with the
`X-Samples-Digest` header, and `/samples/digest` reports the digest alone. A
set is never held whole in memory: the server learns its digest first (one
pass over the set, which learns every profile's digest at once, then cached),
sends the header, and streams the body one sample at a time, each sample
written as the canonical JSON its digest hashes; a body that would not hash to
its header is cut off before it closes. The samples are not paged (an `offset`
or `limit` on an ASR request is a 400): a client that keeps a copy, as
pipette-mgmt's dataset cache does, takes the stream once and pins its digest,
and `/score` refuses (409) a `samples_digest` this server no longer serves for
the eval, dataset and profile. The digest covers everything served, the audio
bytes included, so a pin taken from it names the exact requests the model is
sent. The server reads one row group of source audio at a time and re-hashes
each clip; `/score` reads no audio at all. The full contract:
[httpapi.md](httpapi.md).
