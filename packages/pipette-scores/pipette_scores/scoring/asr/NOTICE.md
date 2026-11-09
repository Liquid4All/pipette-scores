# ASR normalizer provenance

ASR scoring follows the Open ASR Leaderboard's two public protocols, English
and multilingual (FLEURS), so its code is vendored unchanged from its public
source, huggingface/open_asr_leaderboard.

## `open_asr_normalizer/normalizer.py`, `english_abbreviations.py` (English)

Byte-identical to `huggingface/open_asr_leaderboard` at immutable revision
`67e8bd6acea240819ad67080f6f31e15d4a90da5`, and unchanged at
`62a457c5c32d0de4df77e9470308aae7d45b055f` (the revision the rest of the
leaderboard method is cited at):

- https://github.com/huggingface/open_asr_leaderboard/blob/67e8bd6acea240819ad67080f6f31e15d4a90da5/normalizer/normalizer.py
- https://github.com/huggingface/open_asr_leaderboard/blob/67e8bd6acea240819ad67080f6f31e15d4a90da5/normalizer/english_abbreviations.py

`normalizer.py` also holds the `BasicMultilingualTextNormalizer` the
multilingual normalizer below builds on. Licensed under the Apache License 2.0
(see the source headers).

## `open_asr_normalizer/multilingual.py` (multilingual, FLEURS)

Excerpts of `huggingface/open_asr_leaderboard` at
`62a457c5c32d0de4df77e9470308aae7d45b055f`, unchanged:

- `normalizer/data_utils.py` lines 28-94 (`FILLER_WORDS`, `MultilingualNormalizer`;
  the file's sha256 at that revision is
  `1d1a8320567d431bb4bb0066cb40c2d94361148d5eb745f30553b8e7f974f513`):
  https://github.com/huggingface/open_asr_leaderboard/blob/62a457c5c32d0de4df77e9470308aae7d45b055f/normalizer/data_utils.py#L28-L94
- `normalizer/eval_utils.py` lines 68-99 (`normalize_compound_pairs`; sha256
  `362756990a8087cf45acc0cde211e52401e2dca9d13a17b1bf3e1aadf270f6fb`):
  https://github.com/huggingface/open_asr_leaderboard/blob/62a457c5c32d0de4df77e9470308aae7d45b055f/normalizer/eval_utils.py#L68-L99

`data_utils.py` is the leaderboard's data loader as well (it imports `datasets`,
`huggingface_hub` and its own `normalizer` package), so only these parts are
vendored. pipette's are the module header, the imports, and a stand-in for the
leaderboard's `ArmenianTextNormalizer` (`normalizer/armenian_normalizer.py`,
not vendored), which `MultilingualNormalizer` constructs for `hy` and which
refuses to normalize: pipette evaluates no Armenian. `scoring/asr/normalize.py`
selects the normalizer as the leaderboard's `eval_utils._normalize` does (its
English normalizer for `en`, else `data_utils.ml_normalizer`, a
`MultilingualNormalizer(remove_diacritics=False)`, with the language), and
`scoring/asr/align.py` aligns as its `_align_pair` does with
`multilingual=True`. Licensed under the Apache License 2.0 (upstream's
`LICENSE`).

## Checks

`tests/test_asr_scoring.py` pins all three vendored files by sha256.
`tests/test_asr_public_reference.py` compares both protocols with the
leaderboard's own scoring code at the pinned revision, and checks that the
excerpts in `multilingual.py` are that revision's source text (see
`docs/asr-evals.md`, "Checking against the public references").

## Pinned dependencies

`packages/pipette-scores/pyproject.toml`:

- `kaldialign==0.12.0`: the version the leaderboard's scoring environment pins
  (`requirements/requirements_jobs.txt`).
- `num2words==0.5.14`: the multilingual normalizer spells numbers out with it.
  The leaderboard's requirements name it unpinned
  (`requirements/requirements_jobs.txt`), which installs its latest release,
  0.5.14 (2024-12-17); it is pinned so that the FLEURS numbers are
  reproducible.
- `regex==2026.4.4`: the normalizers' `\p{…}` classes come from its Unicode
  tables.
- `soundfile==0.14.0` (libsndfile 1.2.2) and `soxr==1.1.0`: the ASR audio is
  decoded with them, and a clip not at 16 kHz mono resampled with them when its
  dataset is built (`dataset_catalog/_asr_audio.py`), as
  `datasets.Audio(sampling_rate=16000)` does through librosa's default
  `soxr_hq`.

## Licenses

All sources are listed in the repository's `NOTICE`, with their license texts in
`THIRD_PARTY_LICENSES.md` (both ship in the wheel): Apache-2.0 for
open_asr_leaderboard, whose English normalizer is, per its header, mostly
copied from openai/whisper (MIT).

## Updating

An update pins new immutable revisions, re-vendors unchanged, re-runs the
public-reference parity tests, and updates this record. Numbers scored before
and after a normalizer change are not comparable.
