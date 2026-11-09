#!/usr/bin/env bash
# Build a throwaway environment holding the PUBLIC reference implementations the
# ASR scoring and audio are checked against (tests/test_asr_public_reference.py):
#
#   - huggingface/open_asr_leaderboard at a pinned commit: its own normalizers
#     (English, and the multilingual one FLEURS is scored with, which spells
#     numbers out with num2words 0.5.14), row filter, chunk merge, compound-pair
#     step and kaldialign WER (normalizer/eval_utils.py, normalizer/data_utils.py);
#   - datasets 2.19.0 with librosa and soundfile: datasets.Audio(sampling_rate=16000)
#     decoding, as the leaderboard's prepare_data applies it.
#
# None of this is a dependency of pipette-scores. Usage:
#
#   packages/pipette-scores/scripts/setup_asr_reference_env.sh /tmp/asr-reference
#   PIPETTE_ASR_REFERENCE_ENV=/tmp/asr-reference uv run pytest \
#       packages/pipette-scores/tests/test_asr_public_reference.py
#
# Needs uv, git and network access.
set -euo pipefail

OPEN_ASR_LEADERBOARD_COMMIT=62a457c5c32d0de4df77e9470308aae7d45b055f

dir="${1:?usage: $0 DIR}"
mkdir -p "$dir"
dir="$(cd "$dir" && pwd)"

uv venv --quiet --python 3.12 "$dir/venv"
uv pip install --quiet --python "$dir/venv/bin/python" \
    "datasets==2.19.0" "librosa==1.0.0" "soxr==1.1.0" "soundfile==0.14.0" \
    "kaldialign==0.12.0" "num2words==0.5.14" "regex==2026.4.4" "numpy"

if [[ ! -d "$dir/open_asr_leaderboard/.git" ]]; then
    git clone --quiet https://github.com/huggingface/open_asr_leaderboard "$dir/open_asr_leaderboard"
fi
git -C "$dir/open_asr_leaderboard" fetch --quiet origin
git -C "$dir/open_asr_leaderboard" checkout --quiet --detach "$OPEN_ASR_LEADERBOARD_COMMIT"

echo "reference environment ready: PIPETTE_ASR_REFERENCE_ENV=$dir"
