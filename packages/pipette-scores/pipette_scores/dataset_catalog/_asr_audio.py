"""A clip as every model receives it: decoded, 16 kHz mono, then the profile's cut and WAV.

A clip the source ships at 16 kHz mono is stored as shipped; any other
(MonsoonASR's 44.1 kHz, most of URGENT24's 16 to 48 kHz) is brought to 16 kHz
mono once, by the builder (_asr_build), the same way for every prompt profile, as the Open ASR
Leaderboard prepares audio for every model it evaluates:
``normalizer/data_utils.prepare_data`` casts the audio column to
``datasets.Audio(sampling_rate=16000)`` (huggingface/open_asr_leaderboard
@ 62a457c5c32d0de4df77e9470308aae7d45b055f). How that cast decodes depends on
the installed `datasets`: up to 3.x it decodes with soundfile and resamples with
librosa, from 4.0 it decodes and resamples with torchcodec (FFmpeg). The
leaderboard pins one version in one place only, its API-provider image
(``api/Dockerfile``: ``datasets==2.19.0``, "to avoid torchcodec dependency");
its GPU model jobs install `datasets` unpinned (``requirements/requirements_jobs.txt``)
and so, on a current install, decode through torchcodec. pipette follows the
API image's path, `datasets` 2.19.0 with soundfile and librosa, because it is the
leaderboard's only pinned one and is reproducible from pinned Python packages
without FFmpeg. Only the clips that are resampled can differ between the two
paths: MonsoonASR's and URGENT24's non-16 kHz clips (docs/asr-evals.md §4).
`datasets` 2.19.0 ``features/audio.py`` ``Audio.decode_example`` does, in order:

    array, sampling_rate = sf.read(file)          # float64
    array = array.T
    if self.mono:                                 # default True
        array = librosa.to_mono(array)            # mean over channels
    if self.sampling_rate and self.sampling_rate != sampling_rate:
        array = librosa.resample(array, orig_sr=sampling_rate, target_sr=self.sampling_rate)

``librosa.resample``'s default ``res_type="soxr_hq"`` (librosa 0.10 to 1.0) runs
``soxr.resample(y, orig_sr, target_sr, quality="soxr_hq")`` and then
``util.fix_length`` to ``ceil(n * target_sr / orig_sr)`` samples (zero-padded or
cut), in the input's dtype. `resample_16k_mono` is that sequence without the
librosa dependency; tests/test_asr_public_reference.py checks it sample for
sample against `datasets` 2.19.0 with librosa itself, on the same machine:
soxr's high quality runs libsoxr's single-precision engine, which rounds
differently per CPU architecture (a few float32 ulps between macOS arm64, Linux
x86_64 and Linux aarch64), so a resampled clip's samples are exact per
architecture, not across architectures. That is why the builder, not the
server, resamples.

After that, a profile may cut the clip to its model's audio limit (Gemma 4: 30 s,
see asr_prompt_profiles.py) and the result is written as a WAV that carries
those samples exactly (`wav`): 16-bit PCM when every sample is a 16-bit value
(k / 32768 for an integer k in [-32768, 32767], as a 16 kHz mono 16-bit source
decodes, and as some float sources hold), 32-bit float otherwise (a resampled or
downmixed clip, a float source with finer values). Both decode to the same
float32 samples: soundfile reads PCM_16 as k / 32768, and llama.cpp's audio
loader (miniaudio, ``ma_pcm_s16_to_f32``) as k * 0.000030517578125, the same
power of two, so the PCM16 WAV is lossless and half the bytes. The WAV is
written here, not by libsndfile, whose float WAVs carry a PEAK chunk with a
timestamp: the served bytes are the same on every call (`encode`).
"""

from __future__ import annotations

import base64
import io
import struct
from typing import Any

TARGET_RATE = 16000


def resample_16k_mono(audio: bytes) -> Any:
    """The clip as ``datasets.Audio(sampling_rate=16000)`` (2.19.0) decodes it:
    float64, the channels averaged, then resampled with soxr HQ to
    ``ceil(n * 16000 / rate)`` samples when not already at 16 kHz."""
    import math

    import numpy as np
    import soundfile as sf

    array, rate = sf.read(io.BytesIO(audio))
    array = array.T
    if array.ndim > 1:
        # librosa.to_mono: np.mean over every axis but the last.
        array = np.mean(array, axis=tuple(range(array.ndim - 1)))
    if rate == TARGET_RATE:
        return array
    import soxr

    out = soxr.resample(array, rate, TARGET_RATE, quality="soxr_hq")
    size = int(math.ceil(array.shape[-1] * (float(TARGET_RATE) / rate)))
    # librosa.util.fix_length: cut, or pad with zeros at the end.
    if out.shape[-1] > size:
        out = out[:size]
    elif out.shape[-1] < size:
        out = np.pad(out, (0, size - out.shape[-1]), mode="constant")
    return np.asarray(out, dtype=array.dtype)


def float_wav(samples: Any, rate: int = TARGET_RATE) -> bytes:
    """Mono ``samples`` as a 32-bit IEEE-float WAV: the RIFF header, a 16-byte
    ``fmt `` chunk (format 3), a ``fact`` chunk (the frame count) and ``data``,
    the layout libsndfile writes for ``subtype="FLOAT"`` without its PEAK chunk,
    so the bytes depend on the samples alone. The samples are converted to
    little-endian float32 as libsndfile converts them (round to nearest)."""
    import numpy as np

    data = np.ascontiguousarray(np.asarray(samples).reshape(-1), dtype="<f4").tobytes()
    frames = len(data) // 4
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    chunks = b"".join(
        [
            b"fmt " + struct.pack("<I", len(fmt)) + fmt,
            b"fact" + struct.pack("<II", 4, frames),
            b"data" + struct.pack("<I", len(data)) + data,
        ]
    )
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


PCM16_SCALE = 32768.0


def pcm16_values(samples: Any) -> Any | None:
    """``samples`` as the 16-bit integers they are, when each is exactly k / 32768
    for an integer k in [-32768, 32767]; None when any is not (a 16-bit WAV would
    then round it)."""
    import numpy as np

    scaled = np.asarray(samples, dtype=np.float64).reshape(-1) * PCM16_SCALE
    if scaled.size and not (
        np.array_equal(scaled, np.rint(scaled)) and scaled.min() >= -PCM16_SCALE and scaled.max() <= PCM16_SCALE - 1
    ):
        return None
    return scaled.astype("<i2")


def pcm16_wav(values: Any, rate: int = TARGET_RATE) -> bytes:
    """Mono 16-bit integer ``values`` as a PCM WAV: the RIFF header, a 16-byte
    ``fmt `` chunk (format 1) and ``data``, the layout libsndfile writes for
    ``subtype="PCM_16"``."""
    import numpy as np

    data = np.ascontiguousarray(np.asarray(values).reshape(-1), dtype="<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    chunks = b"".join([b"fmt " + struct.pack("<I", len(fmt)) + fmt, b"data" + struct.pack("<I", len(data)) + data])
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def wav(samples: Any, rate: int = TARGET_RATE) -> bytes:
    """Mono ``samples`` as the smallest WAV that carries them exactly: 16-bit
    PCM when every sample is a 16-bit value (`pcm16_values`), 32-bit float
    otherwise. Either decodes to ``samples`` as float32."""
    values = pcm16_values(samples)
    return float_wav(samples, rate) if values is None else pcm16_wav(values, rate)


def to_16k_mono(audio: bytes, *, max_samples: int | None = None) -> Any:
    """The clip's samples as served: 16 kHz mono (`resample_16k_mono`), cut to
    ``max_samples`` when given."""
    mono = resample_16k_mono(audio)
    return mono if max_samples is None else mono[:max_samples]


def encode_samples(samples: Any) -> str:
    """16 kHz mono samples as the served base64 WAV (`wav`)."""
    return base64.b64encode(wav(samples)).decode("ascii")


def encode(audio: bytes, *, max_samples: int | None = None) -> str:
    """The clip as base64 WAV: 16 kHz mono, cut to ``max_samples`` when given
    (`to_16k_mono`), written as 16-bit PCM where that is exact and as 32-bit
    float otherwise (`wav`). The same clip and cut always give the same bytes."""
    return encode_samples(to_16k_mono(audio, max_samples=max_samples))
