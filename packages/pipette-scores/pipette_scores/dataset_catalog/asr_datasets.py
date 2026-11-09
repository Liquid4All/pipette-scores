"""The ASR eval datasets: the Open ASR Leaderboard's English sets, and FLEURS.

One entry per EvalId: where each set's audio and reference come from, at which
pinned revision, how it is scored, how many clips it holds, and the fingerprint
of our full build. The builder (`_asr_build.py`) reads the sources from here and
writes the recipe into each dataset's metadata.json; the loader refuses a
dataset whose recipe, language or fingerprint differs from this catalog's.

**Open ASR Leaderboard, English** (huggingface/open_asr_leaderboard; the
leaderboard Space hf-audio/open_asr_leaderboard). The nine public sets its
English average is taken over, as of its 09-10-2026 version ("Add URGENT24 to
avg English WER", Space ``init.py`` ``default_datasets``): AMI-Cleaned,
Earnings22-Cleaned-AA-chunked, Gigaspeech-Cleaned, LS Clean, LS Other,
SPGISpeech, URGENT24, Voice Arena Monsoon, Voxpopuli-AA-Cleaned, in that order
(the "public" columns of ``normalizer/eval_utils.score_results`` and the
``DATASET_CONFIGS`` of its job scripts, @ 62a457c5c32d0de4df77e9470308aae7d45b055f).
The Space's displayed average also folds in two private sets that no third
party can run, so the mean here is the public-set mean. The EvalIds keep the
``asr_esb_`` prefix of the leaderboard's lineage (the ESB benchmark, Gandhi et
al. 2022, arXiv 2210.13352; the dataset repo was ``esb-datasets-test-only-sorted``)
so that consumers' ids stay stable; the method is the leaderboard's, not ESB's
(ESB scored orthographic WER over other sets).

**FLEURS** (google/fleurs, Conneau et al. 2022, arXiv 2205.12446): English and
the six languages of the Open ASR Leaderboard's multilingual FLEURS runs, scored
with the leaderboard's multilingual scoring (``score_results`` with
``multilingual=True`` and the language, @ 62a457c5c32d0de4df77e9470308aae7d45b055f:
its normalizer, which spells numbers out in the language so that digits and
number words compare equal, and its compound-aware kaldialign word error rate)
applied to FLEURS's own ``transcription`` references. The leaderboard's
published FLEURS rows use its own cleaned copy of the references instead, so
its numbers can differ slightly from these.

The leaderboard loads its datasets unpinned; pinning each source at a revision
is pipette's addition, for reproducibility.
"""

from __future__ import annotations

import dataclasses

from pipette_scores.scoring.asr.align import Aligner
from pipette_scores.scoring.asr.normalize import NormalizerRecipe
from pipette_scores.types import ASR_EVAL_ORDER, EvalId

DATASET_VERSION = "2026.10.1"

# hf-audio/open-asr-leaderboard at the revision that adds urgent2024 ("Add
# urgent2024 (#13)", 2026-10-07). The other configs read here are byte-identical
# to those of the earlier b6bdcd0beb34f8975dc659796176d88f43aff502 (that diff
# touches only README.md and adds urgent2024/ and urgent2024_clean/), so the
# fingerprints recorded at that revision hold.
OPEN_ASR_REPO = "hf-audio/open-asr-leaderboard"
OPEN_ASR_REVISION = "de79e57adf9983993bcc72586d7313460825f5eb"
# Voice Arena Monsoon: the leaderboard's VoiceArena/Monsoon_en_IN_test redirects
# to this repo.
MONSOON_REPO = "VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN"
MONSOON_REVISION = "bc1da7b42ef6e2853123c97bf6d22067e4802d11"
# Earnings22-Cleaned-AA-chunked: the chunks, and their calls' transcripts from the
# parent set (normalizer/data_utils.CHUNKED_DATASETS, load_chunked_data).
EARNINGS22_CHUNKS_REPO = "ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
EARNINGS22_CHUNKS_REVISION = "27f7d0056d593231974a752ce03c1e965da69b3f"
EARNINGS22_CHUNKS_FILE = "earnings22_cleaned_aa_chunked_v1.jsonl"
EARNINGS22_PARENTS_REPO = "ArtificialAnalysis/Earnings22-Cleaned-AA"
EARNINGS22_PARENTS_REVISION = "5999936ee3d8d3cde160e3c65a8cc6ff5b7c5276"
EARNINGS22_PARENTS_FILE = "earnings22_cleaned_aa_v1.jsonl"
# google/fleurs main since 2026-05-15 (its parquet-data/<locale>/ files).
FLEURS_REPO = "google/fleurs"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"


@dataclasses.dataclass(frozen=True)
class AsrDataset:
    eval_id: EvalId
    family: str  # "open_asr_leaderboard" or "fleurs"
    repo: str
    revision: str
    subset: str
    split: str
    reference_column: str
    language: str  # ISO 639-3
    normalizer: NormalizerRecipe
    aligner: Aligner
    open_asr_exclusions: bool
    # Clips in the full build; asserted unless --limit. None = not yet known.
    expected_samples: int | None
    license: str
    license_notes: str
    # Fingerprint of the full build's source rows (_asr_common.source_sha256):
    # the audio-inclusive `content_sha256` with each resampled clip counted as
    # the source shipped it, so any machine reproduces it. Asserted by a full
    # build and checked at load for the canonical DATASET_VERSION. "" = not yet
    # filled (needs a full build).
    expected_source_sha256: str = ""

    @property
    def scoring(self) -> dict[str, str | bool]:
        return {
            "normalizer": self.normalizer.value,
            "aligner": self.aligner.value,
            "open_asr_exclusions": self.open_asr_exclusions,
        }


# `source_sha256` of each dataset's full build (its metadata.json), recorded
# from our builds of this catalog's pinned sources. The input is pinned, not
# the output: MonsoonASR's and URGENT24's clips are resampled by the builder,
# and soxr rounds per CPU architecture, so their stored bytes (and
# `content_sha256`) differ by machine while their source does not. For the
# other sets the two fingerprints are equal. A dataset missing here has not
# been built yet; its build is not fingerprint-checked until it is filled.
EXPECTED_SOURCE_SHA256: dict[str, str] = {
    "asr_esb_ami_cleaned": "797c85259c7fb0b48d480ee59b7dbb200e4980cec9d3fed12a5c2e86214ec60e",
    "asr_esb_gigaspeech_cleaned": "2319583c8dee37975e4e1de4a9027c9c636d3b9f8245f54b9473930fd4671954",
    "asr_esb_librispeech_test_other": "cc542f21a71a52f52a9e403435c6b7b97c6bc6dd035085162c3da18db23afb13",
    "asr_esb_spgispeech": "59a74aa484125a451c685a07a2bc386cac898ecd65f616fa2d8a606c4ab65bf5",
    "asr_esb_urgent2024": "d34af39602eb850899831527a95a52c139023f3e4f7ee1706245ac4ecff18c34",
    "asr_esb_monsoon_en_in": "6f1dc75ba4d4987e2c769739e65f022908f9ed52647b0b3de83c5d66c22528cd",
    "asr_fleurs_en_us": "12d7987ddab4b70be2224e227906cdcd8e691c60d1c4744fa518d3192b101c84",
    "asr_fleurs_es_419": "99ce91c48f7985d36ba99c145b4f140298350a9d41868cc46d2c6b7cdd2e2ee6",
    "asr_fleurs_pt_br": "0eaa5667353c164fd4550fa93eba0de570599d8757338fce9aebc99f8731d2ab",
    "asr_fleurs_de_de": "a0c50e364388dcc00aee89b03ddb97b0f47fc8da295276ec67b1d17cb61f0f9a",
    "asr_fleurs_fr_fr": "3f2b76bd2e42b418ec8ad086d85973b83ebd9e115fc81167f10d0af2c3a1db9d",
    "asr_fleurs_it_it": "8130bebe617d1955cc23ed12463aacf39e5ef2b44fcbf39237e30a53c2bb964c",
    "asr_esb_earnings22_cleaned_aa_chunked": "50d2ffe0cb3214a136f18b17ab27c884e9e147f94b2cf48008d42ac45cf78a4d",
    "asr_esb_librispeech_test_clean": "a197151ac15bc4acca9b34b5dfb330d1034e1bfd93d0a022fc68ec9c29fd5508",
    "asr_esb_voxpopuli_cleaned_aa": "0da49af1cff86560208d27bffbf948e54549d050600f715f29ab7c16f26fc19f",
    "asr_fleurs_nl_nl": "cca0257e92a4ec142b7a57ec2e31e8a4fe0e68715cd91ca550fc3bf7ae94af23",
}


# Each set's license as its dataset card (or the leaderboard's About table,
# Space hf-audio/open_asr_leaderboard constants.py) states it. Every set is for
# internal evaluation only here: the built data is never redistributed.
_OPEN_ASR_LICENSES = {
    "ami_cleaned": ("CC-BY-4.0", "AMI Meeting Corpus; attribution required. Internal evaluation use only."),
    "gigaspeech_cleaned": (
        "Apache-2.0 (gated: SpeechColab terms)",
        "GigaSpeech; access is gated behind the SpeechColab terms of use "
        "(huggingface.co/datasets/speechcolab/gigaspeech). Internal evaluation use only.",
    ),
    "librispeech": (
        "CC-BY-4.0",
        "LibriSpeech (LibriVox public-domain recordings); attribution required. Internal evaluation use only.",
    ),
    "spgispeech": (
        "Kensho SPGISpeech user agreement (non-commercial)",
        "SPGISpeech may be used for non-commercial research only. Internal evaluation use only.",
    ),
    "urgent2024": (
        "Mixed; non-commercial",
        "URGENT 2024 Challenge speech (Common Voice 11.0 CC0, LibriTTS CC-BY-4.0, VCTK ODC-BY, Creative Commons "
        "YouTube audio) mixed with noise and room responses including WHAM! (CC-BY-NC-4.0) and TUT Urban Acoustic "
        "Scenes 2018 (non-commercial, research only), which make the noisy set research/non-commercial use only "
        "(leaderboard About table; huggingface.co/datasets/urgent-challenge/urgent2024_official). Internal "
        "evaluation use only.",
    ),
    "voxpopuli_cleaned_aa": (
        "Apache-2.0",
        "VoxPopuli (European Parliament; CC0) with references cleaned by Artificial Analysis, released under "
        "Apache-2.0 (ArtificialAnalysis/VoxPopuli-Cleaned-AA card). Internal evaluation use only.",
    ),
}


def _open_asr(eval_id: EvalId, subset: str, split: str, expected: int | None) -> AsrDataset:
    license_name, notes = _OPEN_ASR_LICENSES[subset]
    return AsrDataset(
        eval_id=eval_id,
        family="open_asr_leaderboard",
        repo=OPEN_ASR_REPO,
        revision=OPEN_ASR_REVISION,
        subset=subset,
        split=split,
        reference_column="text",
        language="eng",
        normalizer=NormalizerRecipe.OPEN_ASR_ENGLISH,
        aligner=Aligner.KALDIALIGN_COMPOUND,
        open_asr_exclusions=True,
        expected_samples=expected,
        license=license_name,
        license_notes=notes,
        expected_source_sha256=EXPECTED_SOURCE_SHA256.get(eval_id.value, ""),
    )


# The FLEURS locales evaluated: (google/fleurs config, its ISO 639-3 language as
# pycountry names it, clips in its test split at FLEURS_REVISION). English, and
# the languages the Open ASR Leaderboard evaluates on FLEURS (its
# api/run_api_ml.sh `DATASET_CONFIGS`, @ 62a457c5c32d0de4df77e9470308aae7d45b055f:
# de, fr, it, es, pt, nl, hy) without Armenian, in pipette's order. FLEURS ships
# Spanish and Portuguese as es_419 and pt_br only.
FLEURS_LOCALES: tuple[tuple[str, str, int], ...] = (
    ("en_us", "eng", 647),
    ("es_419", "spa", 908),
    ("pt_br", "por", 919),
    ("de_de", "deu", 862),
    ("fr_fr", "fra", 676),
    ("it_it", "ita", 865),
    ("nl_nl", "nld", 364),
)


def _fleurs(locale: str, language: str, expected: int) -> AsrDataset:
    return AsrDataset(
        eval_id=EvalId(f"asr_fleurs_{locale}"),
        family="fleurs",
        repo=FLEURS_REPO,
        revision=FLEURS_REVISION,
        subset=locale,
        split="test",
        # FLEURS's own normalized reference text (google/fleurs `transcription`).
        reference_column="transcription",
        language=language,
        # score_results(multilingual=True, language=<code>): eval_utils._normalize
        # and _align_pair with normalize_compound_pairs.
        normalizer=NormalizerRecipe.OPEN_ASR_MULTILINGUAL,
        aligner=Aligner.KALDIALIGN_COMPOUND_PAIRS,
        # Its multilingual runs drop only a reference that is empty or the ignore
        # marker before normalization (run_eval_ml.py); no FLEURS reference is.
        open_asr_exclusions=False,
        expected_samples=expected,
        license="CC-BY-4.0",
        license_notes="FLEURS (Google); attribution required. Internal evaluation use only.",
        expected_source_sha256=EXPECTED_SOURCE_SHA256.get(f"asr_fleurs_{locale}", ""),
    )


ASR_DATASETS: dict[EvalId, AsrDataset] = {
    d.eval_id: d
    for d in (
        # The Open ASR Leaderboard's English public sets, in its column order.
        _open_asr(EvalId.ASR_ESB_AMI_CLEANED, "ami_cleaned", "test", 7805),
        AsrDataset(
            eval_id=EvalId.ASR_ESB_EARNINGS22_CLEANED_AA_CHUNKED,
            family="open_asr_leaderboard",
            repo=EARNINGS22_CHUNKS_REPO,
            revision=EARNINGS22_CHUNKS_REVISION,
            subset="default",
            split="test",
            reference_column="transcript",
            language="eng",
            normalizer=NormalizerRecipe.OPEN_ASR_ENGLISH,
            aligner=Aligner.KALDIALIGN_COMPOUND,
            open_asr_exclusions=True,
            # 341 chunks of 6 calls, scored per call (eval_utils.merge_chunked_manifest).
            expected_samples=341,
            license="Apache-2.0",
            license_notes=(
                "Earnings-22 (Rev.com; CC-BY-SA-4.0) cleaned and chunked by Artificial Analysis, released under "
                "Apache-2.0 (ArtificialAnalysis/Earnings22-Cleaned-AA and -chunked cards). Internal evaluation use only."
            ),
            expected_source_sha256=EXPECTED_SOURCE_SHA256.get(EvalId.ASR_ESB_EARNINGS22_CLEANED_AA_CHUNKED.value, ""),
        ),
        _open_asr(EvalId.ASR_ESB_GIGASPEECH_CLEANED, "gigaspeech_cleaned", "test", 18768),
        _open_asr(EvalId.ASR_ESB_LIBRISPEECH_TEST_CLEAN, "librispeech", "test.clean", 2620),
        _open_asr(EvalId.ASR_ESB_LIBRISPEECH_TEST_OTHER, "librispeech", "test.other", 2939),
        _open_asr(EvalId.ASR_ESB_SPGISPEECH, "spgispeech", "test", 39341),
        # 1,000 clips (the dataset card's split size); fingerprint pinned above.
        _open_asr(EvalId.ASR_ESB_URGENT2024, "urgent2024", "test", 1000),
        AsrDataset(
            eval_id=EvalId.ASR_ESB_MONSOON_EN_IN,
            family="open_asr_leaderboard",
            repo=MONSOON_REPO,
            revision=MONSOON_REVISION,
            subset="default",
            split="test",
            reference_column="text",
            language="eng",
            normalizer=NormalizerRecipe.OPEN_ASR_ENGLISH,
            aligner=Aligner.KALDIALIGN_COMPOUND,
            open_asr_exclusions=True,
            expected_samples=2102,
            license="CC-BY-4.0",
            license_notes=(
                "Voice Arena Monsoon en-IN public test set (VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN card); "
                "attribution required. Internal evaluation use only."
            ),
            expected_source_sha256=EXPECTED_SOURCE_SHA256.get(EvalId.ASR_ESB_MONSOON_EN_IN.value, ""),
        ),
        _open_asr(EvalId.ASR_ESB_VOXPOPULI_CLEANED_AA, "voxpopuli_cleaned_aa", "test", 628),
        *(_fleurs(locale, language, n) for locale, language, n in FLEURS_LOCALES),
    )
}

if [e for e in ASR_EVAL_ORDER] != list(ASR_DATASETS):
    raise RuntimeError("ASR_DATASETS must hold every asr_* EvalId, in EvalId order")

OPEN_ASR_EVAL_IDS: tuple[EvalId, ...] = tuple(e for e, d in ASR_DATASETS.items() if d.family == "open_asr_leaderboard")
FLEURS_EVAL_IDS: tuple[EvalId, ...] = tuple(e for e, d in ASR_DATASETS.items() if d.family == "fleurs")
