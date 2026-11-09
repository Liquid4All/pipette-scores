"""Central type definitions for pipette_scores."""

import enum
from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, field_validator, model_serializer

# ---------------------------------------------------------------------------
# Eval IDs
# ---------------------------------------------------------------------------


class EvalId(enum.StrEnum):
    IFBENCH = "ifbench"
    IFSTRUCT = "ifstruct"
    GPQA_DIAMOND = "gpqa_diamond"
    MATH_500 = "math_500"
    # VLM evals — string values must equal the pipette-mgmt `parameter_lmms_task`
    # and the scores-server route segment.
    MME = "mme"
    MMSTAR = "mmstar"
    OCRBENCH = "ocrbench"
    MMMU_VAL = "mmmu_val"
    REALWORLDQA = "realworldqa"
    MMBENCH_EN_DEV = "mmbench_en_dev"
    POPE = "pope"
    # ASR evals: one per dataset of the Open ASR Leaderboard's English public
    # sets (in its column order; the asr_esb_ prefix is historical, see
    # dataset_catalog/asr_datasets.py) and of the FLEURS locales evaluated.
    # Values equal the pipette-mgmt task name and the scores-server route segment.
    ASR_ESB_AMI_CLEANED = "asr_esb_ami_cleaned"
    ASR_ESB_EARNINGS22_CLEANED_AA_CHUNKED = "asr_esb_earnings22_cleaned_aa_chunked"
    ASR_ESB_GIGASPEECH_CLEANED = "asr_esb_gigaspeech_cleaned"
    ASR_ESB_LIBRISPEECH_TEST_CLEAN = "asr_esb_librispeech_test_clean"
    ASR_ESB_LIBRISPEECH_TEST_OTHER = "asr_esb_librispeech_test_other"
    ASR_ESB_SPGISPEECH = "asr_esb_spgispeech"
    ASR_ESB_URGENT2024 = "asr_esb_urgent2024"
    ASR_ESB_MONSOON_EN_IN = "asr_esb_monsoon_en_in"
    ASR_ESB_VOXPOPULI_CLEANED_AA = "asr_esb_voxpopuli_cleaned_aa"
    ASR_FLEURS_EN_US = "asr_fleurs_en_us"
    ASR_FLEURS_ES_419 = "asr_fleurs_es_419"
    ASR_FLEURS_PT_BR = "asr_fleurs_pt_br"
    ASR_FLEURS_DE_DE = "asr_fleurs_de_de"
    ASR_FLEURS_FR_FR = "asr_fleurs_fr_fr"
    ASR_FLEURS_IT_IT = "asr_fleurs_it_it"
    ASR_FLEURS_NL_NL = "asr_fleurs_nl_nl"


# The ASR evals in catalog order (for reports and builds), and
# as a set (for membership checks).
ASR_EVAL_ORDER: tuple[EvalId, ...] = tuple(e for e in EvalId if e.value.startswith("asr_"))
ASR_EVAL_IDS: frozenset[EvalId] = frozenset(ASR_EVAL_ORDER)


# ---------------------------------------------------------------------------
# Messages & prompts
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: str
    # String for text-only evals; list of OpenAI-style content blocks (text /
    # image_url) for multimodal evals.
    content: str | list[dict]


def _without_unset(handler: SerializerFunctionWrapHandler, value: BaseModel, *optional: str) -> dict[str, Any]:
    """Serialize, leaving out the named optional fields when they are None, so a
    payload that does not use them keeps its earlier wire shape exactly."""
    data = handler(value)
    for name in optional:
        if data.get(name) is None:
            data.pop(name, None)
    return data


class EvalSample(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    messages: list[ChatMessage]
    # Per-sample chat-template arguments the client passes with the request
    # (ASR: Voxtral's transcription mode and language). Omitted when None.
    chat_template_kwargs: dict[str, Any] | None = None

    @model_serializer(mode="wrap")
    # No return annotation: pydantic would publish it as the OpenAPI schema.
    def _serialize(self, handler: SerializerFunctionWrapHandler):
        return _without_unset(handler, self, "chat_template_kwargs")


# ---------------------------------------------------------------------------
# Dataset samples
# ---------------------------------------------------------------------------


class BaseDatasetSample(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str


class IFBenchSample(BaseDatasetSample):
    key: str
    instruction_id_list: tuple[str, ...]
    prompt: str
    kwargs: tuple[dict, ...]
    n_constraints: int


class IFStructSample(BaseDatasetSample):
    seed: int
    prompt: str
    json_schema: dict[str, Any]
    top_level_count: int | list[int] | None
    top_level_key: str | None
    require_wrapper_key: bool
    require_code_block: bool
    require_no_commentary: bool
    output_format: str
    entity_type: str


class GPQADiamondSample(BaseDatasetSample):
    prompt: str
    answer: str  # correct option letter, e.g. "A".."D"

    @field_validator("answer")
    @classmethod
    def _answer_is_option_letter(_cls, v: str) -> str:
        # Normalize and fail loud on out-of-domain ground truth (e.g. "E", "",
        # multi-char) so a bad calibration value surfaces at load time rather
        # than silently scoring every attempt wrong.
        normalized = v.strip().upper()
        if normalized not in {"A", "B", "C", "D"}:
            raise ValueError(f"answer must be one of A-D, got {v!r}")
        return normalized


class Math500Grader(enum.StrEnum):
    """Which grader a MATH-500 dataset version is scored with (metadata ``grader``)."""

    # The AA-aligned score_math_generic that liquid_evals' math500_std track uses.
    MATH_GENERIC = "math_generic"
    # liquid_evals' plain math500 task: extract_math_answer, then verify_math_answer.
    LIQUID_EVALS_MATH500 = "liquid_evals_math500"


class Math500Sample(BaseDatasetSample):
    prompt: str
    answer: str  # free-form gold answer (LaTeX / number)
    # Set from the dataset's metadata.json; versions that predate the key use math_generic.
    grader: Math500Grader = Math500Grader.MATH_GENERIC


class VlmSample(BaseDatasetSample):
    """Shared sample shape for all VLM evals.

    ``raw_doc`` carries the benchmark-specific fields the scorer needs
    (option columns, answer, category, index, ...) — see
    ``scoring/vlm_upstream/PORTING_NOTES.md`` for the per-benchmark contract.
    """

    prompt: str
    answer: Any
    raw_doc: dict[str, Any]


class AsrSample(BaseDatasetSample):
    """One ASR clip's ground truth (no audio: the scorer never reads it).

    ``scoring`` is the dataset's recipe from metadata.json (normalizer, aligner,
    Open ASR exclusions); ``session``/``chunk_index`` place a chunk of chunked
    long-form audio (Earnings22) in the session it is scored with.
    """

    reference: str
    language: str  # ISO 639-3
    session: str | None = None
    chunk_index: int | None = None
    scoring: dict[str, Any]
    raw_doc: dict[str, Any]


DatasetSample = Union[
    IFBenchSample,
    IFStructSample,
    GPQADiamondSample,
    Math500Sample,
    VlmSample,
    AsrSample,
]


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------


class SampleCompletion(BaseModel):
    id: str
    completion: str
    # How the client's generation ended, as plan-types names it ("eos",
    # "truncated", "doom_loop", "failure", "unknown"); None when not reported.
    stop_reason: str | None = None


# ---------------------------------------------------------------------------
# Scoring — internal score (per-eval scorers) vs. external API shape
# ---------------------------------------------------------------------------


class SampleScore(BaseModel):
    """Per-eval scorer output. Enriched to ScoredSample at the API layer."""

    id: str
    is_correct: bool
    # ASR: the sample's alignment (see ScoredSample.measures). None elsewhere.
    measures: dict[str, Any] | None = None


class ScoredSample(BaseModel):
    id: str
    messages: list[ChatMessage]
    completion: str
    is_correct: bool
    # ASR only: the sample's part in its scoring unit (docs/asr-evals.md §3:
    # the unit's counts and normalized texts on one sample per unit, so summing
    # them over samples gives the dataset's totals). Omitted when None.
    measures: dict[str, Any] | None = None
    # The sample's chat-template arguments as served (ASR voxtral: the
    # transcription mode and language hint). Omitted when None.
    chat_template_kwargs: dict[str, Any] | None = None

    @model_serializer(mode="wrap")
    # No return annotation: pydantic would publish it as the OpenAPI schema.
    def _serialize(self, handler: SerializerFunctionWrapHandler):
        return _without_unset(handler, self, "measures", "chat_template_kwargs")


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class ScoreRequest(BaseModel):
    eval_id: str
    dataset_name: str
    completions: list[SampleCompletion]
    # The job's enable_thinking model flag; None when the job did not set it.
    enable_thinking: bool | None = None
    # The run's prompt profile. VLM evals (dataset_catalog/prompt_profiles.py),
    # optional: grading is the same, but scored_samples carry the prompts it
    # produced. ASR evals (dataset_catalog/asr_prompt_profiles.py), required:
    # each completion is cleaned as that profile says.
    prompt_profile: str | None = None
    # The digest of the samples the run was given (`sha256:<hex>`, as
    # /samples/digest reports it), if known: scoring refuses with 409 when
    # this server no longer serves those samples.
    samples_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")


class ScoreResponse(BaseModel):
    runtime_version: str
    scored_samples: list[ScoredSample]
    # Eval-specific aggregate metrics (e.g. by-language accuracy, tp/fp counts).
    # Free-form because each eval produces its own shape; consumers log it or
    # inspect specific keys they recognize.
    context: dict[str, Any]


class SamplesResponse(BaseModel):
    samples: list[EvalSample]


class SamplesDigestResponse(BaseModel):
    # hashing.samples_digest of what /samples serves for the same request.
    digest: str
    sample_count: int
