"""Central type definitions for pipette_scores."""

import enum
from typing import Any, Union

from pydantic import BaseModel, ConfigDict, field_validator

# ---------------------------------------------------------------------------
# Eval IDs
# ---------------------------------------------------------------------------


class EvalId(enum.StrEnum):
    IFBENCH = "ifbench"
    IFSTRUCT = "ifstruct"
    GPQA_DIAMOND = "gpqa_diamond"
    MATH_500 = "math_500"


# ---------------------------------------------------------------------------
# Messages & prompts
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: str
    content: str


class EvalSample(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    messages: list[ChatMessage]


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


class Math500Sample(BaseDatasetSample):
    prompt: str
    answer: str  # free-form gold answer (LaTeX / number), graded by score_math_generic


DatasetSample = Union[
    IFBenchSample,
    IFStructSample,
    GPQADiamondSample,
    Math500Sample,
]


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------


class SampleCompletion(BaseModel):
    id: str
    completion: str


# ---------------------------------------------------------------------------
# Scoring — internal score (per-eval scorers) vs. external API shape
# ---------------------------------------------------------------------------


class SampleScore(BaseModel):
    """Per-eval scorer output. Enriched to ScoredSample at the API layer."""

    id: str
    is_correct: bool


class ScoredSample(BaseModel):
    id: str
    messages: list[ChatMessage]
    completion: str
    is_correct: bool


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class ScoreRequest(BaseModel):
    eval_id: str
    dataset_name: str
    completions: list[SampleCompletion]


class ScoreResponse(BaseModel):
    runtime_version: str
    scored_samples: list[ScoredSample]
    # Eval-specific aggregate metrics (e.g. by-language accuracy, tp/fp counts).
    # Free-form because each eval produces its own shape; consumers log it or
    # inspect specific keys they recognize.
    context: dict[str, Any]


class SamplesResponse(BaseModel):
    samples: list[EvalSample]
