"""Calibration-run inference spec.

Describes how `generate-completions` drives vLLM for each eval. Lives in the
calibration package because these are vLLM sampling knobs — not a concern of
the scoring service and not part of its HTTP contract.

Each `create-representative-dataset` run records the spec it used in
`metadata.json` (via `spec_for_metadata()`) so the published dataset carries
its own provenance.
"""

from pipette_scores.types import EvalId
from pydantic import BaseModel


class GenerationParams(BaseModel):
    max_tokens: int
    temperature: float = 0.0
    mcq_choices: list[str] | None = None


# max_tokens is the output budget for text evals; for MCQ it's unused by the
# sampler (hard-capped to 1) but still sizes max_model_len, so leave room for
# the prompt.
_GEN_PARAMS: dict[EvalId, GenerationParams] = {
    EvalId.IFBENCH: GenerationParams(max_tokens=2048),
    EvalId.IFSTRUCT: GenerationParams(max_tokens=4096),
    EvalId.GPQA_DIAMOND: GenerationParams(max_tokens=2048, mcq_choices=list("ABCD")),
    EvalId.MATH_500: GenerationParams(max_tokens=4096),
}


def get_generation_params(eval_id: EvalId) -> GenerationParams:
    return _GEN_PARAMS[EvalId(eval_id)]


def spec_for_metadata(eval_id: EvalId) -> dict:
    """JSON-serializable snapshot of the gen params for this eval, for metadata.json."""
    return get_generation_params(eval_id).model_dump(exclude_none=True)
