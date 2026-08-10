"""Inference: generate per-sample completions via vLLM + per-eval generation spec."""

from pipette_calibration.inference.generate import generate_completions
from pipette_calibration.inference.spec import (
    GenerationParams,
    get_generation_params,
    spec_for_metadata,
)

__all__ = [
    "GenerationParams",
    "generate_completions",
    "get_generation_params",
    "spec_for_metadata",
]
