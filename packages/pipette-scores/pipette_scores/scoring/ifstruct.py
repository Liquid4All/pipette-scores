from typing import Any

from ifstruct.eval import EvalResult, build_summary
from ifstruct.validator import validate_response

from pipette_scores.types import IFStructSample, SampleCompletion, SampleScore


def score(
    completions: list[SampleCompletion], samples: list[IFStructSample]
) -> tuple[list[SampleScore], dict[str, Any]]:
    by_id = {s.id: s for s in samples}
    scores: list[SampleScore] = []
    results: list[EvalResult] = []
    for c in completions:
        sample = by_id[c.id]
        v = validate_response(
            response=c.completion,
            json_schema=sample.json_schema,
            top_level_count=sample.top_level_count,
            require_no_commentary=sample.require_no_commentary,
            output_format=sample.output_format,
            top_level_key=sample.top_level_key,
            require_wrapper_key=sample.require_wrapper_key,
            require_code_block=sample.require_code_block,
        )
        scores.append(SampleScore(id=c.id, is_correct=v.passed))
        results.append(
            EvalResult(
                seed=sample.seed,
                model="",
                passed=v.passed,
                score=v.score,
                errors=v.errors,
                details=v.details,
                prompt=sample.prompt,
                response=c.completion,
                latency_ms=0.0,
                output_format=sample.output_format,
                entity_type=sample.entity_type,
                require_wrapper_key=sample.require_wrapper_key,
            )
        )
    return scores, build_summary(results)
