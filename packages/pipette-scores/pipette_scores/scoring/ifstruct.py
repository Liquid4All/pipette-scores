from typing import Any

from ifstruct.eval import EvalResult, build_summary
from ifstruct.validator import validate_response

from pipette_scores.scoring.completion_text import grading_text
from pipette_scores.types import IFStructSample, SampleCompletion, SampleScore


def score(
    completions: list[SampleCompletion], samples: list[IFStructSample], *, enable_thinking: bool | None = None
) -> tuple[list[SampleScore], dict[str, Any]]:
    """Validate each completion against its sample's schema.

    The context is the vendored ``build_summary`` (``pass_rate`` over the
    completions received, and its breakdowns) plus ``ifstruct_blanked_truncated``
    (completions cut short inside their reasoning, see ``grading_text``) and
    ``ifstruct_pass_at_1_blanked``, ``pass_rate`` with those counted wrong.
    ifstruct is served once per prompt, so its pass rate is its pass@1.
    """
    by_id = {s.id: s for s in samples}
    scores: list[SampleScore] = []
    results: list[EvalResult] = []
    blanked = 0
    blanked_passed = 0
    for c in completions:
        sample = by_id[c.id]
        # validate_response strips thinking tags itself, so grading_text's
        # output goes in as is; a second strip would cut at a later </think>.
        text, cut = grading_text(c, enable_thinking)
        v = validate_response(
            response=text,
            json_schema=sample.json_schema,
            top_level_count=sample.top_level_count,
            require_no_commentary=sample.require_no_commentary,
            output_format=sample.output_format,
            top_level_key=sample.top_level_key,
            require_wrapper_key=sample.require_wrapper_key,
            require_code_block=sample.require_code_block,
        )
        scores.append(SampleScore(id=c.id, is_correct=v.passed))
        blanked += cut
        blanked_passed += cut and v.passed
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
    summary = build_summary(results)
    total = summary["total"]
    return scores, {
        **summary,
        "ifstruct_blanked_truncated": blanked,
        "ifstruct_pass_at_1_blanked": (summary["passed"] - blanked_passed) / total if total else 0.0,
    }
