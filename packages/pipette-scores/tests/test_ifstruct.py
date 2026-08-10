from pipette_scores.scoring.ifstruct import score
from pipette_scores.types import IFStructSample, SampleCompletion


_JSON_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
}


def _sample(
    id: str = "1",
    json_schema: dict = _JSON_SCHEMA,
    entity_type: str = "json",
    output_format: str = "json",
    top_level_count: int | list[int] | None = None,
    top_level_key: str | None = None,
    require_wrapper_key: bool = False,
    require_code_block: bool = False,
    require_no_commentary: bool = False,
) -> IFStructSample:
    return IFStructSample(
        id=id,
        seed=1,
        prompt="Generate a JSON object",
        json_schema=json_schema,
        top_level_count=top_level_count,
        top_level_key=top_level_key,
        require_wrapper_key=require_wrapper_key,
        require_code_block=require_code_block,
        require_no_commentary=require_no_commentary,
        output_format=output_format,
        entity_type=entity_type,
    )


def test_score_correct_response():
    samples = [_sample("1")]
    completions = [SampleCompletion(id="1", completion='{"name": "Alice"}')]
    scores, _ = score(completions, samples)
    assert scores[0].is_correct is True


def test_score_incorrect_response():
    samples = [_sample("1")]
    completions = [SampleCompletion(id="1", completion="This is not JSON at all")]
    scores, _ = score(completions, samples)
    assert scores[0].is_correct is False


def test_thinking_tags_stripped_before_validation():
    # validate_response strips reasoning tags before judging, so a <think>
    # block is removed rather than counted as commentary — the JSON passes
    # even when the sample requires no commentary.
    samples = [_sample("1", require_no_commentary=True)]
    completions = [
        SampleCompletion(
            id="1",
            completion='<think>Let me generate JSON.</think>{"name": "Alice"}',
        )
    ]
    scores, _ = score(completions, samples)
    assert scores[0].is_correct is True


def test_thinking_tags_allowed_when_commentary_permitted():
    # Same completion, but the sample does not require no-commentary, so the
    # surrounding <think> prose is ignored and only the JSON content is judged.
    samples = [_sample("1", require_no_commentary=False)]
    completions = [
        SampleCompletion(
            id="1",
            completion='<think>Let me generate JSON.</think>{"name": "Alice"}',
        )
    ]
    scores, _ = score(completions, samples)
    assert scores[0].is_correct is True


def test_zero_samples():
    scores, _ = score([], [])
    assert scores == []


def test_score_context_has_entity_accuracy():
    samples = [_sample("1")]
    completions = [SampleCompletion(id="1", completion='{"name": "Alice"}')]
    _, context = score(completions, samples)
    assert context["by_entity_type"]["json"]["pass_rate"] == 1.0


def test_score_context_categorizes_errors():
    samples = [_sample("1")]
    completions = [SampleCompletion(id="1", completion="not json at all")]
    _, context = score(completions, samples)
    # Plain text should fall into the "no valid JSON" category specifically,
    # not just any error bucket.
    assert context["common_errors"].get("no valid JSON", 0) >= 1, context


def test_score_context_shape_matches_build_summary():
    samples = [_sample("1")]
    completions = [SampleCompletion(id="1", completion='{"name": "Alice"}')]
    _, context = score(completions, samples)
    # build_summary() is the reference aggregator from vendor/ifstruct —
    # we expose its output verbatim as context.
    expected_keys = {
        "passed",
        "total",
        "pass_rate",
        "average_latency_ms",
        "by_format",
        "by_top_level_structure",
        "by_entity_type",
        "common_errors",
    }
    assert expected_keys <= set(context), context
