import pytest

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
        "ifstruct_blanked_truncated",
        "ifstruct_pass_at_1_blanked",
    }
    assert expected_keys <= set(context), context


@pytest.mark.parametrize(
    "completion",
    [
        # Cohere North on vLLM, thinking off: its text section wraps the JSON.
        '<|START_TEXT|>{"name": "Alice"}<|END_TEXT|>',
        # North, thinking on.
        'Only a name is needed.<|END_THINKING|><|START_TEXT|>{"name": "Alice"}<|END_TEXT|>',
        # ERNIE-4.5-Thinking: the prompt opened <think>, the answer is wrapped.
        'Only a name is needed.\n</think>\n\n<response>\n{"name": "Alice"}\n</response>\n',
        # ERNIE behind vLLM's ernie45 parser: the reasoning is gone, the wrapper stays.
        '\n\n<response>\n```json\n{"name": "Alice"}\n```\n</response>\n',
        # Gemma 4 on llama.cpp.
        '<|channel>thought\nOnly a name is needed.<channel|>{"name": "Alice"}',
    ],
)
def test_reasoning_markup_is_not_commentary(completion):
    samples = [_sample("1", require_no_commentary=True)]
    scores, _ = score([SampleCompletion(id="1", completion=completion)], samples)
    assert scores[0].is_correct is True


def test_reasoning_is_stripped_once():
    # The validator strips thinking tags itself. Stripping before it as well
    # would cut again at the </think> inside the JSON string.
    completions = [SampleCompletion(id="1", completion='Plan.</think>{"name": "a </think> b"}')]
    scores, _ = score(completions, [_sample("1")])
    assert scores[0].is_correct is True


@pytest.mark.parametrize("stop_reason", ["truncated", "doom_loop"])
@pytest.mark.parametrize("enable_thinking, blanked", [(True, 1), (False, 0), (None, 0)])
def test_cut_reasoning_is_graded_raw_and_counted(stop_reason, enable_thinking, blanked):
    # The prompt opened the reasoning and generation was cut after a draft:
    # graded whole, as liquid_evals grades it.
    completion = 'Draft:\n```json\n{"name": "Alice"}\n```\nNow check whether the'
    completions = [SampleCompletion(id="1", completion=completion, stop_reason=stop_reason)]
    scores, context = score(completions, [_sample("1")], enable_thinking=enable_thinking)
    assert scores[0].is_correct is True
    assert context["pass_rate"] == 1.0
    # The strict view counts it wrong.
    assert context["ifstruct_blanked_truncated"] == blanked
    assert context["ifstruct_pass_at_1_blanked"] == (0.0 if blanked else 1.0)


def test_cut_reasoning_opened_in_text_fails():
    # The completion opened <think> itself: everything from the tag on goes.
    completion = '<think>Draft:\n```json\n{"name": "Alice"}\n```\nNow check whether the'
    completions = [SampleCompletion(id="1", completion=completion, stop_reason="truncated")]
    scores, context = score(completions, [_sample("1")], enable_thinking=True)
    assert scores[0].is_correct is False
    assert context["ifstruct_blanked_truncated"] == 1
    assert context["ifstruct_pass_at_1_blanked"] == 0.0


def test_runaway_nesting_fails_only_its_own_completion():
    samples = [_sample("1"), _sample("2")]
    completions = [
        SampleCompletion(id="1", completion='{"name": "Alice"}'),
        SampleCompletion(id="2", completion='{"name": [' + '{"name": [' * 20000),
    ]
    scores, context = score(completions, samples)
    assert {s.id: s.is_correct for s in scores} == {"1": True, "2": False}
    assert context["common_errors"].get("nesting too deep") == 1, context
