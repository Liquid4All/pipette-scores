import pytest

import pipette_scores.scoring as scoring
from pipette_scores.types import (
    EvalId,
    GPQADiamondSample,
    IFBenchSample,
    IFStructSample,
    Math500Sample,
    SampleCompletion,
)


def test_unknown_eval_raises():
    with pytest.raises((KeyError, ValueError)):
        scoring.score("nonexistent", [], [])


def test_dispatch_ifbench():
    samples = [
        IFBenchSample(
            id="1",
            key="1",
            instruction_id_list=("count:keywords_multiple",),
            prompt="Use the words banana once, apple twice, cherry three times, date five times, and fig seven times.",
            kwargs=(
                {
                    "keyword1": "banana",
                    "keyword2": "apple",
                    "keyword3": "cherry",
                    "keyword4": "date",
                    "keyword5": "fig",
                },
            ),
            n_constraints=1,
        )
    ]
    completion = " ".join(["banana"] * 1 + ["apple"] * 2 + ["cherry"] * 3 + ["date"] * 5 + ["fig"] * 7)
    completions = [SampleCompletion(id="1", completion=completion)]
    scores, _ = scoring.score("ifbench", completions, samples)
    assert len(scores) == 1
    assert scores[0].is_correct is True


def test_dispatch_ifstruct():
    samples = [
        IFStructSample(
            id="1",
            seed=1,
            prompt="Generate JSON",
            json_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            top_level_count=None,
            top_level_key=None,
            require_wrapper_key=False,
            require_code_block=False,
            require_no_commentary=False,
            output_format="json",
            entity_type="json",
        )
    ]
    completions = [SampleCompletion(id="1", completion='{"name": "Alice"}')]
    scores, _ = scoring.score("ifstruct", completions, samples)
    assert len(scores) == 1
    assert scores[0].is_correct is True


def test_dispatch_passes_enable_thinking():
    samples = [
        IFStructSample(
            id="1",
            seed=1,
            prompt="Generate JSON",
            json_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            top_level_count=None,
            top_level_key=None,
            require_wrapper_key=False,
            require_code_block=False,
            require_no_commentary=False,
            output_format="json",
            entity_type="json",
        )
    ]
    completion = 'Draft:\n```json\n{"name": "Alice"}\n```\nNow check whether the'
    completions = [SampleCompletion(id="1", completion=completion, stop_reason="truncated")]
    scores, context = scoring.score("ifstruct", completions, samples, enable_thinking=True)
    assert scores[0].is_correct is True
    assert context["ifstruct_blanked_truncated"] == 1
    assert context["ifstruct_pass_at_1_blanked"] == 0.0


_KW = {"keyword1": "banana", "keyword2": "apple", "keyword3": "cherry", "keyword4": "date", "keyword5": "fig"}

# One sample per eval, the answer that passes it, and reasoning that would fail
# it if it were graded as the answer.
_EVAL_CASES = {
    EvalId.GPQA_DIAMOND: (
        GPQADiamondSample(id="1", prompt="Q\nA) a\nB) b\nC) c\nD) d", answer="B"),
        "The answer is B",
        "Option C looks right. Answer: C",
    ),
    EvalId.MATH_500: (Math500Sample(id="1", prompt="P", answer="42"), "The answer is 42", "Maybe \\boxed{7}."),
    EvalId.IFBENCH: (
        IFBenchSample(
            id="1",
            key="1",
            instruction_id_list=("count:keywords_multiple",),
            prompt="Use banana once, apple twice, cherry three times, date five times, fig seven times.",
            kwargs=(_KW,),
            n_constraints=1,
        ),
        " ".join(["banana"] * 1 + ["apple"] * 2 + ["cherry"] * 3 + ["date"] * 5 + ["fig"] * 7),
        "banana apple cherry date fig, each once to start",
    ),
    EvalId.IFSTRUCT: (
        IFStructSample(
            id="1",
            seed=1,
            prompt="Generate JSON",
            json_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            top_level_count=None,
            top_level_key=None,
            require_wrapper_key=False,
            require_code_block=False,
            require_no_commentary=True,
            output_format="json",
            entity_type="json",
        ),
        '{"name": "Alice"}',
        'A name only, so {"name": 1} is wrong.',
    ),
}

# Every way the scorer reads a closed reasoning block.
_CLOSED_FORMATS = {
    "think": "<think>{r}</think>{a}",
    "prompt-opened": "{r}\n</think>\n\n{a}",
    "mistral": "[THINK]{r}[/THINK]{a}",
    "gemma4": "<|channel>thought\n{r}<channel|>{a}",
    "north": "{r}<|END_THINKING|><|START_TEXT|>{a}<|END_TEXT|>",
    "ernie": "{r}\n</think>\n\n<response>\n{a}\n</response>\n",
    "gpt-oss": "<|channel|>analysis<|message|>{r}<|end|><|start|>assistant<|channel|>final<|message|>{a}<|return|>",
    "gpt-oss-vllm": "analysis{r}assistantfinal{a}",
}


@pytest.mark.parametrize("enable_thinking", [True, False, None])
@pytest.mark.parametrize("stop_reason", ["eos", "truncated", "doom_loop", "unknown", None])
@pytest.mark.parametrize("fmt", list(_CLOSED_FORMATS.values()), ids=list(_CLOSED_FORMATS))
@pytest.mark.parametrize("eval_id", list(_EVAL_CASES))
def test_closed_reasoning_grades_the_same_whatever_the_stop(eval_id, fmt, stop_reason, enable_thinking):
    # Neither this scorer nor 552094f blanks a sample whose reasoning closed, so
    # its verdict is the one it gets with no stop reason and no thinking flag,
    # and the strict view agrees.
    sample, answer, reasoning = _EVAL_CASES[eval_id]
    # The reasoning fails the sample if it is left in.
    unstripped, _ = scoring.score(eval_id, [SampleCompletion(id="1", completion=f"{reasoning} {answer}")], [sample])
    assert unstripped[0].is_correct is False
    completion = fmt.format(r=reasoning, a=answer)
    baseline, _ = scoring.score(eval_id, [SampleCompletion(id="1", completion=completion)], [sample])
    assert baseline[0].is_correct is True
    c = SampleCompletion(id="1", completion=completion, stop_reason=stop_reason)
    scores, context = scoring.score(eval_id, [c], [sample], enable_thinking=enable_thinking)
    assert scores[0].is_correct is True
    assert context[f"{eval_id}_blanked_truncated"] == 0
    assert context[f"{eval_id}_pass_at_1_blanked"] == 1.0


@pytest.mark.parametrize("eval_id", list(_EVAL_CASES))
def test_empty_completion_is_wrong(eval_id):
    # The strict view counts a blanked sample wrong without grading it again;
    # this holds only because every eval fails an empty completion.
    sample, _, _ = _EVAL_CASES[eval_id]
    scores, _ = scoring.score(eval_id, [SampleCompletion(id="1", completion="")], [sample])
    assert scores[0].is_correct is False


def test_list_eval_ids():
    ids = scoring.list_eval_ids()
    assert set(ids) == set(EvalId)
