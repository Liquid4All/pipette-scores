import pytest

import pipette_scores.scoring as scoring
from pipette_scores.types import (
    EvalId,
    IFBenchSample,
    IFStructSample,
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


def test_list_eval_ids():
    ids = scoring.list_eval_ids()
    assert set(ids) == set(EvalId)
