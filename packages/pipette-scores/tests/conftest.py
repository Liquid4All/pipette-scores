import json

import pytest
import datasets


def _write_parquet(path, rows):
    datasets.Dataset.from_list(rows).to_parquet(str(path))


def _ifstruct_row(prompt):
    return {
        "seed": 0,
        "entity_type": "test_entity",
        "prompt": prompt,
        "json_schema": {"type": "object"},
        "top_level_count": None,
        "top_level_key": None,
        "require_wrapper_key": False,
        "require_code_block": False,
        "require_no_commentary": False,
        "output_format": "json",
    }


@pytest.fixture
def ifstruct_dataset_dir(tmp_path):
    d = tmp_path / "datasets" / "ifstruct" / "test_ds"
    d.mkdir(parents=True)
    rows = [_ifstruct_row("Emit a JSON object."), _ifstruct_row("Emit a different JSON object.")]
    (d / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return d


_IFBENCH_ROWS = [
    {
        "key": "0",
        "instruction_id_list": ["punctuation:no_comma"],
        "prompt": "Write a sentence without commas.",
        "kwargs": [{"num_words": 5}],
        "n_constraints": 1,
    },
    {
        "key": "1",
        "instruction_id_list": ["punctuation:no_comma"],
        "prompt": "Describe the sky without commas.",
        "kwargs": [{"num_words": 5}],
        "n_constraints": 1,
    },
]


@pytest.fixture
def ifbench_dataset_dir(tmp_path):
    d = tmp_path / "datasets" / "ifbench" / "test_ds"
    d.mkdir(parents=True)
    _write_parquet(d / "train.parquet", _IFBENCH_ROWS)
    return d


_GPQA_ROWS = [
    {"prompt": "Q1?\nA) a\nB) b\nC) c\nD) d", "answer": "B"},
    {"prompt": "Q2?\nA) a\nB) b\nC) c\nD) d", "answer": "D"},
]


@pytest.fixture
def gpqa_dataset_dir(tmp_path):
    # GPQA parquet is gitignored, so the test materializes a tiny stand-in with a
    # matching content_sha256 (the loader verifies it).
    from pipette_scores.dataset_catalog.gpqa_diamond import content_sha256

    d = tmp_path / "datasets" / "gpqa_diamond" / "test_ds"
    d.mkdir(parents=True)
    _write_parquet(d / "test.parquet", _GPQA_ROWS)
    (d / "metadata.json").write_text(
        json.dumps(
            {
                "repeats": 1,
                "content_sha256": content_sha256([(r["prompt"], r["answer"]) for r in _GPQA_ROWS]),
            }
        )
    )
    return d


_MATH_500_ROWS = [
    {"prompt": "What is 2+2?\n\nPut your final answer within \\boxed{}.", "answer": "4"},
    {"prompt": "What is 3*3?\n\nPut your final answer within \\boxed{}.", "answer": "9"},
]


@pytest.fixture
def math_500_dataset_dir(tmp_path):
    from pipette_scores.dataset_catalog.math_500 import content_sha256

    d = tmp_path / "datasets" / "math_500" / "test_ds"
    d.mkdir(parents=True)
    _write_parquet(d / "test.parquet", _MATH_500_ROWS)
    (d / "metadata.json").write_text(
        json.dumps(
            {
                "repeats": 1,
                "content_sha256": content_sha256([(r["prompt"], r["answer"]) for r in _MATH_500_ROWS]),
            }
        )
    )
    return d
