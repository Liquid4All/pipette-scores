import json

import pytest

from pipette_scores.dataset_catalog import list_dataset_names, load_eval_samples, load_prompt_samples
from pipette_scores.repeats import logical_id
from pipette_scores.types import EvalId, EvalSample, IFBenchSample


def test_list_dataset_names_empty(tmp_path):
    assert list_dataset_names(EvalId.IFBENCH, root_dir=tmp_path) == []


def test_list_dataset_names_returns_sorted(tmp_path):
    eval_dir = tmp_path / "datasets" / "ifbench"
    (eval_dir / "beta").mkdir(parents=True)
    (eval_dir / "alpha").mkdir(parents=True)
    result = list_dataset_names(EvalId.IFBENCH, root_dir=tmp_path)
    assert result == ["alpha", "beta"]


def test_load_eval_samples_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_eval_samples(EvalId.IFBENCH, "nonexistent", root_dir=tmp_path)


def test_load_eval_samples_ifbench(ifbench_dataset_dir):
    root = ifbench_dataset_dir.parent.parent.parent
    samples = load_eval_samples(EvalId.IFBENCH, "test_ds", root_dir=root)
    assert len(samples) == 2
    assert all(isinstance(s, IFBenchSample) for s in samples)
    assert samples[0].prompt == "Write a sentence without commas."
    assert samples[0].instruction_id_list == ("punctuation:no_comma",)


def test_load_prompt_samples_ifbench(ifbench_dataset_dir):
    root = ifbench_dataset_dir.parent.parent.parent
    prompts = load_prompt_samples(EvalId.IFBENCH, "test_ds", root_dir=root)
    assert len(prompts) == 2
    assert all(isinstance(p, EvalSample) for p in prompts)
    assert prompts[0].messages[0].role == "user"
    assert prompts[0].messages[0].content == "Write a sentence without commas."


def test_load_samples_unique_ids(ifbench_dataset_dir):
    root = ifbench_dataset_dir.parent.parent.parent
    samples = load_eval_samples(EvalId.IFBENCH, "test_ds", root_dir=root)
    assert samples[0].id != samples[1].id


def test_gpqa_loads_and_verifies_integrity(gpqa_dataset_dir):
    root = gpqa_dataset_dir.parent.parent.parent
    samples = load_eval_samples(EvalId.GPQA_DIAMOND, "test_ds", root_dir=root)
    assert len(samples) == 2
    assert {s.answer for s in samples} == {"B", "D"}


def test_gpqa_missing_parquet_is_actionable(tmp_path):
    # Dir + metadata exist but the (gitignored) parquet doesn't — the error must
    # tell the operator how to materialize it.
    d = tmp_path / "datasets" / "gpqa_diamond" / "test_ds"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"repeats": 1}))
    with pytest.raises(FileNotFoundError, match="build_gpqa_diamond_dataset"):
        load_eval_samples(EvalId.GPQA_DIAMOND, "test_ds", root_dir=tmp_path)


def test_gpqa_integrity_mismatch_raises(gpqa_dataset_dir):
    meta = json.loads((gpqa_dataset_dir / "metadata.json").read_text())
    meta["content_sha256"] = "deadbeef"
    (gpqa_dataset_dir / "metadata.json").write_text(json.dumps(meta))
    root = gpqa_dataset_dir.parent.parent.parent
    with pytest.raises(ValueError, match="integrity check failed"):
        load_eval_samples(EvalId.GPQA_DIAMOND, "test_ds", root_dir=root)


def test_math_500_loads_and_verifies_integrity(math_500_dataset_dir):
    root = math_500_dataset_dir.parent.parent.parent
    samples = load_eval_samples(EvalId.MATH_500, "test_ds", root_dir=root)
    assert len(samples) == 2
    assert {s.answer for s in samples} == {"4", "9"}


def test_math_500_missing_parquet_is_actionable(tmp_path):
    d = tmp_path / "datasets" / "math_500" / "test_ds"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"repeats": 1}))
    with pytest.raises(FileNotFoundError, match="build_math_500_dataset"):
        load_eval_samples(EvalId.MATH_500, "test_ds", root_dir=tmp_path)


def test_math_500_integrity_mismatch_raises(math_500_dataset_dir):
    meta = json.loads((math_500_dataset_dir / "metadata.json").read_text())
    meta["content_sha256"] = "deadbeef"
    (math_500_dataset_dir / "metadata.json").write_text(json.dumps(meta))
    root = math_500_dataset_dir.parent.parent.parent
    with pytest.raises(ValueError, match="integrity check failed"):
        load_eval_samples(EvalId.MATH_500, "test_ds", root_dir=root)


def test_ifbench_loads_and_verifies_integrity(ifbench_dataset_dir):
    from pipette_scores.dataset_catalog.ifbench import content_sha256
    from pipette_scores.parquet_io import read_parquet

    rows = read_parquet(ifbench_dataset_dir / "train.parquet")
    (ifbench_dataset_dir / "metadata.json").write_text(json.dumps({"content_sha256": content_sha256(rows)}))
    root = ifbench_dataset_dir.parent.parent.parent
    samples = load_eval_samples(EvalId.IFBENCH, "test_ds", root_dir=root)
    assert len(samples) == 2


def test_ifbench_missing_parquet_is_actionable(tmp_path):
    d = tmp_path / "datasets" / "ifbench" / "test_ds"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({}))
    with pytest.raises(FileNotFoundError, match="build_ifbench_dataset"):
        load_eval_samples(EvalId.IFBENCH, "test_ds", root_dir=tmp_path)


def test_ifbench_integrity_mismatch_raises(ifbench_dataset_dir):
    (ifbench_dataset_dir / "metadata.json").write_text(json.dumps({"content_sha256": "deadbeef"}))
    root = ifbench_dataset_dir.parent.parent.parent
    with pytest.raises(ValueError, match="integrity check failed"):
        load_eval_samples(EvalId.IFBENCH, "test_ds", root_dir=root)


def _set_repeats(dataset_dir, repeats):
    (dataset_dir / "metadata.json").write_text(json.dumps({"repeats": repeats}))


def test_repeats_expands_both_loaders(ifbench_dataset_dir):
    root = ifbench_dataset_dir.parent.parent.parent
    _set_repeats(ifbench_dataset_dir, 3)

    eval_samples = load_eval_samples(EvalId.IFBENCH, "test_ds", root_dir=root)
    prompts = load_prompt_samples(EvalId.IFBENCH, "test_ds", root_dir=root)

    # 2 logical samples × 3 repeats = 6 unique #k ids, the same ids on both sides.
    eval_ids = [s.id for s in eval_samples]
    prompt_ids = [p.id for p in prompts]
    assert len(eval_ids) == len(prompt_ids) == 6
    assert len(set(eval_ids)) == 6
    assert eval_ids == prompt_ids
    assert {logical_id(i) for i in eval_ids} == {logical_id(i) for i in prompt_ids}
    assert len({logical_id(i) for i in eval_ids}) == 2
    assert all("#" in i for i in eval_ids)


def test_repeats_one_is_noop(ifbench_dataset_dir):
    root = ifbench_dataset_dir.parent.parent.parent
    _set_repeats(ifbench_dataset_dir, 1)
    samples = load_eval_samples(EvalId.IFBENCH, "test_ds", root_dir=root)
    assert len(samples) == 2
    assert all("#" not in s.id for s in samples)


def test_missing_metadata_is_noop(ifbench_dataset_dir):
    # No metadata.json at all → repeats defaults to 1, ids unchanged.
    root = ifbench_dataset_dir.parent.parent.parent
    samples = load_eval_samples(EvalId.IFBENCH, "test_ds", root_dir=root)
    assert len(samples) == 2
    assert all("#" not in s.id for s in samples)


def test_invalid_repeats_raises(ifbench_dataset_dir):
    root = ifbench_dataset_dir.parent.parent.parent
    _set_repeats(ifbench_dataset_dir, 0)
    with pytest.raises(ValueError):
        load_eval_samples(EvalId.IFBENCH, "test_ds", root_dir=root)


def test_ifstruct_repeats_one_is_noop(ifstruct_dataset_dir):
    # IFStruct ships with repeats:1; the loaders honor it as a no-op (no #k ids).
    root = ifstruct_dataset_dir.parent.parent.parent
    _set_repeats(ifstruct_dataset_dir, 1)
    eval_samples = load_eval_samples(EvalId.IFSTRUCT, "test_ds", root_dir=root)
    prompts = load_prompt_samples(EvalId.IFSTRUCT, "test_ds", root_dir=root)
    assert len(eval_samples) == len(prompts) == 2
    assert all("#" not in s.id for s in eval_samples)


def test_ifstruct_repeats_expand_both_loaders(ifstruct_dataset_dir):
    # The ifstruct loaders honor repeats > 1 via the same shared helper as ifbench.
    root = ifstruct_dataset_dir.parent.parent.parent
    _set_repeats(ifstruct_dataset_dir, 3)
    eval_ids = [s.id for s in load_eval_samples(EvalId.IFSTRUCT, "test_ds", root_dir=root)]
    prompt_ids = [p.id for p in load_prompt_samples(EvalId.IFSTRUCT, "test_ds", root_dir=root)]
    assert len(eval_ids) == len(prompt_ids) == 6
    assert len(set(eval_ids)) == 6
    assert eval_ids == prompt_ids
    assert len({logical_id(i) for i in eval_ids}) == 2
