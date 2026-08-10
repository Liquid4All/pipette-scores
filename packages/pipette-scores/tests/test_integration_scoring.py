"""Integration tests over real datasets.

Loads fixture datasets (and the shipped `datasets/` tree) via the dataset
catalog loaders, builds completions with known-correct or known-incorrect
answers, and asserts the scoring pipeline produces the expected results.
"""

import json
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipette_scores import dataset_catalog, scoring
from pipette_scores.repeats import logical_id
from pipette_scores.types import EvalId, SampleCompletion

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "rep-2026.03.1"

# tests -> pipette-scores -> packages -> <repo root>
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_IFBENCH_SRC = _REPO_ROOT / "vendor" / "ifbench" / "data" / "IFBench_test.jsonl"


def _correct_count(scores):
    return sum(1 for v in scores if v.is_correct)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ifstruct_samples():
    return dataset_catalog.ifstruct.load_eval_samples(FIXTURE_DIR / "ifstruct")


# ---------------------------------------------------------------------------
# ifstruct
# ---------------------------------------------------------------------------


def test_ifstruct_scoring_runs(ifstruct_samples):
    completions = [SampleCompletion(id=s.id, completion="{}") for s in ifstruct_samples]
    scores, _ = scoring.ifstruct.score(completions, ifstruct_samples)
    assert len(scores) == len(ifstruct_samples)


def test_ifstruct_garbage_scores_incorrect(ifstruct_samples):
    completions = [SampleCompletion(id=s.id, completion="This is not JSON at all.") for s in ifstruct_samples]
    scores, _ = scoring.ifstruct.score(completions, ifstruct_samples)
    assert _correct_count(scores) == 0


# ---------------------------------------------------------------------------
# ifbench — shipped 2026.06.1 dataset (repeats: 5)
# ---------------------------------------------------------------------------


def test_ifbench_2026_06_1_expands_to_1500(tmp_path):
    # ifbench's dataset isn't committed (PIP-237) — materialize it from the vendored
    # upstream source the way the builder does, then assert: repeats: 5 expands the
    # 300 logical samples into 1500 unique #k attempt ids, identical on both loaders
    # so served prompts and scoring ground-truth align.
    raw = [json.loads(line) for line in _IFBENCH_SRC.read_text().splitlines() if line.strip()]
    rows = [{c: r[c] for c in ("key", "instruction_id_list", "prompt", "kwargs")} for r in raw]
    d = tmp_path / "datasets" / "ifbench" / "2026.06.1"
    d.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), d / "train.parquet")
    (d / "metadata.json").write_text(json.dumps({"repeats": 5}))

    eval_samples = dataset_catalog.load_eval_samples(EvalId.IFBENCH, "2026.06.1", root_dir=tmp_path)
    prompts = dataset_catalog.load_prompt_samples(EvalId.IFBENCH, "2026.06.1", root_dir=tmp_path)

    eval_ids = [s.id for s in eval_samples]
    assert len(eval_ids) == 1500
    assert len(set(eval_ids)) == 1500
    assert len({logical_id(i) for i in eval_ids}) == 300
    assert eval_ids == [p.id for p in prompts]
