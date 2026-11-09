"""ASR sets split into ordered git-LFS parts: read as one table, the fingerprint unchanged."""

from __future__ import annotations

import importlib.util
import io
import itertools
import json
import pathlib
import uuid
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from pipette_scores.dataset_catalog import _asr_build, _asr_common, asr_datasets
from pipette_scores.types import EvalId

RATE = 16000
EVAL = EvalId.ASR_FLEURS_NL_NL
_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def _script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wav(seconds: float, seed: int) -> bytes:
    """Noise, so the parquet's size is its audio (as a real set's is), not its footer."""
    noise = 0.1 * np.random.default_rng(seed).standard_normal(int(seconds * RATE)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, noise, RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _build(tmp_path: pathlib.Path, n: int) -> pathlib.Path:
    """A fake built set of ``n`` clips under tmp_path/datasets, uniquely named
    (the loaders cache per directory)."""
    rows = [
        {
            "audio": _wav(0.05 + 0.001 * i, i),
            "audio_format": "wav",
            "reference": f"w{i}",
            "session": None,
            "chunk_index": None,
            "raw_doc": {"id": f"utt-{i}"},
        }
        for i in range(n)
    ]
    name = f"test-{uuid.uuid4().hex[:8]}"
    _asr_build.build(asr_datasets.ASR_DATASETS[EVAL], tmp_path / "datasets", name, rows=iter(rows))
    return tmp_path / "datasets" / EVAL.value / name


def _split(tmp_path: pathlib.Path, n: int = 3 * _asr_build.ROW_GROUP_SIZE + 5) -> tuple[pathlib.Path, pathlib.Path]:
    """A built set (4 row groups by default) and a copy split into three or more
    parts beside it, under another version name, its metadata.json copied as is."""
    single = _build(tmp_path, n)
    split = single.with_name(f"{single.name}-split")
    bound = (single / "test.parquet").stat().st_size // 2
    files = _asr_common.split_parts(single / "test.parquet", split, bound)
    (split / "metadata.json").write_bytes((single / "metadata.json").read_bytes())
    assert len(files) >= 3 and all(f.stat().st_size <= bound for f in files)
    return single, split


def _clips(target: pathlib.Path, start: int, stop: int) -> list[tuple[str, str]]:
    """Rows ``start`` to ``stop`` of the served stream, as (id, clip)."""
    served = itertools.islice(_asr_common.iter_prompt_samples(EVAL, target, "lfm3"), start, stop)
    return [(s.id, s.messages[-1].content[-1]["input_audio"]["data"]) for s in served]


def test_a_split_set_is_the_same_rows_with_the_same_fingerprint(tmp_path):
    single, split = _split(tmp_path)
    files = _asr_common.parquet_files(split)
    count = len(files)
    assert [f.name for f in files] == [f"test-{i:05d}-of-{count:05d}.parquet" for i in range(count)]
    # Cut between row groups, in order: the parts' groups are the build's.
    metas = [pq.ParquetFile(f).metadata for f in files]
    assert [m.row_group(i).num_rows for m in metas for i in range(m.num_row_groups)] == [64, 64, 64, 5]
    rows = _asr_common.verified_rows(split)  # the server's checks, content_sha256 included
    assert rows == _asr_common.verified_rows(single)
    assert _asr_common.content_sha256(rows) == json.loads((split / "metadata.json").read_text())["content_sha256"]
    assert _asr_common.load_eval_samples(split) == _asr_common.load_eval_samples(single)
    # A stretch across a cut serves the same clips, and so does the whole set.
    first = metas[0].num_rows
    assert _clips(split, first - 2, first + 3) == _clips(single, first - 2, first + 3)
    assert _clips(split, 0, 256) == _clips(single, 0, 256)


def test_a_set_under_the_bound_is_copied_whole(tmp_path):
    single = _build(tmp_path, 2)
    [file] = _asr_common.split_parts(single / "test.parquet", tmp_path / "copy", 10**9)
    assert file.name == "test.parquet"
    assert pq.read_table(file).to_pylist() == pq.read_table(single / "test.parquet").to_pylist()
    with pytest.raises(ValueError, match="another directory"):
        _asr_common.split_parts(single / "test.parquet", single, 10**9)


def test_a_row_group_over_the_bound_cannot_be_split(tmp_path):
    single = _build(tmp_path, 3)
    with pytest.raises(ValueError, match="cannot be cut"):
        _asr_common.split_parts(single / "test.parquet", tmp_path / "out", 1000)
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_a_missing_part_is_refused_even_renumbered(tmp_path):
    """The negative control: the layout names a missing part, and a set
    renumbered to hide one keeps a valid layout but not the fingerprint."""
    _, split = _split(tmp_path)
    files = _asr_common.parquet_files(split)
    files[-1].unlink()
    with pytest.raises(_asr_common.AsrDatasetError, match="for every i from 0"):
        _asr_common.parquet_files(split)
    for i, f in enumerate(files[:-1]):
        f.rename(split / _asr_common.part_name(i, len(files) - 1))
    with pytest.raises(_asr_common.AsrDatasetError, match=r"content_sha256 \w+ != \w+ from metadata.json"):
        _asr_common.verified_rows(split)


def test_parts_out_of_order_are_refused(tmp_path):
    """The rows' digest is order-free, so swapped parts keep it; their ids do not."""
    _, split = _split(tmp_path)
    a, b = _asr_common.parquet_files(split)[:2]
    a.rename(split / "swap")
    b.rename(a)
    (split / "swap").rename(b)
    with pytest.raises(_asr_common.AsrDatasetError, match="must be its position"):
        _asr_common.verified_rows(split)


@pytest.mark.parametrize(
    "stray, match",
    [
        ("test.parquet", "keep one layout"),
        ("test-extra.parquet", r"not a test-<i>-of-<n>\.parquet part"),
        ("test-00000-of-00009.parquet", "for every i from 0"),
    ],
)
def test_a_mixed_or_stray_layout_is_refused(tmp_path, stray, match):
    single, split = _split(tmp_path)
    (split / stray).write_bytes((single / "test.parquet").read_bytes())
    with pytest.raises(_asr_common.AsrDatasetError, match=match):
        _asr_common.verified_rows(split)


def test_a_dataset_with_no_parquet_is_not_found(tmp_path):
    single = _build(tmp_path, 1)
    (single / "test.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        _asr_common.parquet_files(single)


def test_verify_checks_a_split_set_and_fails_a_missing_part(tmp_path, monkeypatch, capsys):
    _, split = _split(tmp_path)
    verify = _script("verify_asr_datasets")
    monkeypatch.setattr("sys.argv", ["verify", "--root", str(tmp_path / "datasets"), "--version", split.name])
    assert verify.main() == 0
    out = capsys.readouterr().out
    assert f"{EVAL.value}: ok (197 clips in {len(_asr_common.parquet_files(split))} parts, 0 audio mismatches" in out
    _asr_common.parquet_files(split)[1].unlink()
    assert verify.main() == 1
    assert f"{EVAL.value}: FAILED" in capsys.readouterr().out


def test_the_split_script_lays_out_a_tree_and_checks_it(tmp_path, monkeypatch, capsys):
    single = _build(tmp_path, 2 * _asr_build.ROW_GROUP_SIZE)
    bound = (single / "test.parquet").stat().st_size * 3 // 4
    out = tmp_path / "lfs"
    argv = ["split", "--src", str(tmp_path / "datasets"), "--out", str(out), "--version", single.name]
    monkeypatch.setattr("sys.argv", [*argv, "--max-part-bytes", str(bound), EVAL.value])
    assert _script("split_asr_dataset").main() == 0
    target = out / EVAL.value / single.name
    assert sorted(p.name for p in target.iterdir()) == [
        "metadata.json",
        "test-00000-of-00002.parquet",
        "test-00001-of-00002.parquet",
    ]
    assert (target / "metadata.json").read_bytes() == (single / "metadata.json").read_bytes()
    assert f"{EVAL.value}: 128 clips in 2 file(s)" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="not empty"):
        _script("split_asr_dataset").main()


def test_neither_scoring_nor_the_stream_holds_a_sets_audio(tmp_path, monkeypatch):
    """The integrity load (what scoring uses) reads no audio column, and the
    samples stream reads one row group at a time as it goes, so a 9 GB set is
    never in memory."""
    _, split = _split(tmp_path)
    tables: list[Any] = []
    read_table = pq.read_table

    def spy_table(path, columns=None, **kw):
        tables.append(columns)
        return read_table(path, columns=columns, **kw)

    monkeypatch.setattr(pq, "read_table", spy_table)
    _asr_common.load_eval_samples(split)
    assert tables and all(c is not None and "audio" not in c for c in tables)
    groups: list[tuple[Any, int]] = []
    read_row_group = pq.ParquetFile.read_row_group

    def spy_group(self, i, columns=None, **kw):
        groups.append((columns, i))
        return read_row_group(self, i, columns=columns, **kw)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", spy_group)
    served = list(itertools.islice(_asr_common.iter_prompt_samples(EVAL, split, "lfm3"), 3))
    assert len(served) == 3 and len(groups) == 1 and groups[0][0] == ["audio"]
