"""The eval packages: mgmt's tar bytes, the refusals, the command and the store."""

import hashlib
import io
import json
import pathlib
import tarfile
from unittest.mock import MagicMock

import boto3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
from botocore.exceptions import ClientError

from pipette_scores import cli, dataset_catalog, packages
from pipette_scores.dataset_catalog import _asr_build, asr_datasets, asr_prompt_profiles, prompt_profiles
from pipette_scores.hashing import samples_digest
from pipette_scores.types import ChatMessage, EvalId, EvalSample
from test_hashing import ASR_SAMPLES, SAMPLES

_PNG = b"\x89PNG\r\n\x1a\n"  # the golden samples' image, iVBORw0KGgo=


def _package(tmp_path: pathlib.Path, samples: list[EvalSample], eval_id: str, profile: str | None = None):
    package = packages.Package(eval_id, "2026.10.1", profile, tmp_path)
    for sample in samples:
        package.add(sample)
    return package


@pytest.mark.parametrize(
    "samples, eval_id, profile, identity, digest",
    [
        (
            SAMPLES,
            "mmstar",
            None,
            "5654ef883c378c4325cfe70594545d4a19c094defdf21b32487f9de94a6b4849",
            "cecf36b2e29ae61be4fe20df91f8f0e849a3b718e340e831b469b3fd7db90f73",
        ),
        (
            ASR_SAMPLES,
            "asr_fleurs_nl_nl",
            "voxtral",
            "b49fae68dfd0d735e59dc3df621d8b6189ec7977fc5830b54b368bd10f396ef7",
            "b36109aa457771c0273adefb6c467dab0a2c104b376818539da5eb9a3199f550",
        ),
    ],
)
def test_the_golden_samples_pack_to_the_tar_mgmt_pins(tmp_path, samples, eval_id, profile, identity, digest):
    # mgmt's plan-types and format::write_corpus pack the same samples to these bytes.
    package = _package(tmp_path, samples, eval_id, profile)
    tar = b"".join(package.tar())
    assert package.identity == identity
    assert (hashlib.sha256(tar).hexdigest(), len(tar)) == (digest, 4096)


def test_a_package_reads_back_as_a_tar(tmp_path):
    jpeg = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"}}
    png = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
    samples = [*SAMPLES, EvalSample(id="c3", messages=[ChatMessage(role="user", content=[jpeg, png])])]
    package = _package(tmp_path, samples, "mmstar")
    with tarfile.open(fileobj=io.BytesIO(b"".join(package.tar()))) as tar:
        members = tar.getmembers()
        files = {m.name: tar.extractfile(m).read() for m in members}
    names = [m.name for m in members]
    assert names[0] == "manifest.json" and names[-1] == "samples.jsonl" and names == sorted(names)
    assert {(m.mode, m.uid, m.gid, m.mtime) for m in members} == {(0o644, 0, 0, 0)}
    assert json.loads(files["manifest.json"]) == {
        "format": "vlm-eval-corpus/1",
        "adapter": "evals-server@1",
        "upstream": {
            "source": "evals_server",
            "eval_id": "mmstar",
            "dataset_version": "2026.10.1",
            "identity": package.identity,
        },
        "samples": 3,
        "media": 2,
    }
    # The PNG two samples share is one file; each file is named by its sha256.
    media = {name: body for name, body in files.items() if name.startswith("media/")}
    assert sorted(media) == sorted(
        f"media/{hashlib.sha256(body).hexdigest()}.{ext}"
        for body, ext in [(_PNG, "png"), (b"\xff\xd8\xff\xe0\x00\x10", "jpg")]
    )
    lines = [json.loads(line) for line in files["samples.jsonl"].splitlines()]
    assert [line["id"] for line in lines] == ["a1", "b2", "c3"]
    images = [p for line in lines for m in line["messages"] if isinstance(m["content"], list) for p in m["content"]]
    images = [p for p in images if p["type"] == "image"]
    assert [p["bytes"] for p in images] == [len(media[p["path"]]) for p in images] == [8, 6, 8]


@pytest.mark.parametrize(
    "eval_id, part, refusal",
    [
        ("mmstar", {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgp="}}, "canonical"),
        ("mmstar", {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBO Rw0KGgo="}}, "canonical"),
        ("mmstar", {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}, "is not data:image"),
        ("mmstar", {"type": "image_url", "image_url": {"url": "data:image/svg+xml;base64,PHN2Zz4="}}, "is not data:"),
        ("mmstar", {"type": "image_url", "image_url": {"url": "data:image/png;base64,", "detail": "low"}}, "no place"),
        ("mmstar", {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}, "no place"),
        ("mmstar", {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,"}}, "no place"),
        ("mmstar", {"type": "input_audio", "input_audio": {"data": "UklGRiQAAABXQVZF", "format": "wav"}}, "no place"),
        (
            "asr_fleurs_nl_nl",
            {"type": "input_audio", "input_audio": {"data": "ZkxhQw==", "format": "flac"}},
            "wav only",
        ),
        (
            "asr_fleurs_nl_nl",
            {"type": "input_audio", "input_audio": {"data": "UklGRiQAAABXQVZ", "format": "wav"}},
            "canonical",
        ),
    ],
)
def test_a_part_that_would_not_rebuild_is_refused(tmp_path, eval_id, part, refusal):
    sample = EvalSample(id="x", messages=[ChatMessage(role="user", content=[part])])
    with pytest.raises(ValueError, match=refusal):
        _package(tmp_path, [sample], eval_id)


def test_a_set_with_an_id_twice_or_no_samples_is_refused(tmp_path):
    with pytest.raises(ValueError, match="b2 is served twice"):
        _package(tmp_path, [SAMPLES[1], SAMPLES[1]], "mmstar")
    with pytest.raises(ValueError, match="no samples"):
        _package(tmp_path, [], "mmstar").save(packages.LocalStore(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_a_path_ustar_cannot_hold_is_refused():
    assert len(packages._header("a" * 100, 8**11 - 1)) == 512
    for path, size in [("a" * 101, 0), ("média", 0), ("a", 8**11)]:
        with pytest.raises(ValueError, match="does not fit a ustar header"):
            packages._header(path, size)


# ---------------------------------------------------------------------------
# The command, into a local directory
# ---------------------------------------------------------------------------


def _vlm_dataset(root: pathlib.Path, eval_id: str, name: str) -> None:
    rows = [
        {
            "sample_id": f"s{i}",
            "prompt": question,
            "answer": json.dumps("yes"),
            "raw_doc": json.dumps({"question": question}),
            "num_images": 1,
            "image_0": _PNG,
        }
        for i, question in enumerate(["Is there a dog?", "Is there a cat?"])
    ]
    (root / "datasets" / eval_id / name).mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), root / "datasets" / eval_id / name / "test.parquet")


def _asr_dataset(root: pathlib.Path, name: str) -> None:
    def wav(seconds: float) -> bytes:
        buffer = io.BytesIO()
        sf.write(buffer, np.full(int(seconds * 16000), 0.25, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    rows = [
        {
            "audio": wav(1 + i / 10),
            "audio_format": "wav",
            "reference": reference,
            "session": None,
            "chunk_index": None,
            "raw_doc": {"id": f"u{i}"},
        }
        for i, reference in enumerate(["een", "twee"])
    ]
    _asr_build.build(asr_datasets.ASR_DATASETS[EvalId.ASR_FLEURS_NL_NL], root / "datasets", name, rows=iter(rows))


def _cli(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr("sys.argv", ["pipette-scores", *argv])
    cli.main()


def test_the_command_packs_each_dataset_once(tmp_path, monkeypatch):
    data, out = tmp_path / "data", tmp_path / "out"
    _vlm_dataset(data, "pope", "2026.10.1")
    _asr_dataset(data, "test-1")
    (data / "datasets" / "mmstar" / "broken").mkdir(parents=True)
    (data / "datasets" / "mmstar" / "broken" / "test.parquet").write_bytes(b"not parquet")
    built, build = [], packages.build

    def spy(eval_id, name, *rest):
        built.append((eval_id, name))
        build(eval_id, name, *rest)

    monkeypatch.setattr(packages, "build", spy)
    monkeypatch.delenv("PIPETTE_SCORES_VERSION", raising=False)
    argv = ["packages", "--datasets", str(data), "--packages", str(out)]
    for version in [["--version", "dev"], []]:
        with pytest.raises(SystemExit, match="not dev"):
            _cli(monkeypatch, *argv, *version)

    # A dataset that fails fails the run, once the others are packed.
    with pytest.raises(SystemExit, match="1 datasets failed: mmstar/broken"):
        _cli(monkeypatch, *argv, "--version", "v1")
    assert built == [("mmstar", "broken"), ("pope", "2026.10.1"), ("asr_fleurs_nl_nl", "test-1")]
    pope = json.loads((out / "index/v1/pope/2026.10.1.json").read_text())
    asr = json.loads((out / "index/v1/asr_fleurs_nl_nl/test-1.json").read_text())
    assert [e["prompt_profile"] for e in pope["profiles"]] == [None, *prompt_profiles.PROFILES]
    assert [e["prompt_profile"] for e in asr["profiles"]] == list(asr_prompt_profiles.PROFILES)
    # Each digest is the one /samples serves; the samples' one image is stored once.
    for entry in pope["profiles"]:
        served = dataset_catalog.load_prompt_samples.__wrapped__("pope", "2026.10.1", data, entry["prompt_profile"])
        assert (entry["samples_digest"], entry["sample_count"], entry["package"]["media"]) == (
            samples_digest(served),
            2,
            1,
        )
    for entry in asr["profiles"]:
        streamed = dataset_catalog.iter_asr_prompt_samples(
            EvalId.ASR_FLEURS_NL_NL, "test-1", entry["prompt_profile"], root_dir=data
        )
        assert (entry["samples_digest"], entry["sample_count"]) == (samples_digest(streamed), 2)
    packed = {e["package"]["digest"]: e["package"]["bytes"] for e in pope["profiles"] + asr["profiles"]}
    tars = {tar.name: tar.read_bytes() for tar in (out / "tars").iterdir()}
    assert {name: (hashlib.sha256(tar).hexdigest(), len(tar)) for name, tar in tars.items()} == {
        f"{digest}.tar": (digest, size) for digest, size in packed.items()
    }

    # An indexed dataset is skipped; one packed again finds its tars there and leaves them be.
    (data / "datasets" / "mmstar" / "broken" / "test.parquet").unlink()
    (data / "datasets" / "mmstar" / "broken").rmdir()
    (out / "index/v1/asr_fleurs_nl_nl/test-1.json").unlink()
    stamps = {tar: (tar.stat().st_ino, tar.stat().st_mtime_ns) for tar in (out / "tars").iterdir()}
    built.clear()
    _cli(monkeypatch, *argv, "--version", "v1")
    assert built == [("asr_fleurs_nl_nl", "test-1")]
    assert {tar: (tar.stat().st_ino, tar.stat().st_mtime_ns) for tar in (out / "tars").iterdir()} == stamps
    assert json.loads((out / "index/v1/asr_fleurs_nl_nl/test-1.json").read_text()) == asr


# ---------------------------------------------------------------------------
# The S3 store
# ---------------------------------------------------------------------------


def _s3(monkeypatch):
    client = MagicMock()
    client.exceptions.ClientError = ClientError
    client.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")
    client.create_multipart_upload.return_value = {"UploadId": "u"}
    client.upload_part.side_effect = lambda **kw: {"ETag": f"e{kw['PartNumber']}", "ChecksumCRC32": "c"}
    monkeypatch.setattr(boto3, "client", lambda service: client)
    return packages.S3Store("s3://bucket/scores-packages"), client


def _refused(operation: str) -> ClientError:
    return ClientError({"Error": {"Code": "PreconditionFailed"}}, operation)


def test_a_tar_goes_up_in_parts_completed_only_if_absent(monkeypatch):
    monkeypatch.setattr(packages, "_PART_BYTES", 4)
    store, client = _s3(monkeypatch)
    ids = {"Bucket": "bucket", "Key": "scores-packages/tars/t.tar", "UploadId": "u"}
    assert store.create("tars/t.tar", iter([b"abc", b"defgh", b"ij"]))
    assert [c.kwargs["Body"] for c in client.upload_part.call_args_list] == [b"abcd", b"efgh", b"ij"]
    parts = [{"PartNumber": n, "ETag": f"e{n}", "ChecksumCRC32": "c"} for n in (1, 2, 3)]
    client.complete_multipart_upload.assert_called_once_with(**ids, MultipartUpload={"Parts": parts}, IfNoneMatch="*")
    client.abort_multipart_upload.assert_not_called()

    # Written by another run meanwhile (412): there already, and the parts are given up.
    client.complete_multipart_upload.side_effect = _refused("CompleteMultipartUpload")
    assert not store.create("tars/t.tar", iter([b"abc"]))
    client.abort_multipart_upload.assert_called_once_with(**ids)

    def broken():
        yield b"abcd"
        raise OSError("read failed")

    client.abort_multipart_upload.reset_mock()
    with pytest.raises(OSError, match="read failed"):
        store.create("tars/t.tar", broken())
    client.abort_multipart_upload.assert_called_once_with(**ids)

    # There at the check: nothing is sent.
    client.head_object.side_effect = None
    client.create_multipart_upload.reset_mock()
    assert not store.create("tars/t.tar", iter([b"abc"]))
    client.create_multipart_upload.assert_not_called()


def test_an_index_is_put_once(monkeypatch):
    store, client = _s3(monkeypatch)
    assert store.create("index/v1/mme/2026.10.1.json", b"{}\n")
    client.put_object.assert_called_once_with(
        Bucket="bucket", Key="scores-packages/index/v1/mme/2026.10.1.json", Body=b"{}\n", IfNoneMatch="*"
    )
    client.put_object.side_effect = _refused("PutObject")
    assert not store.create("index/v1/mme/2026.10.1.json", b"{}\n")
    client.create_multipart_upload.assert_not_called()
