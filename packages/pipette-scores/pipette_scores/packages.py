"""The eval packages: each prompt set ``/samples`` serves for a VLM or ASR eval,
packed into the tar mgmt hands a device, and the write-once store mgmt reads
them from (`pipette-scores packages`, docs/packages.md).

A package is byte for byte the tar mgmt's dataset-cache build packed from
``/samples`` (mgmt ``dataset_cache/format.rs`` ``write_corpus``, plan-types
``vlm_eval_corpus.rs``), so a catalog pin names the same bytes whichever built
them: an uncompressed ustar of ``manifest.json``, every image or clip moved out
of its sample into ``media/<sha256>.<ext>`` (stored once however many samples
share it), and ``samples.jsonl``, every header field fixed.

The store, ``s3://<bucket>/<prefix>`` or a local directory, is written once:

- ``tars/<tar sha256>.tar``
- ``index/<scores version>/<eval_id>/<dataset_version>.json``: every profile's
  samples digest, sample count and package, written once all are stored.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import hashlib
import json
import logging
import os
import pathlib
import re
import tempfile
from collections.abc import Iterable, Iterator
from typing import Any

from pipette_scores import dataset_catalog
from pipette_scores.dataset_catalog import _asr_common, asr_prompt_profiles, prompt_profiles
from pipette_scores.hashing import _AS_SERVED, canonical_sample
from pipette_scores.types import ASR_EVAL_IDS, EvalId, EvalSample

logger = logging.getLogger(__name__)

VLM_FORMAT = "vlm-eval-corpus/1"
ASR_FORMAT = "asr-eval-corpus/1"
ADAPTER = "evals-server@1"

_BLOCK = 512
_PART_BYTES = 64 << 20
# An image's file extension is its subtype (jpg for jpeg), as mgmt's `extension` takes it.
_SUBTYPE = re.compile(r"[a-z0-9]{1,16}")


def profiles(eval_id: str) -> list[str | None]:
    """The prompt profiles ``/samples`` takes for ``eval_id`` (api/app.py
    `_validate_prompt_profile`), None for a VLM set's own prompts. A text eval
    has none, and so no package."""
    if dataset_catalog.is_vlm_eval(eval_id):
        return [None, *(prompt_profiles.PROFILES if eval_id in prompt_profiles.BENCHMARKS else ())]
    if eval_id in ASR_EVAL_IDS:
        return list(asr_prompt_profiles.PROFILES)
    return []


def index_key(scores_version: str, eval_id: str, dataset_version: str) -> str:
    return f"index/{scores_version}/{eval_id}/{dataset_version}.json"


def _compact(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def _decoded(sample_id: str, payload: str) -> bytes:
    """``payload``'s bytes, when it is their canonical base64: only then does a
    client's encoding of them rebuild the string the samples digest hashed."""
    with contextlib.suppress(ValueError):
        data = base64.b64decode(payload, validate=True)
        if base64.b64encode(data).decode("ascii") == payload:
            return data
    raise ValueError(f"sample {sample_id}: a media payload is not canonical base64, so it would not rebuild")


def _header(path: str, size: int) -> bytes:
    """mgmt's ``format.rs`` ``header``: name, size and checksum, the rest fixed."""
    if not (path.isascii() and 1 <= len(path) <= 100) or size >= 8**11:
        raise ValueError(f"{path!r} ({size} bytes) does not fit a ustar header")
    h = bytearray(_BLOCK)
    h[: len(path)] = path.encode("ascii")
    h[100:108] = b"0000644\0"
    h[108:116] = b"0000000\0"
    h[116:124] = b"0000000\0"
    h[124:136] = b"%011o\0" % size
    h[136:148] = b"00000000000\0"
    h[148:156] = b" " * 8
    h[156:157] = b"0"
    h[257:263] = b"ustar\0"
    h[263:265] = b"00"
    h[148:156] = b"%06o\0 " % sum(h)
    return bytes(h)


class Package:
    """One profile's package, built as its samples arrive: their lines in
    memory, the media they name as files under ``scratch`` (each written once,
    shared by the dataset's profiles), and their samples digest."""

    def __init__(self, eval_id: str, dataset_version: str, prompt_profile: str | None, scratch: pathlib.Path):
        self.eval_id, self.dataset_version, self.prompt_profile = eval_id, dataset_version, prompt_profile
        self.format = ASR_FORMAT if eval_id in ASR_EVAL_IDS else VLM_FORMAT
        self._scratch = scratch
        (scratch / "media").mkdir(exist_ok=True)
        self._digest = hashlib.sha256()
        self._lines: list[bytes] = []
        self._media: set[str] = set()
        self._ids: set[str] = set()

    def __str__(self) -> str:
        return f"{self.eval_id}/{self.dataset_version} profile={self.prompt_profile}"

    @property
    def identity(self) -> str:
        """The samples digest's hex, as a catalog pin's ``identity`` holds it."""
        return self._digest.hexdigest()

    def add(self, sample: EvalSample) -> None:
        """The next sample as ``/samples`` serves it, hashed into the samples
        digest and written as its line."""
        if sample.id in self._ids:
            raise ValueError(f"{self}: sample id {sample.id} is served twice")
        self._ids.add(sample.id)
        self._digest.update(canonical_sample(sample))
        self._digest.update(b"\n")
        served = _AS_SERVED.dump_python(sample, mode="json")
        line = {"id": served["id"], "messages": [self._message(served["id"], m) for m in served["messages"]]}
        if served.get("chat_template_kwargs") is not None:
            # mgmt's map is ordered by key, at every depth.
            line["chat_template_kwargs"] = json.loads(json.dumps(served["chat_template_kwargs"], sort_keys=True))
        self._lines.append(_compact(line) + b"\n")

    def _message(self, sample_id: str, message: dict[str, Any]) -> dict[str, Any]:
        content = message["content"]
        if not isinstance(content, str):
            content = [self._part(sample_id, part) for part in content]
        return {"role": message["role"], "content": content}

    def _part(self, sample_id: str, part: dict[str, Any]) -> dict[str, Any]:
        """A served part as the package line holds it (mgmt's ``VlmEvalCorpusPart``):
        text as it is, the format's own media part moved out to a file. A part
        the line would not rebuild exactly is refused."""
        kind = part.get("type")
        media, inner = ("input_audio", {"data", "format"}) if self.format == ASR_FORMAT else ("image_url", {"url"})
        if kind == "text" and part.keys() == {"type", "text"}:
            return {"type": "text", "text": part["text"]}
        if kind != media or part.keys() != {"type", media} or part[media].keys() != inner:
            raise ValueError(f"sample {sample_id}: a {kind!r} part, keys {sorted(part)}, has no place in {self.format}")
        if media == "input_audio":
            if part[media]["format"] != "wav":
                raise ValueError(f"sample {sample_id}: a {part[media]['format']!r} clip; {self.format} holds wav only")
            path, size = self._put(sample_id, part[media]["data"], "wav")
            return {"type": "audio", "path": path, "format": "wav", "bytes": size}
        head, marker, payload = part[media]["url"].partition(";base64,")
        subtype = head.removeprefix("data:image/")
        if not (marker and head.startswith("data:image/") and _SUBTYPE.fullmatch(subtype)):
            raise ValueError(f"sample {sample_id}: image URL {head[:40]!r}... is not data:image/<subtype>;base64,")
        path, size = self._put(sample_id, payload, "jpg" if subtype == "jpeg" else subtype)
        return {"type": "image", "path": path, "mime": f"image/{subtype}", "bytes": size}

    def _put(self, sample_id: str, payload: str, ext: str) -> tuple[str, int]:
        data = _decoded(sample_id, payload)
        path = f"media/{hashlib.sha256(data).hexdigest()}.{ext}"
        file = self._scratch / path
        if not file.exists():
            file.write_bytes(data)
        self._media.add(path)
        return path, len(data)

    def _rebuilt(self, line: bytes) -> EvalSample:
        """The served sample back from its line and media, as a client rebuilds it."""
        sample = json.loads(line)

        def part(p: dict[str, Any]) -> dict[str, Any]:
            if p["type"] == "text":
                return p
            payload = base64.b64encode((self._scratch / p["path"]).read_bytes()).decode("ascii")
            if p["type"] == "image":
                return {"type": "image_url", "image_url": {"url": f"data:{p['mime']};base64,{payload}"}}
            return {"type": "input_audio", "input_audio": {"data": payload, "format": p["format"]}}

        for message in sample["messages"]:
            if not isinstance(message["content"], str):
                message["content"] = [part(p) for p in message["content"]]
        return EvalSample.model_validate(sample)

    def tar(self) -> Iterator[bytes]:
        """The package's bytes as mgmt's ``write_corpus`` writes them: entries
        sorted by path, each padded to a block, then two zero blocks (no padding
        to a 10240-byte record, so not `tarfile`)."""
        upstream = {
            "source": "evals_server",
            "eval_id": self.eval_id,
            "dataset_version": self.dataset_version,
            **({} if self.prompt_profile is None else {"prompt_profile": self.prompt_profile}),
            "identity": self.identity,
        }
        manifest = {
            "format": self.format,
            "adapter": ADAPTER,
            "upstream": upstream,
            "samples": len(self._lines),
            "media": len(self._media),
        }
        entries: dict[str, bytes | pathlib.Path] = {
            "manifest.json": _compact(manifest) + b"\n",
            "samples.jsonl": b"".join(self._lines),
            **{path: self._scratch / path for path in self._media},
        }
        for path in sorted(entries):
            body = entries[path]
            size = len(body) if isinstance(body, bytes) else body.stat().st_size
            yield _header(path, size)
            if isinstance(body, bytes):
                yield body
            else:
                with body.open("rb") as file:
                    yield from iter(functools.partial(file.read, 1 << 20), b"")
            yield bytes(-size % _BLOCK)
        yield bytes(2 * _BLOCK)

    def save(self, store: Store) -> dict[str, Any]:
        """Require the samples rebuilt from their lines and media files to hash
        to the samples digest (mgmt's ``rebuilt_digest``), store the tar unless
        ``store`` holds it, and return the package's index entry."""
        if not self._lines:
            raise ValueError(f"{self}: no samples")
        rebuilt = hashlib.sha256()
        for line in self._lines:
            rebuilt.update(canonical_sample(self._rebuilt(line)))
            rebuilt.update(b"\n")
        if rebuilt.hexdigest() != self.identity:
            raise ValueError(f"{self}: rebuilt from their lines and media, the samples hash to {rebuilt.hexdigest()}")
        sha, size = hashlib.sha256(), 0
        for chunk in self.tar():
            sha.update(chunk)
            size += len(chunk)
        digest = sha.hexdigest()
        written = store.create(f"tars/{digest}.tar", self.tar())
        logger.info(
            "eval=%s dataset=%s profile=%s identity=%s digest=%s bytes=%d %s",
            self.eval_id,
            self.dataset_version,
            self.prompt_profile,
            self.identity,
            digest,
            size,
            "uploaded" if written else "already present",
        )
        return {
            "prompt_profile": self.prompt_profile,
            "samples_digest": f"sha256:{self.identity}",
            "sample_count": len(self._lines),
            "package": {"format": self.format, "digest": digest, "bytes": size, "media": len(self._media)},
        }


def build(eval_id: str, dataset_version: str, root_dir: pathlib.Path | str, store: Store, scores_version: str) -> None:
    """Pack every profile of one dataset under ``root_dir`` (a
    ``PIPETTE_SCORES_DATA_DIR``), store each tar ``store`` lacks, then the
    dataset's index for ``scores_version``. A VLM set is loaded one profile at
    a time; an ASR set is read in one pass over its audio for every profile."""
    with tempfile.TemporaryDirectory(prefix="packages-") as tmp:
        scratch = pathlib.Path(tmp)
        if eval_id in ASR_EVAL_IDS:
            packages = {name: Package(eval_id, dataset_version, name, scratch) for name in profiles(eval_id)}
            dataset_dir = dataset_catalog._dataset_dir(eval_id, dataset_version, root_dir)
            for requests in _asr_common.iter_profile_requests(EvalId(eval_id), dataset_dir):
                for name, request in requests.items():
                    packages[name].add(request)
            entries = [package.save(store) for package in packages.values()]
        else:
            entries = []
            for profile in profiles(eval_id):
                package = Package(eval_id, dataset_version, profile, scratch)
                for sample in dataset_catalog.load_prompt_samples.__wrapped__(
                    eval_id, dataset_version, root_dir, prompt_profile=profile
                ):
                    package.add(sample)
                entries.append(package.save(store))
    index = {"eval_id": eval_id, "dataset_version": dataset_version, "scores_version": scores_version}
    key = index_key(scores_version, eval_id, dataset_version)
    written = store.create(key, (json.dumps({**index, "profiles": entries}, indent=2) + "\n").encode())
    logger.info("eval=%s dataset=%s %s %s", eval_id, dataset_version, key, "uploaded" if written else "already present")


class S3Store:
    """``s3://<bucket>/<prefix>``; a key is the object key under the prefix."""

    def __init__(self, url: str):
        import boto3

        bucket, _, prefix = url.removeprefix("s3://").partition("/")
        if not bucket or not prefix.strip("/"):
            raise ValueError(f"{url} is not s3://<bucket>/<prefix>")
        self._client = boto3.client("s3")
        self._bucket, self._prefix = bucket, prefix.strip("/")

    def _key(self, key: str) -> str:
        return f"{self._prefix}/{key}"

    def list(self, prefix: str) -> list[str]:
        pages = self._client.get_paginator("list_objects_v2").paginate(Bucket=self._bucket, Prefix=self._key(prefix))
        return [o["Key"].removeprefix(self._key("")) for page in pages for o in page.get("Contents", [])]

    def read(self, key: str) -> bytes | None:
        try:
            return self._client.get_object(Bucket=self._bucket, Key=self._key(key))["Body"].read()
        except self._client.exceptions.NoSuchKey:
            return None

    def download(self, key: str, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self._bucket, self._key(key), str(path))

    def _exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
        except self._client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise
        return True

    def create(self, key: str, body: bytes | Iterable[bytes]) -> bool:
        """Write ``key`` once (``body`` a whole object, or its chunks); False
        when it is there already, a concurrent run's write included (412)."""
        key = self._key(key)
        if self._exists(key):
            return False
        try:
            if isinstance(body, bytes):
                self._client.put_object(Bucket=self._bucket, Key=key, Body=body, IfNoneMatch="*")
            else:
                self._upload(key, body)
        except self._client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "PreconditionFailed":
                return False
            raise
        return True

    def _upload(self, key: str, chunks: Iterable[bytes]) -> None:
        """A multipart upload in parts of ``_PART_BYTES``, so a tar of gigabytes
        is never held whole, completed only if ``key`` is still absent; given
        up on any failure, so no parts are left behind."""
        upload = self._client.create_multipart_upload(Bucket=self._bucket, Key=key, ChecksumAlgorithm="CRC32")
        ids = {"Bucket": self._bucket, "Key": key, "UploadId": upload["UploadId"]}
        try:
            parts = []
            for number, body in enumerate(_parts(chunks), start=1):
                part = self._client.upload_part(**ids, PartNumber=number, Body=body, ChecksumAlgorithm="CRC32")
                parts.append({"PartNumber": number, "ETag": part["ETag"], "ChecksumCRC32": part["ChecksumCRC32"]})
            self._client.complete_multipart_upload(**ids, MultipartUpload={"Parts": parts}, IfNoneMatch="*")
        except BaseException:
            self._client.abort_multipart_upload(**ids)
            raise


def _parts(chunks: Iterable[bytes]) -> Iterator[bytes]:
    """``chunks`` regrouped into parts of ``_PART_BYTES``, the last one shorter."""
    buffer = bytearray()
    for chunk in chunks:
        buffer += chunk
        while len(buffer) >= _PART_BYTES:
            yield bytes(buffer[:_PART_BYTES])
            del buffer[:_PART_BYTES]
    if buffer:
        yield bytes(buffer)


class LocalStore:
    """A local directory in the bucket's place; a key is a path under it."""

    def __init__(self, root: pathlib.Path):
        self._root = root

    def read(self, key: str) -> bytes | None:
        try:
            return (self._root / key).read_bytes()
        except FileNotFoundError:
            return None

    def create(self, key: str, body: bytes | Iterable[bytes]) -> bool:
        """Write ``key`` once: to a temporary file, then linked into place,
        which fails rather than replace a file already there."""
        path = self._root / key
        if path.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, partial = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "wb") as out:
                out.writelines([body] if isinstance(body, bytes) else body)
            os.link(partial, path)
        except FileExistsError:
            return False
        finally:
            os.unlink(partial)
        return True


Store = S3Store | LocalStore


def open_store(url: str) -> Store:
    """The store at ``url``: ``s3://<bucket>/<prefix>``, or a local directory."""
    return S3Store(url) if url.startswith("s3://") else LocalStore(pathlib.Path(url))
