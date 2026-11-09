"""Serve and score the ASR eval datasets (`test.parquet` + `metadata.json`).

A set too large for one git-LFS object (SPGISpeech, GigaSpeech) is stored as
ordered parts instead of `test.parquet`: `test-00000-of-0000N.parquet`, … ,
holding the rows in order (`parquet_files`; `split_parts` writes them). The parts
are read as one table, and the fingerprint below does not depend on where the
rows are cut, so a split set keeps the build's `metadata.json` unchanged.

The scorer reads every column but the audio; the samples stream and their
digest read the audio one row group at a time (the builder writes small row
groups) and hold one sample's request at a time, so no path holds a whole set's
audio in memory. The audio is stored at 16 kHz mono (the builder resamples a
clip the source ships otherwise, and the loader refuses any other row), and
each served clip is decoded the same way for every profile (_asr_audio).

Integrity, checked before a dataset is scored or served (`verified_rows`):

- ``content_sha256`` (required) is the audio-inclusive fingerprint, as VL's
  field of the same name: every column, the audio through its ``audio_sha256``.
  Since the audio digest is a column, it is recomputed at load without reading
  any audio, and each served clip is re-hashed against that column, so the
  stream carries exactly the fingerprinted bytes.
- ``source_sha256`` (`source_sha256`) is the same fingerprint over the rows as
  the source shipped them: a clip the builder resampled counts with its source
  bytes' digest, rate, channels, length and format (kept in its ``raw_doc``).
  It does not depend on the resampling, which rounds per CPU architecture, so
  any machine reproduces it. For the canonical dataset version it must equal
  the catalog's ``expected_source_sha256`` (when filled).
- Order: row ``i`` has ``doc_id == i`` and ``sample_id == short_hash("<eval_id>|i")``,
  so a sample's id is its position in the source split.
- The recipe: metadata.json's ``scoring`` and every row's ``language`` equal the
  catalog's (`asr_datasets.py`), the clip count its ``expected_samples``.
- A ``--limit`` smoke build (metadata ``limit`` set) is never served or scored.
"""

from __future__ import annotations

import functools
import hashlib
import json
import pathlib
import re
from collections.abc import Iterator, Sequence
from typing import Any

import pyarrow.parquet as pq

from pipette_scores.dataset_catalog import _asr_audio
from pipette_scores.hashing import canonical_sample, short_hash
from pipette_scores.types import AsrSample, ChatMessage, EvalId, EvalSample

# Every column but the audio bytes, whose sha256 is a column of its own.
_TEXT_COLUMNS = (
    "sample_id",
    "doc_id",
    "reference",
    "language",
    "session",
    "chunk_index",
    "audio_format",
    "audio_sha256",
    "sample_rate",
    "channels",
    "num_frames",
    "raw_doc",
)


class AsrDatasetError(ValueError):
    """A materialized ASR dataset that must not be scored or served."""


SINGLE_FILE = "test.parquet"
_PART = re.compile(r"test-(\d{5})-of-(\d{5})\.parquet")


def part_name(index: int, count: int) -> str:
    return f"test-{index:05d}-of-{count:05d}.parquet"


def parquet_files(dataset_dir: pathlib.Path) -> list[pathlib.Path]:
    """The dataset's parquet files in row order: ``test.parquet``, or every part
    ``test-<i>-of-<n>.parquet`` for i = 0 … n-1. Refuses both layouts at once, a
    part missing, a part of another count, or a stray ``test-*.parquet``: a dataset
    must be read whole or not at all (its count and fingerprint checks would catch
    a missing part too; this names the problem)."""
    single = dataset_dir / SINGLE_FILE
    named = sorted(p.name for p in dataset_dir.glob("test-*.parquet"))
    if not named:
        if not single.is_file():
            raise FileNotFoundError(f"{dataset_dir} has no {SINGLE_FILE} or test-<i>-of-<n>.parquet parts")
        return [single]

    def refuse(problem: str) -> AsrDatasetError:
        return AsrDatasetError(f"ASR integrity check failed for {dataset_dir}: {problem}")

    if single.exists():
        raise refuse(f"both {SINGLE_FILE} and parts ({', '.join(named)}): keep one layout")
    matches = [_PART.fullmatch(n) for n in named]
    if not all(matches):
        raise refuse(f"{', '.join(n for n, m in zip(named, matches) if not m)} is not a test-<i>-of-<n>.parquet part")
    count = int(matches[0].group(2))  # type: ignore[union-attr]
    expected = [part_name(i, count) for i in range(count)]
    if count < 1 or named != expected:
        raise refuse(
            f"parts {', '.join(named)} are not test-<i>-of-{count:05d}.parquet for every i from 0 to {count - 1}"
        )
    return [dataset_dir / n for n in expected]


def _pack(sizes: Sequence[int], budget: int) -> list[list[int]]:
    """Row-group indices in order, a new run started where the next would pass ``budget``."""
    runs: list[list[int]] = [[]]
    used = 0
    for index, size in enumerate(sizes):
        if runs[-1] and used + size > budget:
            runs.append([])
            used = 0
        runs[-1].append(index)
        used += size
    return runs


def split_parts(source: pathlib.Path, out_dir: pathlib.Path, max_part_bytes: int) -> list[pathlib.Path]:
    """Write ``source`` (a dataset's ``test.parquet``) into ``out_dir`` as the
    fewest parts of at most ``max_part_bytes`` each, cut between row groups, the
    rows, row groups, schema and compression unchanged and in order; a file within
    the bound is rewritten as one ``test.parquet``. Returns the files written. Copy
    ``metadata.json`` beside them as it is: ``content_sha256`` is a function of
    the rows, not of the files, so it holds for the parts."""
    if out_dir.resolve() == source.parent.resolve():
        raise ValueError(f"write the parts of {source} into another directory, not beside it")
    file = pq.ParquetFile(source)
    meta = file.metadata
    sizes = [
        sum(group.column(c).total_compressed_size for c in range(group.num_columns))
        for group in (meta.row_group(i) for i in range(meta.num_row_groups))
    ]
    if not sizes:
        raise ValueError(f"{source} has no row groups")
    # Each part also carries a footer, at most the whole file's.
    budget = max_part_bytes - (source.stat().st_size - sum(sizes))
    if max(sizes) > budget:
        raise ValueError(f"{source} cannot be cut between row groups into parts of {max_part_bytes} bytes")
    # The fewest parts (filling each in order until the next row group would pass
    # the budget), then the most even cut into that many: the least budget that
    # still needs no more parts.
    count = len(_pack(sizes, budget))
    low, high = max(sizes), budget
    while low < high:
        middle = (low + high) // 2
        low, high = (low, middle) if len(_pack(sizes, middle)) <= count else (middle + 1, high)
    groups = _pack(sizes, low)
    names = [SINGLE_FILE] if len(groups) == 1 else [part_name(i, len(groups)) for i in range(len(groups))]
    compression = meta.row_group(0).column(0).compression.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    partials: list[pathlib.Path] = []
    try:
        for name, group in zip(names, groups):
            partial = out_dir / f"{name}.partial"
            partials.append(partial)
            with pq.ParquetWriter(partial, file.schema_arrow, compression=compression) as writer:
                for index in group:
                    table = file.read_row_group(index)
                    writer.write_table(table, row_group_size=table.num_rows)
            if partial.stat().st_size > max_part_bytes:
                raise ValueError(f"{name} came out at {partial.stat().st_size} bytes, over {max_part_bytes}")
    except BaseException:
        for partial in partials:
            partial.unlink(missing_ok=True)
        raise
    final = [out_dir / name for name in names]
    for partial, path in zip(partials, final):
        partial.replace(path)
    return final


def language_fields(alpha_3: str) -> dict[str, Any]:
    """The language facts the prompt profiles need, fixed at build time (pycountry
    is build-only): the language's English name (pycountry's ISO 639-3 reference
    name), and the ISO 639-1 code pycountry gives (Voxtral's language hint), None
    when there is none."""
    import pycountry

    code = alpha_3.strip().lower()
    language = pycountry.languages.get(alpha_3=code)
    if language is None:
        raise ValueError(f"Unknown ISO 639-3 language code: {alpha_3!r}")
    alpha_2 = getattr(language, "alpha_2", None)
    return {"language_name": str(language.name), "language_alpha_2": alpha_2 if isinstance(alpha_2, str) else None}


def content_sha256(rows: Sequence[dict[str, Any]]) -> str:
    """The dataset's audio-inclusive fingerprint: every column, the audio through
    its ``audio_sha256``. Order-independent like VL's (the order is pinned by
    ``doc_id``, a hashed column checked against the position); the builder
    computes ``audio_sha256`` from the bytes, the loader recomputes this from
    the columns without reading audio."""
    canonical = "\n".join(
        sorted("\x1f".join("" if r.get(c) is None else str(r[c]) for c in _TEXT_COLUMNS) for r in rows)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# A resampled clip's source, as the builder records it in raw_doc, by the column it stands for.
SOURCE_KEYS = {
    "source_audio_sha256": "audio_sha256",
    "source_audio_format": "audio_format",
    "source_sample_rate": "sample_rate",
    "source_channels": "channels",
    "source_num_frames": "num_frames",
}


def source_row(row: dict[str, Any]) -> dict[str, Any]:
    """The row as it would be stored unresampled: a resampled clip's source
    fields back in their columns and out of ``raw_doc``."""
    doc = json.loads(row["raw_doc"])
    if "source_audio_sha256" not in doc:
        return row
    source = {column: doc.pop(key) for key, column in SOURCE_KEYS.items()}
    return {**row, **source, "raw_doc": json.dumps(doc, ensure_ascii=False, sort_keys=True)}


def source_sha256(rows: Sequence[dict[str, Any]]) -> str:
    """`content_sha256` of the rows as the source shipped them (`source_row`):
    the catalog's pin, the same on every CPU architecture."""
    return content_sha256([source_row(r) for r in rows])


def read_metadata(dataset_dir: pathlib.Path) -> dict[str, Any]:
    return json.loads((dataset_dir / "metadata.json").read_text())


def _check_dataset(dataset_dir: pathlib.Path, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    from pipette_scores.dataset_catalog import asr_datasets

    def refuse(problem: str) -> AsrDatasetError:
        return AsrDatasetError(f"ASR integrity check failed for {dataset_dir}: {problem}")

    eval_id_value = metadata.get("eval_id")
    if eval_id_value != dataset_dir.parent.name:
        raise refuse(f"metadata.json names eval {eval_id_value!r}, the dataset sits under {dataset_dir.parent.name!r}")
    try:
        dataset = asr_datasets.ASR_DATASETS[EvalId(eval_id_value)]
    except (KeyError, ValueError):
        raise refuse(f"{eval_id_value!r} is not an ASR eval") from None
    if metadata.get("limit") is not None:
        raise refuse(f"a --limit {metadata['limit']} smoke build is never served or scored")
    expected = metadata.get("content_sha256")
    if not expected:
        raise refuse("metadata.json has no content_sha256")
    actual = content_sha256(rows)
    if actual != expected:
        raise refuse(
            f"content_sha256 {actual} != {expected} from metadata.json (each served clip is "
            "re-hashed against its audio_sha256; verify_asr_datasets.py checks them all offline)"
        )
    canonical = dataset_dir.name == asr_datasets.DATASET_VERSION
    if canonical and dataset.expected_source_sha256:
        source = source_sha256(rows)
        if source != dataset.expected_source_sha256:
            raise refuse(
                f"source_sha256 {source} is not the catalog's expected_source_sha256 "
                f"{dataset.expected_source_sha256} for {asr_datasets.DATASET_VERSION}"
            )
    if canonical and dataset.expected_samples is not None and len(rows) != dataset.expected_samples:
        raise refuse(f"{len(rows)} clips, the catalog expects {dataset.expected_samples}")
    if metadata.get("num_samples") != len(rows):
        raise refuse(f"metadata.json says {metadata.get('num_samples')} clips, the parquet holds {len(rows)}")
    if metadata.get("scoring") != dataset.scoring:
        raise refuse(f"scoring recipe {metadata.get('scoring')} is not the catalog's {dataset.scoring}")
    for position, row in enumerate(rows):
        if row["doc_id"] != position or row["sample_id"] != short_hash(f"{eval_id_value}|{position}"):
            raise refuse(
                f"row {position} is doc_id {row['doc_id']} / sample {row['sample_id']}: a sample's id must be "
                "its position in the source split"
            )
        if row["language"] != dataset.language:
            raise refuse(
                f"row {position} is in {row['language']!r}, the catalog's {dataset.eval_id} is {dataset.language!r}"
            )
        # Serving must not resample: soxr rounds per CPU architecture.
        if (row["sample_rate"], row["channels"]) != (_asr_audio.TARGET_RATE, 1):
            raise refuse(
                f"row {position} is {row['sample_rate']} Hz x {row['channels']}: stored clips are 16 kHz mono "
                "(a build before resampling moved to the builder; rebuild it)"
            )


@functools.lru_cache(maxsize=64)
def _verified(dataset_dir: pathlib.Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    rows = [
        row
        for path in parquet_files(dataset_dir)
        for row in pq.read_table(path, columns=list(_TEXT_COLUMNS)).to_pylist()
    ]
    metadata = read_metadata(dataset_dir)
    _check_dataset(dataset_dir, metadata, rows)
    return metadata, tuple(rows)


def verified_rows(dataset_dir: pathlib.Path) -> tuple[dict[str, Any], ...]:
    """Every row without its audio, once the dataset passes the checks above
    (memoized per directory: datasets are immutable build artifacts)."""
    return _verified(dataset_dir.resolve())[1]


def load_eval_samples(dataset_dir: pathlib.Path) -> list[AsrSample]:
    metadata, rows = _verified(dataset_dir.resolve())
    scoring = metadata["scoring"]
    return [
        AsrSample(
            id=row["sample_id"],
            reference=row["reference"],
            language=row["language"],
            session=row["session"],
            chunk_index=row["chunk_index"],
            scoring=scoring,
            raw_doc=json.loads(row["raw_doc"]),
        )
        for row in rows
    ]


def num_samples(dataset_dir: pathlib.Path) -> int:
    return len(verified_rows(dataset_dir))


def _audio_from(dataset_dir: pathlib.Path, rows: Sequence[dict[str, Any]]) -> Iterator[bytes]:
    """The audio of every row, in order, one row group read at a time (across
    the parts of a split set, in order), each clip re-hashed against its row's
    ``audio_sha256``."""
    position = 0
    for path in parquet_files(dataset_dir):
        file = pq.ParquetFile(path)
        for index in range(file.metadata.num_row_groups):
            clips = file.read_row_group(index, columns=["audio"]).column("audio").to_pylist()
            for audio in clips:
                if hashlib.sha256(audio).hexdigest() != rows[position]["audio_sha256"]:
                    raise AsrDatasetError(
                        f"ASR integrity check failed for {dataset_dir}: row {position}'s audio does not match its "
                        "audio_sha256"
                    )
                yield audio
                position += 1


def encode_for(audio: bytes, profile: Any) -> str:
    """The clip as ``profile`` sends it: 16 kHz mono WAV (16-bit PCM where that
    is exact, else 32-bit float), cut to the profile's sample cap if it has one
    (_asr_audio.encode)."""
    return _asr_audio.encode(audio, max_samples=profile.audio_max_samples)


def _clip(eval_id: EvalId, row: dict[str, Any]) -> Any:
    from pipette_scores.dataset_catalog import asr_prompt_profiles

    doc = json.loads(row["raw_doc"])
    return asr_prompt_profiles.AsrClip(
        eval_id=eval_id,
        language=row["language"],
        language_name=doc["language_name"],
        language_alpha_2=doc.get("language_alpha_2"),
    )


def _request(eval_id: EvalId, row: dict[str, Any], profile_name: str, audio_b64: str | None) -> EvalSample:
    """One row as the profile's request; ``audio_b64=None`` keeps the audio part
    as a placeholder, its format and no data (the prompt a scored sample is
    shown with: the turn order stays visible, the clip is the caller's)."""
    from pipette_scores.dataset_catalog import asr_prompt_profiles

    profile = asr_prompt_profiles.profile(profile_name)
    clip = _clip(eval_id, row)
    audio = {"type": "input_audio", "input_audio": {"data": audio_b64 or "", "format": "wav"}}
    messages = profile.messages(clip, audio)
    return EvalSample(
        id=row["sample_id"],
        messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
        chat_template_kwargs=profile.chat_template_kwargs(clip),
    )


def iter_prompt_samples(eval_id: EvalId, dataset_dir: pathlib.Path, profile_name: str) -> Iterator[EvalSample]:
    """The profile's requests, with their audio, for the whole set, in order,
    one at a time: memory holds one row group's source audio and one
    request, never the set. The dataset is verified before the first request
    (`verified_rows`), and each clip is re-hashed as it is read."""
    from pipette_scores.dataset_catalog import asr_prompt_profiles

    profile = asr_prompt_profiles.profile(profile_name)
    dataset_dir = dataset_dir.resolve()
    rows = verified_rows(dataset_dir)

    def requests() -> Iterator[EvalSample]:
        for position, audio in enumerate(_audio_from(dataset_dir, rows)):
            yield _request(eval_id, rows[position], profile_name, encode_for(audio, profile))

    return requests()


def iter_profile_requests(eval_id: EvalId, dataset_dir: pathlib.Path) -> Iterator[dict[str, EvalSample]]:
    """Each row's request for every profile, by profile name, in served order and
    in one pass over the audio: each clip is decoded once, and encoded once per
    distinct cut (a profile's cap only matters for a clip longer than it). Each
    request is the one `iter_prompt_samples` serves for that profile."""
    from pipette_scores.dataset_catalog import asr_prompt_profiles

    dataset_dir = dataset_dir.resolve()
    rows = verified_rows(dataset_dir)
    for position, audio in enumerate(_audio_from(dataset_dir, rows)):
        mono = _asr_audio.to_16k_mono(audio)
        encoded: dict[int | None, str] = {}
        requests = {}
        for name, profile in asr_prompt_profiles.PROFILES.items():
            cap = profile.audio_max_samples
            cut = cap if cap is not None and len(mono) > cap else None
            if cut not in encoded:
                encoded[cut] = _asr_audio.encode_samples(mono if cut is None else mono[:cut])
            requests[name] = _request(eval_id, rows[position], name, encoded[cut])
        yield requests


def prompt_digests(eval_id: EvalId, dataset_dir: pathlib.Path) -> dict[str, tuple[str, int]]:
    """Every profile's ``samples_digest`` of `iter_prompt_samples`, and the
    sample count, in one pass over the audio (`iter_profile_requests`), each
    request hashed into its profile's digest. A server asked for one profile's
    digest learns them all for the price of one pass."""
    from pipette_scores.dataset_catalog import asr_prompt_profiles

    hashers = {name: hashlib.sha256() for name in asr_prompt_profiles.PROFILES}
    for requests in iter_profile_requests(eval_id, dataset_dir):
        for name, request in requests.items():
            hashers[name].update(canonical_sample(request))
            hashers[name].update(b"\n")
    count = len(verified_rows(dataset_dir))
    return {name: (f"sha256:{h.hexdigest()}", count) for name, h in hashers.items()}


def load_prompt_text_samples(eval_id: EvalId, dataset_dir: pathlib.Path, profile_name: str) -> list[EvalSample]:
    """Every request with its audio parts emptied (format kept, no data): what a
    scored sample shows as its prompt."""
    return [_request(eval_id, row, profile_name, None) for row in verified_rows(dataset_dir)]
