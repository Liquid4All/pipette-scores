"""Build the ASR eval datasets from their pinned sources (build-only: needs the `build` extra).

Each dataset is read as the Open ASR Leaderboard loads it
(huggingface/open_asr_leaderboard ``normalizer/data_utils.load_data`` and
``load_chunked_data``), at the catalog's pinned revision and in the split's row
order, so a sample's position (``doc_id``) is its row in the source split:
`datasets.load_dataset(repo, subset, split=..., revision=...)` for the
leaderboard and FLEURS sets, and for chunked Earnings22 the chunk manifest, its
WAVs and the parent set's call transcripts. The audio column is read with
`Audio(decode=False)`. A clip already at 16 kHz mono is kept as the source
ships it. Any other (MonsoonASR's 44.1 kHz, most of URGENT24) is brought to
16 kHz mono here, once, as the leaderboard's ``prepare_data`` does before
inference (_asr_audio's ``resample_16k_mono``), and stored as the WAV serving
would write: serving resamples nothing, so a set's served samples, and the
corpus pins taken from them, are the same on every CPU architecture (soxr's
resampler rounds per architecture). Each such row's ``raw_doc`` records the
source's rate and channels.

Written to ``<out>/<eval_id>/<version>/``: ``test.parquet`` (one row per clip)
and ``metadata.json`` (source pins, the scoring recipe, the fingerprint, the
audio summary and the license). Rows are streamed to the parquet one row group
at a time, so a build holds one row group of audio, not the dataset's. A full
build of the canonical version must match the catalog's ``expected_samples`` and
``expected_source_sha256`` (when filled) before the file lands. A ``--limit``
smoke build may not take the canonical version, and is never served. The output
may not be inside this repository's checkout (the sets are not redistributable).
"""

from __future__ import annotations

import collections
import datetime
import hashlib
import io
import json
import logging
import pathlib
from collections.abc import Iterable, Iterator
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from pipette_scores.dataset_catalog import _asr_audio, asr_datasets
from pipette_scores.dataset_catalog._asr_common import content_sha256, language_fields, source_sha256
from pipette_scores.dataset_catalog.asr_datasets import AsrDataset
from pipette_scores.hashing import short_hash

logger = logging.getLogger(__name__)

# Rows per parquet row group: the samples stream reads one group at a time.
ROW_GROUP_SIZE = 64

# Where a built set may be kept, as each metadata.json records it. It is not
# part of any fingerprint, so rewording it moves no pin.
REDISTRIBUTION = (
    "internal use only: keep a built set only in the organization's private storage, for internal, "
    "non-commercial benchmarking; never commit it to a public repository, or redistribute it in any "
    "form"
)

# This repository's checkout, in the workspace layout
# (<repo>/packages/pipette-scores/pipette_scores/dataset_catalog/_asr_build.py).
_CHECKOUT = pathlib.Path(__file__).resolve().parents[4]

_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string()),
        pa.field("doc_id", pa.int64()),
        pa.field("reference", pa.string()),
        pa.field("language", pa.string()),
        pa.field("session", pa.string()),
        pa.field("chunk_index", pa.int64()),
        pa.field("audio", pa.binary()),
        pa.field("audio_format", pa.string()),
        pa.field("audio_sha256", pa.string()),
        pa.field("sample_rate", pa.int32()),
        pa.field("channels", pa.int32()),
        pa.field("num_frames", pa.int64()),
        pa.field("raw_doc", pa.string()),
    ]
)


def _audio_format(path: Any) -> str:
    """A name for the clip's format: its file extension when alphanumeric, else wav."""
    if isinstance(path, str) and path:
        ext = pathlib.PurePath(path).suffix.lower().lstrip(".")
        if ext.isalnum():
            return ext
    return "wav"


def _hf_rows(dataset: AsrDataset) -> Iterator[dict[str, Any]]:
    from datasets import Audio, load_dataset

    data = load_dataset(dataset.repo, dataset.subset, split=dataset.split, revision=dataset.revision)
    data = data.cast_column("audio", Audio(decode=False))
    items = cast(Iterable[dict[str, Any]], data)
    for item in items:
        audio = item["audio"]
        audio_bytes = audio.get("bytes")
        if audio_bytes is None:
            audio_bytes = pathlib.Path(audio["path"]).read_bytes()
        raw = item.get(dataset.reference_column)
        yield {
            "audio": audio_bytes,
            "audio_format": _audio_format(audio.get("path")),
            "reference": str(raw).strip() if raw is not None else "",
            "session": None,
            "chunk_index": None,
            "raw_doc": {k: v for k, v in item.items() if k != "audio"},
        }


def _jsonl(repo: str, revision: str, filename: str) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line]


def _earnings22_rows() -> Iterator[dict[str, Any]]:
    """As ``load_chunked_data`` attaches them: each chunk carries its call's full
    transcript (the parent set's ``transcript``, read through ``get_text``), the
    call (``parent_id``) and its ``chunk_index`` in the call."""
    from huggingface_hub import snapshot_download

    parents = _jsonl(
        asr_datasets.EARNINGS22_PARENTS_REPO,
        asr_datasets.EARNINGS22_PARENTS_REVISION,
        asr_datasets.EARNINGS22_PARENTS_FILE,
    )
    references = {row["id"]: row["transcript"] for row in parents}
    chunks = _jsonl(
        asr_datasets.EARNINGS22_CHUNKS_REPO,
        asr_datasets.EARNINGS22_CHUNKS_REVISION,
        asr_datasets.EARNINGS22_CHUNKS_FILE,
    )
    audio_root = pathlib.Path(
        snapshot_download(
            asr_datasets.EARNINGS22_CHUNKS_REPO,
            repo_type="dataset",
            revision=asr_datasets.EARNINGS22_CHUNKS_REVISION,
            allow_patterns=["audio/*.wav"],
        )
    )
    counts = collections.Counter(chunk["parent_id"] for chunk in chunks)
    if len(chunks) != 341 or set(counts) != set(references) or len(references) != 6:
        raise SystemExit("Cleaned Earnings22 must contain 341 chunks of six calls.")
    for chunk in chunks:
        yield {
            "audio": (audio_root / "audio" / chunk["file_name"]).read_bytes(),
            "audio_format": "wav",
            "reference": references[chunk["parent_id"]],
            "session": chunk["parent_id"],
            "chunk_index": int(chunk["chunk_index"]),
            "raw_doc": {**chunk, "parent_chunk_count": counts[chunk["parent_id"]]},
        }


def source_rows(dataset: AsrDataset) -> Iterator[dict[str, Any]]:
    if dataset.eval_id == asr_datasets.EvalId.ASR_ESB_EARNINGS22_CLEANED_AA_CHUNKED:
        return _earnings22_rows()
    return _hf_rows(dataset)


def _resolved_data_files(dataset: AsrDataset) -> dict[str, Any]:
    """The source files `datasets` resolves the load to, and the `datasets`
    version that resolved them."""
    import datasets

    if dataset.eval_id == asr_datasets.EvalId.ASR_ESB_EARNINGS22_CLEANED_AA_CHUNKED:
        return {}
    builder = datasets.load_dataset_builder(dataset.repo, dataset.subset, revision=dataset.revision)
    files = builder.config.data_files or {}
    return {
        "data_files": [str(f) for f in files.get(dataset.split, [])],
        "datasets_version": datasets.__version__,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, bytes):
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


def _prepared(dataset: AsrDataset, rows: Iterable[dict[str, Any]], limit: int | None) -> Iterator[dict[str, Any]]:
    """Rows ready to write, with each clip's header read (soundfile must read it)."""
    import soundfile as sf

    lang = language_fields(dataset.language)
    for position, row in enumerate(rows):
        if limit is not None and position >= limit:
            break
        audio, audio_format = row["audio"], row["audio_format"]
        info = sf.info(io.BytesIO(audio))
        raw_doc = {**_json_safe(row["raw_doc"]), **lang, "audio_subtype": info.subtype}
        if info.samplerate != _asr_audio.TARGET_RATE or info.channels != 1:
            # The source row's columns, for _asr_common.source_sha256 (the catalog's pin).
            raw_doc |= {
                "source_audio_sha256": hashlib.sha256(audio).hexdigest(),
                "source_audio_format": audio_format,
                "source_sample_rate": int(info.samplerate),
                "source_channels": int(info.channels),
                "source_num_frames": int(info.frames),
            }
            audio, audio_format = _asr_audio.wav(_asr_audio.resample_16k_mono(audio)), "wav"
            info = sf.info(io.BytesIO(audio))
        yield {
            "sample_id": short_hash(f"{dataset.eval_id.value}|{position}"),
            "doc_id": position,
            "reference": row["reference"],
            "language": dataset.language,
            "session": row["session"],
            "chunk_index": row["chunk_index"],
            "audio": audio,
            "audio_format": audio_format,
            "audio_sha256": hashlib.sha256(audio).hexdigest(),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "num_frames": int(info.frames),
            "raw_doc": json.dumps(raw_doc, ensure_ascii=False, sort_keys=True),
            "_container": str(info.format),
            # The stored clip's; raw_doc's audio_subtype is the source's (and hashed, so it stays).
            "_subtype": str(info.subtype),
        }


def _stream(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Write ``rows`` to ``path`` one row group at a time; returns them without
    their audio (the audio's size kept as ``_audio_bytes``)."""
    kept: list[dict[str, Any]] = []
    batch: list[dict[str, Any]] = []
    with pq.ParquetWriter(path, _SCHEMA, compression="zstd") as writer:

        def flush() -> None:
            writer.write_table(pa.Table.from_pylist(batch, schema=_SCHEMA), row_group_size=ROW_GROUP_SIZE)
            batch.clear()

        for row in rows:
            batch.append({k: v for k, v in row.items() if not k.startswith("_")})
            kept.append({**{k: v for k, v in row.items() if k != "audio"}, "_audio_bytes": len(row["audio"])})
            if len(batch) == ROW_GROUP_SIZE:
                flush()
        if batch:
            flush()
    return kept


def audio_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    subtypes = collections.Counter(r["_subtype"] for r in rows)
    durations = [r["num_frames"] / r["sample_rate"] for r in rows]
    return {
        # As stored: 16 kHz, a clip the source ships otherwise resampled at build.
        "sample_rates": dict(sorted(collections.Counter(r["sample_rate"] for r in rows).items())),
        "source_sample_rates": dict(
            sorted(
                collections.Counter(
                    json.loads(r["raw_doc"]).get("source_sample_rate", r["sample_rate"]) for r in rows
                ).items()
            )
        ),
        "channels": dict(sorted(collections.Counter(r["channels"] for r in rows).items())),
        # The container soundfile reads, e.g. WAV or FLAC (the rows' audio_format
        # is the clip's file extension).
        "formats": dict(sorted(collections.Counter(r["_container"] for r in rows).items())),
        "subtypes": dict(sorted(subtypes.items())),
        "total_seconds": round(sum(durations), 3),
        "max_seconds": round(max(durations, default=0.0), 3),
        "clips_over_30_s": sum(d > 30.0 for d in durations),
        "audio_bytes": sum(r["_audio_bytes"] for r in rows),
    }


def write_metadata(
    out_dir: pathlib.Path,
    dataset: AsrDataset,
    version: str,
    rows: list[dict[str, Any]],
    limit: int | None,
    resolved: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "repo": dataset.repo,
        "revision": dataset.revision,
        "subset": dataset.subset,
        "split": dataset.split,
        "reference_column": dataset.reference_column,
        **(resolved or {}),
    }
    if dataset.eval_id == asr_datasets.EvalId.ASR_ESB_EARNINGS22_CLEANED_AA_CHUNKED:
        source |= {
            "manifest": asr_datasets.EARNINGS22_CHUNKS_FILE,
            "reference_repo": asr_datasets.EARNINGS22_PARENTS_REPO,
            "reference_revision": asr_datasets.EARNINGS22_PARENTS_REVISION,
            "reference_manifest": asr_datasets.EARNINGS22_PARENTS_FILE,
        }
    metadata = {
        "eval_id": dataset.eval_id.value,
        "dataset": version,
        "family": dataset.family,
        "source": source,
        "num_samples": len(rows),
        # Audio-inclusive fingerprint (the audio through each row's audio_sha256),
        # recomputed at load from the columns; see _asr_common.
        "content_sha256": content_sha256(rows),
        # The catalog's pin: the rows as the source shipped them (_asr_common.source_sha256).
        "source_sha256": source_sha256(rows),
        "scoring": dataset.scoring,
        "audio": audio_summary(rows),
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        # Not None for a smoke build, which is never served.
        "limit": limit,
        "license": dataset.license,
        "license_notes": dataset.license_notes,
        "redistribution": REDISTRIBUTION,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def refuse_inside_checkout(out_root: pathlib.Path, checkout: pathlib.Path = _CHECKOUT) -> None:
    """Refuse an output under this repository's checkout: nothing ignores every
    path there, and the sets must never be committed or baked into an image."""
    if not (checkout / ".git").exists():  # an installed wheel: no checkout to protect
        return
    resolved = out_root.resolve()
    if resolved == checkout or checkout in resolved.parents:
        raise SystemExit(
            f"refusing to build ASR datasets into {resolved}: it is inside the repository checkout "
            f"{checkout}. Build outside it (e.g. --out /data/pipette-scores/datasets) and point "
            "PIPETTE_SCORES_DATA_DIR at its parent."
        )


def build(
    dataset: AsrDataset,
    out_root: pathlib.Path,
    version: str = asr_datasets.DATASET_VERSION,
    *,
    limit: int | None = None,
    rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one dataset into ``<out_root>/<eval_id>/<version>/``; returns its
    metadata. ``rows`` replaces the pinned source (tests)."""
    if limit is not None and version == asr_datasets.DATASET_VERSION:
        raise SystemExit(f"a --limit build may not take the canonical version {version}; pass another --version")
    refuse_inside_checkout(out_root)
    target = out_root / dataset.eval_id.value / version
    target.mkdir(parents=True, exist_ok=True)
    partial = target / "test.parquet.partial"
    from_source = rows is None
    try:
        kept = _stream(partial, _prepared(dataset, source_rows(dataset) if rows is None else rows, limit))
        # The catalog's count and fingerprint pin the canonical version, as the
        # loader checks them (another version name is a test or a new build).
        if limit is None and version == asr_datasets.DATASET_VERSION:
            if dataset.expected_samples is not None and len(kept) != dataset.expected_samples:
                raise SystemExit(
                    f"expected {dataset.expected_samples} {dataset.eval_id} clips, got {len(kept)}: "
                    "wrong or changed source?"
                )
            digest = source_sha256(kept)
            if dataset.expected_source_sha256 and digest != dataset.expected_source_sha256:
                raise SystemExit(
                    f"{dataset.eval_id}: source_sha256 {digest} != expected {dataset.expected_source_sha256}: "
                    "the source or the row formatting changed; review before updating the pin."
                )
        resolved = _resolved_data_files(dataset) if from_source else None
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    path = target / "test.parquet"
    partial.replace(path)
    metadata = write_metadata(target, dataset, version, kept, limit, resolved)
    logger.info("Wrote %d clips to %s (%.1f MB)", len(kept), path, path.stat().st_size / 1e6)
    return metadata
