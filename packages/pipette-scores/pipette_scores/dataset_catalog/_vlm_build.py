"""Shared build machinery for the 7 VLM benchmark dataset builder scripts.

The one-shot builders in ``packages/pipette-scores/scripts/build_<eval>_dataset.py``
(mme, mmstar, ocrbench, mmmu_val, realworldqa, mmbench_en_dev, pope) declare a
:class:`BenchmarkSpec` (pinned VLMEvalKit TSV URL + MD5, prompt style, license
facts) and delegate here. Like the other third-party datasets (ifbench,
gpqa_diamond, math_500), the output is **materialized locally and never
committed** — a license review concluded none of the 7 benchmarks may be
redistributed in transformed form (see docs/dataset-licenses.md).

Small pieces of VLMEvalKit logic are vendored (with source citations) instead
of importing the ``vlmeval`` package:

- TSV parsing:            ``vlmeval/smp/file.py`` ``load_tsv`` (pd.read_csv, sep='\\t')
- image pointer dedup:    ``vlmeval/dataset/image_base.py`` ``ImageBaseDataset.__init__``
- list-in-a-string parse: ``vlmeval/smp/misc.py`` ``toliststr`` (ast.literal_eval
                          instead of eval)
- MCQ prompt:             ``vlmeval/dataset/image_mcq.py`` ``ImageMCQDataset.build_prompt``

Output contract (read by ``pipette_scores.dataset_catalog._vlm_common``):
``datasets/<eval_id>/<version>/test.parquet`` with columns ``sample_id``,
``doc_id``, ``prompt``, ``answer``, ``num_images``, ``image_0..image_N``
(raw encoded image bytes), ``raw_doc`` (JSON) — plus ``metadata.json``.
"""

from __future__ import annotations

import argparse
import ast
import base64
import dataclasses
import datetime
import hashlib
import io
import json
import logging
import os
import pathlib
import string
import tempfile
from typing import Any, Callable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipette_scores.dataset_catalog._vlm_common import metadata_content_sha256

from pipette_scores.hashing import short_hash

logger = logging.getLogger(__name__)

# VLMEvalKit commit the TSV URL/MD5 pins and vendored logic were taken from.
VLMEVALKIT_COMMIT = "8eca8e5"

# Default semantic dataset version (year.month.patch); override with --version.
DEFAULT_DATASET_VERSION = "2026.10.1"

# Re-encode settings used only when --max-image-px triggers a resize
# (matches the legacy edge-evals extractor: JPEG quality 85, LANCZOS).
JPEG_QUALITY = 85

_DEFAULT_CACHE_DIR = pathlib.Path(
    os.environ.get(
        "PIPETTE_VLM_TSV_CACHE",
        pathlib.Path.home() / ".cache" / "pipette-scores" / "vlmevalkit",
    )
)


@dataclasses.dataclass(frozen=True)
class BenchmarkSpec:
    """Everything a per-benchmark builder script declares."""

    eval_id: str
    vlmevalkit_name: str  # VLMEvalKit DATASET_URL key, e.g. 'MMBench_DEV_EN_V11'
    url: str
    md5: str
    # 'mcq'  -> VLMEvalKit ImageMCQDataset.build_prompt text (Hint/Question/Options)
    # 'bare' -> the raw `question` column verbatim (MME / POPE / OCRBench)
    prompt_style: str
    license: str
    license_notes: str
    # Sample count of the full build; asserted unless --limit is used.
    # None = not yet known (filled after the first build).
    expected_samples: int | None = None
    # Fingerprint of the full original-resolution build ('' until filled after
    # the first build); asserted only for full, non-resized builds.
    expected_content_sha256: str = ""
    # Optional row filter applied before --limit (e.g. MMMU: keep VAL split).
    filter_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    # Extra lines appended to the post-build license notice (e.g. the
    # RealWorldQA CC-BY-ND no-redistribution warning).
    extra_notice: str | None = None


# ---------------------------------------------------------------------------
# Download + integrity
# ---------------------------------------------------------------------------


def _file_md5(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_tsv(
    url: str,
    md5: str,
    cache_dir: pathlib.Path | None = None,
    insecure: bool = False,
) -> pathlib.Path:
    """Download (or reuse a cached copy of) a pinned upstream TSV, verifying MD5.

    ``insecure=True`` disables TLS certificate verification only — the
    opencompass.openxlab.space leaf certificate has been observed expired.
    Integrity is still enforced via the pinned MD5, which comes from
    VLMEvalKit's DATASET_MD5 tables (commit 8eca8e5), not from the server.
    """
    import requests

    cache_dir = pathlib.Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / url.rsplit("/", 1)[-1]

    if target.exists():
        if _file_md5(target) == md5:
            logger.info("Using cached %s (md5 ok)", target)
            return target
        logger.warning("Cached %s has wrong md5, re-downloading", target)
        target.unlink()

    logger.info("Downloading %s -> %s", url, target)
    if insecure:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    try:
        with requests.get(url, stream=True, timeout=120, verify=not insecure) as resp:
            resp.raise_for_status()
            fd, tmp_name = tempfile.mkstemp(dir=cache_dir, suffix=".part")
            try:
                with os.fdopen(fd, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                os.replace(tmp_name, target)
            except BaseException:
                pathlib.Path(tmp_name).unlink(missing_ok=True)
                raise
    except requests.exceptions.SSLError as e:
        raise RuntimeError(
            f"TLS verification failed for {url}: {e}\n"
            "The opencompass.openxlab.space certificate has been observed expired. "
            "Re-run with --insecure to skip TLS verification; file integrity is "
            "still enforced against the MD5 pinned from VLMEvalKit."
        ) from e

    got = _file_md5(target)
    if got != md5:
        target.unlink()
        raise RuntimeError(f"MD5 mismatch for {url}: expected {md5}, got {got} — refusing to build from it")
    return target


def load_tsv(path: pathlib.Path) -> pd.DataFrame:
    # Parity with VLMEvalKit `vlmeval/smp/file.py` load_tsv: only an empty cell is
    # missing, so an option whose text is "None" or "NA" is kept.
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------


def resolve_image_pointers(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve the TSV space-saving convention where an ``image`` cell holds
    another row's index instead of base64 (cells <= 64 chars are pointers) —
    e.g. MMBench circular-rotation rows point at their base question's image.

    Vendored from VLMEvalKit ``vlmeval/dataset/image_base.py``
    ``ImageBaseDataset.__init__`` (commit 8eca8e5).
    """
    if "image" not in df.columns:
        return df
    indices = [str(x) for x in df["index"]]
    image_map = {idx: str(img) for idx, img in zip(indices, df["image"])}
    for k in image_map:
        if len(image_map[k]) <= 64:
            idx = image_map[k]
            if idx not in image_map or len(image_map[idx]) <= 64:
                raise ValueError(f"image pointer {k!r} -> {idx!r} does not resolve to a base64 image")
            image_map[k] = image_map[idx]
    df = df.copy()
    df["image"] = [image_map[k] for k in indices]
    return df


def _to_list_str(s: Any) -> list[str]:
    # Vendored from VLMEvalKit `vlmeval/smp/misc.py` toliststr, with
    # ast.literal_eval instead of eval (the cell is untrusted input).
    if isinstance(s, str) and s.startswith("[") and s.endswith("]"):
        return [str(x) for x in ast.literal_eval(s)]
    if isinstance(s, str):
        return [s]
    if isinstance(s, list):
        return [str(x) for x in s]
    raise TypeError(f"cannot interpret image cell of type {type(s)}")


def images_from_cell(cell: Any) -> list[bytes]:
    """Decode the ``image`` cell (one base64 string, or a stringified list of
    them for multi-image rows) into raw image bytes, byte-exact."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    return [base64.b64decode(b64) for b64 in _to_list_str(cell)]


def sniff_image_format(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    if data.startswith(b"BM"):
        return "bmp"
    return "unknown"


def maybe_resize(data: bytes, max_image_px: int | None) -> bytes:
    """Default: return the original bytes untouched (accuracy parity with
    VLMEvalKit). With ``max_image_px`` set, images whose longest side exceeds
    the cap are downscaled (LANCZOS) and re-encoded as JPEG q=85; smaller
    images keep their original bytes."""
    if max_image_px is None:
        return data
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        w, h = img.size
        if max(w, h) <= max_image_px:
            return data
        scale = max_image_px / max(w, h)
        resized = img.convert("RGB").resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_mcq_prompt(line: dict) -> str:
    """Text part of VLMEvalKit's ``ImageMCQDataset.build_prompt``
    (``vlmeval/dataset/image_mcq.py`` @ 8eca8e5), kept byte-identical:

        [Hint: {hint}\\n]                                  # only if present
        Question: {question}\\n
        Options:\\nA. {A}\\nB. {B}\\n...                   # letters present in the row
        Please select the correct answer from the options above. \\n
    """
    question = line["question"]
    options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
    options_prompt = "Options:\n"
    for key, item in options.items():
        options_prompt += f"{key}. {item}\n"
    hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
    prompt = ""
    if hint is not None:
        prompt += f"Hint: {hint}\n"
    prompt += f"Question: {question}\n"
    if len(options):
        prompt += options_prompt
        prompt += "Please select the correct answer from the options above. \n"
    return prompt


def build_bare_prompt(line: dict) -> str:
    """MME / POPE / OCRBench: the raw ``question`` column, verbatim."""
    return str(line["question"])


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


def _json_safe(v: Any) -> Any:
    """Make a pandas cell JSON-serializable (numpy scalars -> python, NaN -> None)."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if hasattr(v, "item"):  # numpy scalar
        return v.item()
    if isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def row_raw_doc(line: dict) -> dict:
    """All non-image TSV columns, JSON-safe. ``image_path`` is kept (MME's
    acc_plus pairing key); the base64 ``image`` column is dropped — the bytes
    are stored in the parquet image columns."""
    return {k: _json_safe(v) for k, v in line.items() if k != "image"}


def build_rows(
    df: pd.DataFrame,
    spec: BenchmarkSpec,
    max_image_px: int | None,
) -> tuple[list[dict], dict[str, int]]:
    """Build parquet-ready rows. Returns (rows, image format counts).

    Each row dict holds ``_images`` (list of bytes) which ``write_dataset``
    spreads into ``image_0..image_N`` columns.
    """
    prompt_fn = {"mcq": build_mcq_prompt, "bare": build_bare_prompt}[spec.prompt_style]
    rows: list[dict] = []
    format_counts: dict[str, int] = {}
    for pos, (_, series) in enumerate(df.iterrows()):
        line = series.to_dict()
        images = [maybe_resize(b, max_image_px) for b in images_from_cell(line.get("image"))]
        for b in images:
            fmt = sniff_image_format(b)
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
        index = line.get("index", pos)
        answer = line.get("answer")
        rows.append(
            {
                "sample_id": short_hash(f"{spec.eval_id}|{index}"),
                "doc_id": int(index) if str(index).lstrip("-").isdigit() else pos,
                "prompt": prompt_fn(line),
                "answer": json.dumps(_json_safe(answer)) if not pd.isna(answer) else None,
                "num_images": len(images),
                "_images": images,
                "raw_doc": json.dumps(row_raw_doc(line), ensure_ascii=False),
            }
        )
    return rows, format_counts


def content_sha256(rows: list[dict]) -> str:
    """Fingerprint of the dataset's logical content (same idea as the
    math_500 builder): hashes the canonical rows — prompt, answer, raw_doc and
    the image bytes' digests — not the parquet bytes, which aren't stable
    across writer versions."""
    canonical = "\n".join(
        sorted(
            "\x1f".join(
                [
                    r["sample_id"],
                    r["prompt"],
                    r["answer"] or "",
                    r["raw_doc"],
                    *[hashlib.sha256(img).hexdigest() for img in r["_images"]],
                ]
            )
            for r in rows
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_dataset(out_dir: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    """Write ``test.parquet`` with an explicit schema: image columns are
    binary even when sparsely populated (multi-image benchmarks)."""
    max_images = max((r["num_images"] for r in rows), default=0)
    n_img_cols = max(max_images, 1)

    fields = [
        pa.field("sample_id", pa.string()),
        pa.field("doc_id", pa.int64()),
        pa.field("prompt", pa.string()),
        pa.field("answer", pa.string()),
        pa.field("num_images", pa.int32()),
        *[pa.field(f"image_{i}", pa.binary()) for i in range(n_img_cols)],
        pa.field("raw_doc", pa.string()),
    ]
    columns: dict[str, list] = {f.name: [] for f in fields}
    for r in rows:
        images = r["_images"]
        for key in ("sample_id", "doc_id", "prompt", "answer", "num_images", "raw_doc"):
            columns[key].append(r[key])
        for i in range(n_img_cols):
            columns[f"image_{i}"].append(images[i] if i < len(images) else None)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "test.parquet"
    table = pa.Table.from_pydict(columns, schema=pa.schema(fields))
    pq.write_table(table, path, compression="zstd")
    return path


def write_metadata(
    out_dir: pathlib.Path,
    spec: BenchmarkSpec,
    version: str,
    num_samples: int,
    digest: str,
    metadata_digest: str,
    format_counts: dict[str, int],
    max_image_px: int | None,
    limit: int | None,
) -> None:
    metadata = {
        "eval_id": spec.eval_id,
        "dataset": version,
        "source": {
            "provider": f"VLMEvalKit pinned TSV (commit {VLMEVALKIT_COMMIT})",
            "vlmevalkit_dataset": spec.vlmevalkit_name,
            "url": spec.url,
            "md5": spec.md5,
        },
        "num_samples": num_samples,
        # Image-inclusive digest (offline verify_vlm_datasets.py) + cheap
        # image-free digest the serving loader re-verifies at load.
        "content_sha256": digest,
        "metadata_content_sha256": metadata_digest,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        # None = original resolution/bytes (accuracy parity with VLMEvalKit).
        "max_image_px": max_image_px,
        "jpeg_quality": JPEG_QUALITY if max_image_px is not None else None,
        "image_formats": dict(sorted(format_counts.items())),
        "limit": limit,
        "license": spec.license,
        "license_notes": spec.license_notes,
        "redistribution": (
            "internal use only: keep it solely in the organization's private storage; "
            "never commit it to a public repo or redistribute it in any form "
            "(see docs/dataset-licenses.md)"
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def print_license_notice(spec: BenchmarkSpec, out_dir: pathlib.Path, num_samples: int) -> None:
    bar = "=" * 78
    lines = [
        bar,
        f"Built {spec.eval_id} ({num_samples} samples) -> {out_dir}",
        "",
        f"LICENSE: {spec.license}",
        spec.license_notes,
        "",
        "This dataset was built locally for evaluation use. Do NOT commit it to a",
        "repository, upload it, or redistribute it in any form. If a pipette-mgmt",
        "server will serve these samples (base64 images) to clients outside your",
        "organization, that is distribution too — get legal sign-off first.",
    ]
    if spec.extra_notice:
        lines += ["", spec.extra_notice]
    lines.append(bar)
    print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# Build + CLI entrypoint (shared by the 7 scripts)
# ---------------------------------------------------------------------------


def build(
    spec: BenchmarkSpec,
    out_dir: pathlib.Path,
    version: str = DEFAULT_DATASET_VERSION,
    *,
    limit: int | None = None,
    max_image_px: int | None = None,
    insecure: bool = False,
    cache_dir: pathlib.Path | None = None,
) -> int:
    """Run one benchmark's build into ``<out_dir>/<version>/``. Returns the
    number of samples written."""
    tsv_path = download_tsv(spec.url, spec.md5, cache_dir=cache_dir, insecure=insecure)
    df = load_tsv(tsv_path)
    df = resolve_image_pointers(df)
    if spec.filter_fn is not None:
        df = spec.filter_fn(df)
    if limit is not None:
        df = df.head(limit)
        logger.warning("--limit %d set: building a truncated dataset (smoke-test only)", limit)
    elif spec.expected_samples is not None and len(df) != spec.expected_samples:
        raise SystemExit(f"expected {spec.expected_samples} {spec.eval_id} rows, got {len(df)} — wrong/changed TSV?")

    rows, format_counts = build_rows(df, spec, max_image_px)

    digest = content_sha256(rows)
    if limit is None and max_image_px is None and spec.expected_content_sha256:
        if digest != spec.expected_content_sha256:
            raise SystemExit(
                f"content hash {digest} != expected {spec.expected_content_sha256} — upstream "
                f"data or the row formatting changed; review before updating the pin."
            )

    target_dir = out_dir / version
    path = write_dataset(target_dir, rows)
    write_metadata(
        target_dir,
        spec,
        version,
        len(rows),
        digest,
        metadata_content_sha256(rows),
        format_counts,
        max_image_px,
        limit,
    )
    logger.info("Wrote %d rows to %s (%.1f MB)", len(rows), path, path.stat().st_size / 1e6)
    print_license_notice(spec, target_dir, len(rows))
    return len(rows)


def run_cli(spec: BenchmarkSpec, description: str | None = None) -> None:
    """Shared argparse entrypoint for the per-benchmark builder scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--version",
        default=DEFAULT_DATASET_VERSION,
        help=f"Dataset version dir name (default: {DEFAULT_DATASET_VERSION})",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("datasets") / spec.eval_id,
        help=f"datasets/{spec.eval_id} root (version subdir is created under it)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Build just the first N rows (smoke-testing only; not a valid eval dataset)",
    )
    parser.add_argument(
        "--max-image-px",
        type=int,
        default=None,
        help=(
            "Downscale images whose longest side exceeds N px (JPEG re-encode). "
            "Default keeps original bytes for accuracy parity with VLMEvalKit; "
            "use e.g. 512 to shrink edge payloads."
        ),
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help=(
            "Skip TLS certificate verification for the upstream TSV download "
            "(opencompass.openxlab.space has served an expired cert). File "
            "integrity is still enforced via the pinned MD5."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=None,
        help="Where downloaded TSVs are cached (default: ~/.cache/pipette-scores/vlmevalkit)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    build(
        spec,
        args.out_dir,
        args.version,
        limit=args.limit,
        max_image_px=args.max_image_px,
        insecure=args.insecure,
        cache_dir=args.cache_dir,
    )
