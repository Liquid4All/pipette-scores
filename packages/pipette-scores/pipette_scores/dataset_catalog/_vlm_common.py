import base64
import hashlib
import json
import pathlib
from collections.abc import Callable

from pipette_scores.parquet_io import read_parquet
from pipette_scores.types import ChatMessage, EvalSample, VlmSample


def metadata_content_sha256(rows: list[dict]) -> str:
    """Cheap, **image-free** integrity digest over the non-image columns
    (``sample_id``, ``prompt``, ``answer``, ``raw_doc``).

    Recomputable at load without decoding images, so the serving path can
    re-verify a materialized parquet against the fingerprint the builder
    wrote to ``metadata.json`` — the VLM analogue of the text loaders'
    ``content_sha256`` re-check. It deliberately does **not** hash image
    bytes (that stays in the builder's image-inclusive ``content_sha256``,
    verified offline by ``verify_vlm_datasets.py``); this narrows the
    load-time gap to "no image-byte check" rather than "no check at all".
    Order-independent (rows are sorted by id) so it's stable across writers.
    """
    items = sorted(
        (
            str(r["sample_id"]),
            str(r["prompt"]),
            "" if r.get("answer") is None else str(r["answer"]),
            str(r["raw_doc"]),
        )
        for r in rows
    )
    canonical = "\n".join("\x1f".join(fields) for fields in items)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_verified(dataset_dir: pathlib.Path) -> list[dict]:
    """Read ``test.parquet`` and re-verify its non-image digest against
    ``metadata.json`` when the fingerprint is present. Absent fingerprint
    (datasets built before this field) → no check, matching prior behavior.
    """
    rows = read_parquet(dataset_dir / "test.parquet")
    meta_path = dataset_dir / "metadata.json"
    expected = None
    if meta_path.is_file():
        expected = json.loads(meta_path.read_text()).get("metadata_content_sha256")
    if expected:
        actual = metadata_content_sha256(rows)
        if actual != expected:
            raise ValueError(
                f"VLM integrity check failed for {dataset_dir}: metadata_content_sha256 "
                f"{actual} != {expected} from metadata.json — the materialized dataset "
                "doesn't match the committed fingerprint (image bytes are checked "
                "separately by verify_vlm_datasets.py)."
            )
    return rows


# Image bytes are stored exactly as encoded in the parquet (the builders keep
# the original upstream bytes by default), so the mime type is sniffed from
# the magic bytes rather than assumed.
_MAGIC_MIME = (
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
)


def _image_format(image_bytes: bytes) -> str:
    for magic, fmt in _MAGIC_MIME:
        if image_bytes.startswith(magic):
            return fmt
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "webp"
    return "jpeg"  # historical default


def _image_to_data_uri(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/{_image_format(image_bytes)};base64,{b64}"


def _images_from_row(row: dict) -> list[bytes]:
    n = int(row.get("num_images", 0) or 0)
    out = []
    for i in range(n):
        b = row.get(f"image_{i}")
        if b is not None:
            out.append(b)
    return out


def _interleaved_blocks(prompt: str, images: list[bytes]) -> list[dict] | None:
    """Interleave images at ``<image N>`` placeholders in the prompt.

    Port of VLMEvalKit ``MMMUDataset.split_MMMU`` (vlmeval/dataset/image_mcq.py
    @ 8eca8e5): MMMU questions embed 1-based ``<image N>`` references; the
    referenced image is inserted at the placeholder position and the text kept
    otherwise unchanged. Returns None when the prompt has no (well-formed)
    placeholders so the caller falls back to the images-first layout.
    """
    segs = prompt.split("<image ")
    if len(segs) == 1 or not images:
        return None

    blocks: list[dict] = []
    if segs[0]:
        blocks.append({"type": "text", "text": segs[0]})
    for seg in segs[1:]:
        # Upstream asserts the placeholder shape (single digit + '>'); we
        # fall back to the default layout on anything malformed instead.
        if len(seg) < 2 or not seg[0].isdigit() or seg[1] != ">":
            return None
        image_idx = int(seg[0]) - 1
        if not 0 <= image_idx < len(images):
            return None
        blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_to_data_uri(images[image_idx])},
            }
        )
        if seg[2:]:
            blocks.append({"type": "text", "text": seg[2:]})
    return blocks


def _content_blocks(prompt: str, images: list[bytes], *, interleave: bool = False) -> list[dict]:
    """Images first, then the prompt; ``interleave`` places them at their
    ``<image N>`` placeholders instead, as upstream does for MMMU only."""
    interleaved = _interleaved_blocks(prompt, images) if interleave else None
    if interleaved is not None:
        return interleaved
    blocks: list[dict] = [
        {
            "type": "image_url",
            "image_url": {"url": _image_to_data_uri(img_bytes)},
        }
        for img_bytes in images
    ]
    blocks.append({"type": "text", "text": prompt})
    return blocks


def load_eval_samples(dataset_dir: pathlib.Path) -> list[VlmSample]:
    rows = _read_verified(dataset_dir)
    return [
        VlmSample(
            id=row["sample_id"],
            prompt=row["prompt"],
            answer=json.loads(row["answer"]) if row.get("answer") else None,
            raw_doc=json.loads(row["raw_doc"]),
        )
        for row in rows
    ]


def load_prompt_samples(
    dataset_dir: pathlib.Path,
    *,
    interleave_images: bool = False,
    prompt: Callable[[dict, str], str] | None = None,
    system: str | None = None,
    prefill: str | None = None,
) -> list[EvalSample]:
    """``prompt``, given a row's raw document and its stored prompt, returns
    the text to send instead (a prompt profile); by default the stored prompt.
    ``system``, if given, is sent as a system message before the user turn.
    ``prefill``, if given, is sent as an assistant message after it: the start
    of the reply, which the server continues."""
    rows = _read_verified(dataset_dir)
    preamble = [ChatMessage(role="system", content=system)] if system else []
    postamble = [ChatMessage(role="assistant", content=prefill)] if prefill else []
    return [
        EvalSample(
            id=row["sample_id"],
            messages=[
                *preamble,
                ChatMessage(
                    role="user",
                    content=_content_blocks(
                        prompt(json.loads(row["raw_doc"]), row["prompt"]) if prompt else row["prompt"],
                        _images_from_row(row),
                        interleave=interleave_images,
                    ),
                ),
                *postamble,
            ],
        )
        for row in rows
    ]
