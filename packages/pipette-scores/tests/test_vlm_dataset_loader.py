"""Unit tests for the shared VLM dataset loader (_vlm_common)."""

from pipette_scores.dataset_catalog._vlm_common import (
    _content_blocks,
    _image_format,
    _image_to_data_uri,
    _interleaved_blocks,
)

_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 8
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
_WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 4


class TestImageMime:
    def test_sniffs_common_formats(self):
        assert _image_format(_JPEG) == "jpeg"
        assert _image_format(_PNG) == "png"
        assert _image_format(_WEBP) == "webp"
        assert _image_format(b"GIF89a\x00") == "gif"
        assert _image_format(b"BM\x00\x00") == "bmp"

    def test_unknown_defaults_to_jpeg(self):
        assert _image_format(b"\x00\x01\x02\x03") == "jpeg"

    def test_data_uri_carries_sniffed_mime(self):
        assert _image_to_data_uri(_PNG).startswith("data:image/png;base64,")
        assert _image_to_data_uri(_JPEG).startswith("data:image/jpeg;base64,")


class TestInterleaving:
    def test_no_placeholder_returns_none(self):
        assert _interleaved_blocks("Question: what?\n", [_PNG]) is None

    def test_no_images_returns_none(self):
        assert _interleaved_blocks("See <image 1>.", []) is None

    def test_mmmu_style_interleave(self):
        blocks = _interleaved_blocks("Compare <image 1> with <image 2>.", [_PNG, _JPEG])
        assert [b["type"] for b in blocks] == ["text", "image_url", "text", "image_url", "text"]
        assert blocks[0]["text"] == "Compare "
        assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert blocks[3]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert blocks[4]["text"] == "."

    def test_repeated_reference_reuses_image(self):
        blocks = _interleaved_blocks("<image 1> vs <image 1>", [_PNG])
        assert [b["type"] for b in blocks] == ["image_url", "text", "image_url"]

    def test_malformed_placeholder_falls_back(self):
        # not "<image N>" shaped -> None (caller uses images-first layout)
        assert _interleaved_blocks("a <image one> b", [_PNG]) is None
        assert _interleaved_blocks("a <image 3> b", [_PNG]) is None  # out of range

    def test_content_blocks_default_layout(self):
        blocks = _content_blocks("plain question", [_PNG, _JPEG])
        assert [b["type"] for b in blocks] == ["image_url", "image_url", "text"]
        assert blocks[2]["text"] == "plain question"

    def test_placeholder_stays_text_unless_interleaving(self):
        # Upstream splits <image N> for MMMU only; MMStar keeps it as text.
        blocks = _content_blocks("See <image 1>.", [_PNG])
        assert [b["type"] for b in blocks] == ["image_url", "text"]
        assert blocks[1]["text"] == "See <image 1>."
        interleaved = _content_blocks("See <image 1>.", [_PNG], interleave=True)
        assert [b["type"] for b in interleaved] == ["text", "image_url", "text"]


class TestLoadTimeIntegrity:
    """The loader re-verifies the cheap image-free digest from metadata.json,
    the VLM analogue of the text loaders' content_sha256 re-check. Backward
    compatible: no digest in metadata → no check (datasets built before it)."""

    def _write_dataset(self, tmp_path, meta_extra: dict | None):
        import json

        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [
            {
                "sample_id": "s1",
                "doc_id": "d1",
                "prompt": "Q1?",
                "answer": json.dumps("A"),
                "raw_doc": json.dumps({"k": 1}),
                "num_images": 0,
            },
            {
                "sample_id": "s2",
                "doc_id": "d2",
                "prompt": "Q2?",
                "answer": json.dumps("B"),
                "raw_doc": json.dumps({"k": 2}),
                "num_images": 0,
            },
        ]
        d = tmp_path / "ds"
        d.mkdir()
        pq.write_table(pa.Table.from_pylist(rows), d / "test.parquet")
        meta = {"eval_id": "x", "num_samples": len(rows)}
        if meta_extra:
            meta.update(meta_extra)
        (d / "metadata.json").write_text(json.dumps(meta))
        return d, rows

    def test_matching_digest_loads(self, tmp_path):
        from pipette_scores.dataset_catalog._vlm_common import (
            load_eval_samples,
            metadata_content_sha256,
        )

        d, rows = self._write_dataset(tmp_path, {"metadata_content_sha256": None})
        # write the real digest
        import json

        digest = metadata_content_sha256(rows)
        meta = json.loads((d / "metadata.json").read_text())
        meta["metadata_content_sha256"] = digest
        (d / "metadata.json").write_text(json.dumps(meta))

        samples = load_eval_samples(d)
        assert [s.id for s in samples] == ["s1", "s2"]

    def test_mismatched_digest_raises(self, tmp_path):
        import pytest

        from pipette_scores.dataset_catalog._vlm_common import load_eval_samples

        d, _ = self._write_dataset(tmp_path, {"metadata_content_sha256": "deadbeef" * 8})
        with pytest.raises(ValueError, match="integrity check failed"):
            load_eval_samples(d)

    def test_absent_digest_skips_check(self, tmp_path):
        # No metadata_content_sha256 → loads unguarded (pre-feature datasets).
        from pipette_scores.dataset_catalog._vlm_common import load_eval_samples

        d, _ = self._write_dataset(tmp_path, None)
        assert len(load_eval_samples(d)) == 2

    def test_digest_is_image_free_and_order_independent(self, tmp_path):
        from pipette_scores.dataset_catalog._vlm_common import metadata_content_sha256

        _, rows = self._write_dataset(tmp_path, None)
        # order-independent
        assert metadata_content_sha256(rows) == metadata_content_sha256(list(reversed(rows)))
        # image bytes don't affect it (adding an image column is a no-op)
        with_img = [{**r, "image_0": b"\xff\xd8\xff", "num_images": 1} for r in rows]
        assert metadata_content_sha256(with_img) == metadata_content_sha256(rows)
