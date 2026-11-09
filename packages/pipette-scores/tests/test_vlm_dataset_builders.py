"""Unit tests for the VLM dataset builder scripts (synthetic TSV rows, no network).

Prompt-format expectations follow
pipette_scores/scoring/vlm_upstream/PORTING_NOTES.md (VLMEvalKit
``ImageMCQDataset.build_prompt`` @ 8eca8e5). The per-benchmark specs are
loaded from the one-shot scripts in ``packages/pipette-scores/scripts/``.
"""

import base64
import importlib.util
import io
import json
import pathlib

import pandas as pd
import pytest
from PIL import Image

from pipette_scores.dataset_catalog import _vlm_build, _vlm_common

_SCRIPTS_DIR = pathlib.Path(__file__).parents[1] / "scripts"
_EVAL_IDS = ["mme", "mmstar", "ocrbench", "mmmu_val", "realworldqa", "mmbench_en_dev", "pope"]


def _load_script(eval_id: str):
    path = _SCRIPTS_DIR / f"build_{eval_id}_dataset.py"
    spec = importlib.util.spec_from_file_location(f"build_{eval_id}_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCRIPTS = {eval_id: _load_script(eval_id) for eval_id in _EVAL_IDS}
SPECS = {eval_id: mod.SPEC for eval_id, mod in _SCRIPTS.items()}


def _png_b64(size: int = 8, color=(255, 0, 0)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _jpeg_bytes(size: int = 128) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (0, 128, 255)).save(buf, format="JPEG")
    return buf.getvalue()


def test_load_tsv_keeps_literal_none_options(tmp_path):
    # MMMU validation_Geography_15: option D is the text "None" and the answer.
    path = tmp_path / "t.tsv"
    path.write_text("index\tquestion\tA\tB\tC\tD\tanswer\n1\tq\tx\ty\t\tNone\tD\n")
    row = _vlm_build.load_tsv(path).iloc[0]
    assert row["D"] == "None"
    assert pd.isna(row["C"])


# ---------------------------------------------------------------------------
# Prompt format (PORTING_NOTES MCQ spec)
# ---------------------------------------------------------------------------


class TestMcqPrompt:
    def test_full_format_with_hint(self):
        line = {
            "question": "What is shown?",
            "hint": "Look closely.",
            "A": "a cat",
            "B": "a dog",
            "C": "a bird",
            "D": "a fish",
        }
        assert _vlm_build.build_mcq_prompt(line) == (
            "Hint: Look closely.\n"
            "Question: What is shown?\n"
            "Options:\n"
            "A. a cat\n"
            "B. a dog\n"
            "C. a bird\n"
            "D. a fish\n"
            "Please select the correct answer from the options above. \n"
        )

    def test_no_hint_and_missing_options(self):
        line = {"question": "Pick one.", "hint": float("nan"), "A": "yes", "B": "no", "C": float("nan"), "D": None}
        assert _vlm_build.build_mcq_prompt(line) == (
            "Question: Pick one.\nOptions:\nA. yes\nB. no\nPlease select the correct answer from the options above. \n"
        )

    def test_open_question_has_no_options_block(self):
        line = {"question": "Describe the image.", "A": float("nan"), "B": float("nan")}
        assert _vlm_build.build_mcq_prompt(line) == "Question: Describe the image.\n"

    def test_bare_prompt_is_question_verbatim(self):
        q = "Is there a car in the image? Please answer yes or no."
        assert _vlm_build.build_bare_prompt({"question": q}) == q


# ---------------------------------------------------------------------------
# MMBench: circular rotations + image pointer dedup
# ---------------------------------------------------------------------------


class TestMmbench:
    def _df(self):
        img = _png_b64()
        return pd.DataFrame(
            [
                # base question
                {
                    "index": 42,
                    "question": "What color?",
                    "hint": float("nan"),
                    "A": "red",
                    "B": "blue",
                    "C": "green",
                    "D": "gray",
                    "answer": "A",
                    "category": "color",
                    "l2-category": "attribute",
                    "image": img,
                },
                # circular rotation row: image cell is a pointer to index 42
                {
                    "index": 1000042,
                    "question": "What color?",
                    "hint": float("nan"),
                    "A": "blue",
                    "B": "green",
                    "C": "gray",
                    "D": "red",
                    "answer": "D",
                    "category": "color",
                    "l2-category": "attribute",
                    "image": "42",
                },
            ]
        )

    def test_rotation_rows_preserved_with_original_index(self):
        df = _vlm_build.resolve_image_pointers(self._df())
        rows, _ = _vlm_build.build_rows(df, SPECS["mmbench_en_dev"], max_image_px=None)
        assert len(rows) == 2
        indices = [json.loads(r["raw_doc"])["index"] for r in rows]
        assert indices == [42, 1000042]
        # circular grouping key: rotations of one base question share index % 1e6
        assert indices[1] % int(1e6) == indices[0]

    def test_image_pointer_resolves_to_same_bytes(self):
        df = _vlm_build.resolve_image_pointers(self._df())
        rows, _ = _vlm_build.build_rows(df, SPECS["mmbench_en_dev"], max_image_px=None)
        assert rows[0]["_images"] == rows[1]["_images"]
        assert rows[1]["num_images"] == 1

    def test_unresolvable_pointer_raises(self):
        df = self._df()
        df.loc[1, "image"] = "999"  # no such index
        with pytest.raises(ValueError):
            _vlm_build.resolve_image_pointers(df)

    def test_raw_doc_carries_options_answer_and_hint(self):
        df = pd.DataFrame(
            [
                {
                    "index": 7,
                    "question": "Q?",
                    "hint": "H.",
                    "A": "a",
                    "B": "b",
                    "answer": "B",
                    "category": "c",
                    "l2-category": "l2",
                    "image": _png_b64(),
                }
            ]
        )
        rows, _ = _vlm_build.build_rows(df, SPECS["mmbench_en_dev"], max_image_px=None)
        doc = json.loads(rows[0]["raw_doc"])
        assert doc["A"] == "a" and doc["B"] == "b"
        assert doc["answer"] == "B"
        assert doc["hint"] == "H."
        assert doc["category"] == "c" and doc["l2-category"] == "l2"
        assert "image" not in doc
        assert rows[0]["prompt"].startswith("Hint: H.\nQuestion: Q?\n")


# ---------------------------------------------------------------------------
# MME: pairing key
# ---------------------------------------------------------------------------


class TestMme:
    def test_raw_doc_carries_pairing_key_shared_by_question_pair(self):
        img = _png_b64()
        df = pd.DataFrame(
            [
                {
                    "index": 0,
                    "category": "existence",
                    "image_path": "existence/0001.jpg",
                    "question": "Is there a dog? Please answer yes or no.",
                    "answer": "Yes",
                    "image": img,
                },
                {
                    "index": 1,
                    "category": "existence",
                    "image_path": "existence/0001.jpg",
                    "question": "Is there a cat? Please answer yes or no.",
                    "answer": "No",
                    "image": img,
                },
            ]
        )
        rows, _ = _vlm_build.build_rows(df, SPECS["mme"], max_image_px=None)
        docs = [json.loads(r["raw_doc"]) for r in rows]
        assert docs[0]["image_path"] == docs[1]["image_path"] == "existence/0001.jpg"
        assert docs[0]["answer"] == "Yes" and docs[1]["answer"] == "No"
        assert rows[0]["prompt"] == "Is there a dog? Please answer yes or no."
        assert rows[0]["sample_id"] != rows[1]["sample_id"]


# ---------------------------------------------------------------------------
# MMMU: VAL filter, open questions, multi-image cells
# ---------------------------------------------------------------------------


class TestMmmu:
    def test_val_filter_excludes_dev_and_asserts_count(self):
        df = pd.DataFrame(
            {
                "index": range(1050),
                "split": ["dev"] * 150 + ["validation"] * 900,
            }
        )
        out = _SCRIPTS["mmmu_val"].keep_validation(df)
        assert len(out) == 900
        assert (out["split"] == "validation").all()

        with pytest.raises(SystemExit):
            _SCRIPTS["mmmu_val"].keep_validation(df.head(1000))

    def test_multi_image_cell_and_placeholder_prompt(self):
        img_a, img_b = _png_b64(color=(255, 0, 0)), _png_b64(color=(0, 255, 0))
        df = pd.DataFrame(
            [
                {
                    "index": 1,
                    "question": "Compare <image 1> with <image 2>.",
                    "split": "validation",
                    "A": "left",
                    "B": "right",
                    "answer": "A",
                    "category": "Art",
                    "image": str([img_a, img_b]),
                }
            ]
        )
        rows, _ = _vlm_build.build_rows(df, SPECS["mmmu_val"], max_image_px=None)
        assert rows[0]["num_images"] == 2
        assert rows[0]["_images"] == [base64.b64decode(img_a), base64.b64decode(img_b)]
        # placeholders stay in the prompt text (interleaving happens at load time)
        assert "<image 1>" in rows[0]["prompt"] and "<image 2>" in rows[0]["prompt"]

    def test_open_question_options_stay_missing_in_raw_doc(self):
        df = pd.DataFrame(
            [
                {
                    "index": 2,
                    "question": "How many?",
                    "split": "validation",
                    "A": float("nan"),
                    "B": float("nan"),
                    "answer": "42",
                    "category": "Math",
                    "image": _png_b64(),
                }
            ]
        )
        rows, _ = _vlm_build.build_rows(df, SPECS["mmmu_val"], max_image_px=None)
        doc = json.loads(rows[0]["raw_doc"])
        assert doc["A"] is None and doc["B"] is None
        assert doc["answer"] == "42"
        assert rows[0]["prompt"] == "Question: How many?\n"


# ---------------------------------------------------------------------------
# OCRBench / POPE raw_doc fields
# ---------------------------------------------------------------------------


class TestYornAndOcr:
    def test_ocrbench_category_case_and_answer_list(self):
        df = pd.DataFrame(
            [
                {
                    "index": 3,
                    "image": _png_b64(),
                    "question": "Recognize the expression.",
                    "answer": "['x^2', 'x**2']",
                    "category": "Handwritten Mathematical Expression Recognition",
                }
            ]
        )
        rows, _ = _vlm_build.build_rows(df, SPECS["ocrbench"], max_image_px=None)
        doc = json.loads(rows[0]["raw_doc"])
        assert doc["category"] == "Handwritten Mathematical Expression Recognition"
        assert doc["answer"] == "['x^2', 'x**2']"  # stringified list, verbatim
        assert rows[0]["prompt"] == "Recognize the expression."

    def test_pope_category_and_bare_prompt(self):
        df = pd.DataFrame(
            [
                {
                    "image": _png_b64(),
                    "question": "Is there a bottle in the image?",
                    "answer": "Yes",
                    "category": "adversarial,popular",
                    "index": 9,
                }
            ]
        )
        rows, _ = _vlm_build.build_rows(df, SPECS["pope"], max_image_px=None)
        doc = json.loads(rows[0]["raw_doc"])
        assert doc["category"] == "adversarial,popular"
        assert rows[0]["prompt"] == "Is there a bottle in the image?"
        assert json.loads(rows[0]["answer"]) == "Yes"


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------


class TestImages:
    def test_default_keeps_original_bytes(self):
        data = _jpeg_bytes(128)
        assert _vlm_build.maybe_resize(data, None) is data

    def test_max_image_px_downscales_oversized(self):
        data = _jpeg_bytes(128)
        out = _vlm_build.maybe_resize(data, 64)
        with Image.open(io.BytesIO(out)) as img:
            assert max(img.size) == 64
            assert img.format == "JPEG"

    def test_max_image_px_keeps_small_images_untouched(self):
        data = _jpeg_bytes(32)
        assert _vlm_build.maybe_resize(data, 64) is data

    def test_sniff_formats(self):
        assert _vlm_build.sniff_image_format(_jpeg_bytes()) == "jpeg"
        assert _vlm_build.sniff_image_format(base64.b64decode(_png_b64())) == "png"


# ---------------------------------------------------------------------------
# Specs, metadata + license fields
# ---------------------------------------------------------------------------


class TestMetadataAndSpecs:
    def test_all_seven_builders_have_specs(self):
        assert sorted(SPECS) == sorted(_EVAL_IDS)
        for eval_id, spec in SPECS.items():
            assert spec.eval_id == eval_id

    def test_every_spec_has_license_fields_and_pin(self):
        for spec in SPECS.values():
            assert spec.license
            assert spec.license_notes
            assert spec.url.startswith("https://")
            assert len(spec.md5) == 32

    def test_realworldqa_is_nd_with_prominent_notice(self):
        spec = SPECS["realworldqa"]
        assert "CC-BY-ND" in spec.license
        assert spec.extra_notice and "MUST NOT be redistributed" in spec.extra_notice

    def test_metadata_json_fields(self, tmp_path):
        spec = SPECS["pope"]
        _vlm_build.write_metadata(
            tmp_path, spec, "full", 3, "d" * 64, "e" * 64, {"png": 3}, max_image_px=None, limit=None
        )
        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["eval_id"] == "pope"
        assert meta["dataset"] == "full"
        assert meta["source"]["url"] == spec.url
        assert meta["source"]["md5"] == spec.md5
        assert meta["num_samples"] == 3
        assert meta["content_sha256"] == "d" * 64
        assert meta["metadata_content_sha256"] == "e" * 64
        assert meta["max_image_px"] is None
        assert meta["license"] == spec.license
        assert meta["license_notes"] == spec.license_notes
        assert "built_at" in meta
        assert "internal use only" in meta["redistribution"]

    def test_content_sha256_is_order_independent_and_image_sensitive(self):
        row = {
            "sample_id": "abc",
            "prompt": "p",
            "answer": None,
            "raw_doc": "{}",
            "_images": [b"\x01\x02"],
        }
        row2 = {**row, "sample_id": "def"}
        assert _vlm_build.content_sha256([row, row2]) == _vlm_build.content_sha256([row2, row])
        assert _vlm_build.content_sha256([row]) != _vlm_build.content_sha256([{**row, "_images": [b"\x01\x03"]}])


# ---------------------------------------------------------------------------
# Round-trip: builder output -> pipette_scores VLM loader
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_parquet_loads_through_vlm_common(self, tmp_path):
        img_a, img_b = _png_b64(color=(255, 0, 0)), _png_b64(color=(0, 255, 0))
        df = pd.DataFrame(
            [
                {
                    "index": 1,
                    "question": "Compare <image 1> and <image 2>.",
                    "split": "validation",
                    "A": "same",
                    "B": "different",
                    "answer": "B",
                    "category": "Art",
                    "image": str([img_a, img_b]),
                },
                {
                    "index": 2,
                    "question": "What is this?",
                    "split": "validation",
                    "A": "x",
                    "B": "y",
                    "answer": "A",
                    "category": "Art",
                    "image": img_a,
                },
            ]
        )
        rows, _ = _vlm_build.build_rows(df, SPECS["mmmu_val"], max_image_px=None)
        _vlm_build.write_dataset(tmp_path, rows)

        eval_samples = _vlm_common.load_eval_samples(tmp_path)
        assert len(eval_samples) == 2
        assert eval_samples[0].raw_doc["answer"] == "B"

        prompts = _vlm_common.load_prompt_samples(tmp_path, interleave_images=True)
        blocks = prompts[0].messages[0].content
        # <image N> placeholders are interleaved: text, image, text, image, text
        kinds = [b["type"] for b in blocks]
        assert kinds == ["text", "image_url", "text", "image_url", "text"]
        assert blocks[0]["text"] == "Question: Compare "
        # PNG bytes are emitted with a png mime, not the historical jpeg default
        assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")

        # row without placeholders: images first, then the full prompt text
        kinds2 = [b["type"] for b in prompts[1].messages[0].content]
        assert kinds2 == ["image_url", "text"]
