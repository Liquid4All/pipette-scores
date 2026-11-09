#!/usr/bin/env python
"""Integrity verifier for the 7 built VLM benchmark datasets.

Run from the repo root after `build_vlm_datasets.sbatch` (or the individual
builders):

    uv run --no-sync --project packages/pipette-scores \
        python packages/pipette-scores/scripts/verify_vlm_datasets.py \
        [--datasets-root datasets] [--version 2026.10.1] [--evals mme,pope,...]

Checks, per eval:
  - metadata.json present with num_samples / source / license fields
  - parquet row count == metadata num_samples (and == the known full-set size
    where one is pinned below)
  - sample_ids unique and non-empty; prompts non-empty; raw_doc parses as JSON
    and carries the per-eval fields the scorers need (PORTING_NOTES.md)
  - EVERY image decodes: PIL verify() (header pass) + full load() (pixel pass)
  - image_i columns are populated exactly for i < num_images
  - the image-inclusive content_sha256, recomputed from the parquet, equals
    metadata.json's and, for a full original-resolution build, the builder
    script's pinned EXPECTED_CONTENT_SHA256 (so a substituted image fails)
  - loader round-trip: dataset_catalog.load_eval_samples / load_prompt_samples
    return the same count and well-formed multimodal messages (data-URI mime)
  - mmbench_en_dev: circular rotation groups (index % 1e6) are complete enough
    to score (every group's rotations all present)
  - pope: every exploded category is one of random/popular/adversarial

Exit code 0 = all checks pass; 1 = any failure (summary printed either way).
"""

import argparse
import importlib
import io
import json
import pathlib
import sys
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from PIL import Image

from pipette_scores.dataset_catalog._vlm_build import content_sha256

# Known full-set sizes (None = not pinned yet; report only).
EXPECTED_ROWS: dict[str, int | None] = {
    "mme": 2374,
    "mmstar": 1500,
    "ocrbench": 1000,
    "mmmu_val": 900,
    "realworldqa": 765,
    "mmbench_en_dev": 4876,
    "pope": 5127,
}

# raw_doc fields each scorer consumes (see scoring/vlm_upstream/PORTING_NOTES.md).
REQUIRED_RAW_DOC_KEYS: dict[str, set[str]] = {
    "mme": {"category"},  # + image_path or question_id (pairing key, checked below)
    "mmstar": {"answer"},
    "ocrbench": {"category"},
    "mmmu_val": {"answer"},
    "realworldqa": {"answer"},
    "mmbench_en_dev": {"answer", "index"},
    "pope": {"answer", "category"},
}

POPE_CATEGORIES = {"random", "popular", "adversarial"}


def _builder_pin(eval_id: str) -> str:
    """The image-inclusive digest the eval's builder script pins: a trusted
    value from outside the dataset artifact."""
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    try:
        return importlib.import_module(f"build_{eval_id}_dataset").SPEC.expected_content_sha256
    finally:
        sys.path.pop(0)


def fail(errors: list[str], msg: str) -> None:
    errors.append(msg)
    print(f"    FAIL: {msg}")


def verify_eval(eval_id: str, root: pathlib.Path, version: str) -> list[str]:
    errors: list[str] = []
    d = root / eval_id / version
    print(f"\n=== {eval_id} ({d}) ===")
    meta_path = d / "metadata.json"
    pq_path = d / "test.parquet"
    if not meta_path.exists() or not pq_path.exists():
        fail(errors, f"missing {meta_path.name if not meta_path.exists() else pq_path.name}")
        return errors

    meta = json.loads(meta_path.read_text())
    for key in ("num_samples", "source", "license"):
        if key not in meta:
            fail(errors, f"metadata.json missing '{key}'")

    table = pq.read_table(pq_path)
    n = table.num_rows
    print(f"    rows: {n}")
    if meta.get("num_samples") != n:
        fail(errors, f"metadata num_samples={meta.get('num_samples')} != parquet rows={n}")
    expected = EXPECTED_ROWS.get(eval_id)
    if expected is not None and n != expected:
        fail(errors, f"expected full-set size {expected}, got {n}")

    cols = table.column_names
    image_cols = sorted((c for c in cols if c.startswith("image_")), key=lambda c: int(c.split("_")[1]))
    rows = table.to_pylist()

    # --- id/prompt/raw_doc checks ---
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        fail(errors, f"{len(ids) - len(set(ids))} duplicate sample_ids")
    if any(not i for i in ids):
        fail(errors, "empty sample_id present")
    empty_prompts = sum(1 for r in rows if not (r["prompt"] or "").strip())
    if empty_prompts:
        fail(errors, f"{empty_prompts} empty prompts")

    required = REQUIRED_RAW_DOC_KEYS[eval_id]
    bad_raw, missing_keys = 0, Counter()
    mme_missing_pair_key = 0
    for r in rows:
        try:
            doc = json.loads(r["raw_doc"])
        except (TypeError, json.JSONDecodeError):
            bad_raw += 1
            continue
        for k in required:
            if k not in doc:
                missing_keys[k] += 1
        if eval_id == "mme" and not (doc.get("image_path") or doc.get("question_id")):
            mme_missing_pair_key += 1
    if bad_raw:
        fail(errors, f"{bad_raw} raw_doc values are not valid JSON")
    for k, c in missing_keys.items():
        fail(errors, f"raw_doc missing '{k}' in {c} rows")
    if mme_missing_pair_key:
        fail(errors, f"{mme_missing_pair_key} MME rows lack a pairing key (image_path/question_id)")

    # --- image integrity: decode every image fully ---
    corrupt = 0
    fmt_counter: Counter = Counter()
    num_images_mismatch = 0
    for idx, r in enumerate(rows):
        n_img = r["num_images"]
        for i, col in enumerate(image_cols):
            blob = r[col]
            if i < n_img:
                if blob is None:
                    num_images_mismatch += 1
                    continue
                try:
                    img = Image.open(io.BytesIO(blob))
                    img.verify()  # header/CRC pass
                    img = Image.open(io.BytesIO(blob))
                    img.load()  # full pixel decode
                    fmt_counter[img.format] += 1
                except Exception as e:  # noqa: BLE001 — any decode failure is corruption
                    corrupt += 1
                    if corrupt <= 3:
                        print(f"    corrupt image: row {idx} ({r['sample_id']}) {col}: {e}")
            elif blob is not None:
                num_images_mismatch += 1
    total_imgs = sum(fmt_counter.values())
    print(f"    images decoded: {total_imgs}  formats: {dict(fmt_counter)}")
    if corrupt:
        fail(errors, f"{corrupt} corrupt/undecodable images")
    if num_images_mismatch:
        fail(errors, f"{num_images_mismatch} rows where image_i population != num_images")

    # --- content: the image-inclusive digest, against the build's and the pin ---
    # A decodable image is not the right image: only the digest over every
    # image's bytes tells a substituted one apart. The serving loader checks
    # the image-free metadata digest alone.
    digest = content_sha256([{**r, "_images": [r[col] for col in image_cols[: r["num_images"]]]} for r in rows])
    if digest != meta.get("content_sha256"):
        fail(errors, f"content_sha256 {digest} != metadata.json's {meta.get('content_sha256')}")
    pinned = _builder_pin(eval_id)
    if meta.get("max_image_px") is None and meta.get("limit") is None and pinned:
        if digest != pinned:
            fail(errors, f"content_sha256 {digest} != the builder's pinned {pinned}")
        else:
            print("    content_sha256 matches the builder's pin")
    else:
        print("    note: a resized or partial build; content_sha256 checked against metadata.json only")

    # --- per-eval structural checks ---
    if eval_id == "mmbench_en_dev":
        groups: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            doc = json.loads(r["raw_doc"])
            groups[int(doc["index"]) % 1_000_000].append(doc)
        sizes = Counter(len(v) for v in groups.values())
        print(f"    circular groups: {len(groups)}  group-size distribution: {dict(sizes)}")
        # Upstream MMBench_DEV_EN_V11.tsv itself ships some groups with fewer
        # rotations than options (28 groups) — VLMEvalKit scores those as-is,
        # so option-count shortfall is
        # informational. A build-side row loss instead shows up as
        # NON-CONSECUTIVE or duplicated rotation offsets: fail only on that.
        short, broken = 0, 0
        for g in groups.values():
            offsets = sorted(int(doc["index"]) // 1_000_000 for doc in g)
            if offsets != list(range(len(offsets))):
                broken += 1
                continue
            n_opts = sum(1 for k in ("A", "B", "C", "D") if (g[0].get(k) or "") != "")
            if len(g) != n_opts:
                short += 1
        if short:
            print(f"    note: {short} groups have fewer rotations than options (upstream ships these)")
        if broken:
            fail(
                errors,
                f"{broken} circular groups with non-consecutive/duplicate rotation offsets (build-side row loss)",
            )
    if eval_id == "pope":
        bad_cat = Counter()
        for r in rows:
            doc = json.loads(r["raw_doc"])
            for c in str(doc.get("category", "")).split(","):
                if c.strip() and c.strip() not in POPE_CATEGORIES:
                    bad_cat[c.strip()] += 1
        if bad_cat:
            fail(errors, f"unexpected POPE categories: {dict(bad_cat)}")

    # --- loader round-trip ---
    from pipette_scores import dataset_catalog
    from pipette_scores.types import EvalId

    eid = EvalId(eval_id)
    scoring_samples = dataset_catalog.load_eval_samples(eid, version, root_dir=root.parent)
    prompt_samples = dataset_catalog.load_prompt_samples(eid, version, root_dir=root.parent)
    if len(scoring_samples) != n or len(prompt_samples) != n:
        fail(errors, f"loader round-trip counts: eval={len(scoring_samples)} prompt={len(prompt_samples)} != {n}")
    else:
        first = prompt_samples[0]
        content = first.messages[0].content
        blocks = content if isinstance(content, list) else []
        img_blocks = [b for b in blocks if b.get("type") == "image_url"]
        ok_uri = all(b["image_url"]["url"].startswith("data:image/") for b in img_blocks)
        if not img_blocks or not ok_uri:
            fail(errors, f"prompt sample malformed: {len(img_blocks)} image blocks, data-URI ok={ok_uri}")
        else:
            print(f"    loader round-trip OK ({len(img_blocks)} image block(s) in first sample)")

    if not errors:
        print("    PASS")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", default="datasets")
    ap.add_argument("--version", default="2026.10.1")
    ap.add_argument("--evals", default=",".join(EXPECTED_ROWS))
    args = ap.parse_args()

    root = pathlib.Path(args.datasets_root).resolve()
    all_errors: dict[str, list[str]] = {}
    for eval_id in args.evals.split(","):
        eval_id = eval_id.strip()
        all_errors[eval_id] = verify_eval(eval_id, root, args.version)

    print("\n================ SUMMARY ================")
    failed = {k: v for k, v in all_errors.items() if v}
    for eval_id in all_errors:
        print(f"  {eval_id:16s} {'FAIL (' + str(len(all_errors[eval_id])) + ')' if eval_id in failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
