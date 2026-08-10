# Datasets

Each dataset backs one of the scoreable evals (`pipette_scores.types.EvalId`: `ifbench`, `ifstruct`, `gpqa_diamond`, `math_500`). Some are established public benchmarks, run verbatim; one (`ifstruct`) is a Liquid in-house creation.

## 1. Overview

| Eval | Source | Origin |
|------|--------|--------|
| ifbench | [allenai/IFBench](https://github.com/allenai/IFBench) (vendored in `vendor/ifbench`; verifier code Apache-2.0, dataset ODC-By-1.0) | Public |
| ifstruct | Generated via the Liquid `ifstruct` package (`vendor/ifstruct`) | Liquid in-house |
| gpqa_diamond | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) (Diamond config) | Public (gated, CC BY 4.0) |
| math_500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | Public (MIT) |

## 2. Storage Layout

Datasets live under the data root, keyed by eval id and dataset version:

```
$PIPETTE_SCORES_DATA_DIR/
  datasets/
    {eval_id}/
      {dataset_name}/
        metadata.json        # version, repeats, provenance (upstream repo + revision + content_sha256)
        test.jsonl           # or a Parquet file, depending on the eval's loader
```

- **`PIPETTE_SCORES_DATA_DIR`** — environment variable pointing to the data root. Defaults to the parent directory of the `pipette_scores` package.
- **`dataset_name`** — the pinned dataset version (`ifbench` → `2026.06.1`, `ifstruct` → `release_v1_0`, `gpqa_diamond` → `2026.06.1`, `math_500` → `2026.06.1`).
- **Committed vs materialized** — only the Liquid-owned `ifstruct` dataset is committed to the repo. The third-party datasets (`ifbench`, `gpqa_diamond`, `math_500`) are gitignored and **materialized at deploy** from their pinned upstream revision by `packages/pipette-scores/scripts/build_<eval>_dataset.py` (a `content_sha256` in the metadata is self-asserted at build and re-verified at load).

## 3. Sample IDs

All sample IDs are deterministic 12-character hex strings produced by SHA256 hashing a key derived from the sample content. The hash input varies per eval to ensure uniqueness. For evals with `metadata.repeats = N`, the samples endpoint additionally serves salted `<id>#0 … <id>#(N-1)` variants of each base id.

## 4. Prompt vs Scoring Samples

Each sample exists in two forms: a **prompt** version containing only the chat messages for inference (no answers), and a **scoring** version containing the full ground truth. Both share the same ID, which links a model completion back to its expected answer.

---

## 5. ifbench

Scoring: [ifbench](scoring.md#2-ifbench)

The upstream IFBench test set — **300** single-turn prompts, each pairing a user-style request with one or more machine-checkable constraints (keyword counts, format/casing rules, line structure, etc.). Run verbatim, vendored from [allenai/IFBench](https://github.com/allenai/IFBench) at a pinned commit. The verifier code is Apache-2.0; the dataset rows are ODC-By-1.0 and should be used in accordance with Ai2's Responsible Use Guidelines. Served with `metadata.repeats = 5` (1500 attempt ids), reported as pass@1.

**Dataset version:** `2026.06.1` (materialized at deploy; not committed).

---

## 6. ifstruct

Scoring: [ifstruct](scoring.md#3-ifstruct)

Structured format generation — Liquid in-house dataset. The full **2000-task** IFStruct set (`release_v1_0`). Each task gives an explicit output schema and structural constraints; prompts are humanised and include distractors (fields mentioned then explicitly rejected), so a model passes only by following the final, settled instruction. The model must return JSON or YAML that satisfies the schema. Each prompt is served and scored once (no repeats).

**Dataset version:** `release_v1_0` (committed).

---

## 7. gpqa_diamond

Scoring: [gpqa_diamond](scoring.md#4-gpqa_diamond)

The **Diamond** subset of [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) — all **198** graduate-level, "Google-proof" science questions (biology, physics, chemistry). Each is presented as a 4-option (A–D) MCQ; the options are shuffled deterministically per question and baked into the served prompt. Run verbatim from a pinned HF revision. GPQA is **gated** (CC BY 4.0, questions must not be revealed in plain text), so the dataset is materialized from upstream rather than committed. Served with `metadata.repeats = 5` (990 attempt ids), reported as pass@1.

**Dataset version:** `2026.06.1` (materialized at deploy; not committed).

---

## 8. math_500

Scoring: [math_500](scoring.md#5-math_500)

[HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) — the **500**-problem test split (from OpenAI's PRM800K work) of the Hendrycks MATH benchmark, spanning algebra, geometry, number theory, precalculus, and more at difficulty levels 1–5. Open-ended (not MCQ): the model reasons step by step and gives its final answer in `\boxed{}`. MIT-licensed but, like the other third-party datasets, materialized from a pinned revision rather than committed. Served with `metadata.repeats = 5` (2500 attempt ids), reported as pass@1.

**Dataset version:** `2026.06.1` (materialized at deploy; not committed).
