# Datasets

Each dataset backs one of the scoreable evals (`pipette_scores.types.EvalId`): the text evals `ifbench`, `ifstruct`, `gpqa_diamond`, `math_500` and the VLM evals `mme`, `mmstar`, `ocrbench`, `mmmu_val`, `realworldqa`, `mmbench_en_dev`, `pope`. Some are established public benchmarks, run verbatim; one (`ifstruct`) is a Liquid in-house creation.

## 1. Overview

| Eval | Source | Origin |
|------|--------|--------|
| ifbench | [allenai/IFBench](https://github.com/allenai/IFBench) (test set from the `vendor/ifbench` submodule, checkers in `vendor/ifbench-checkers`; verifier code Apache-2.0, dataset ODC-By-1.0) | Public |
| ifstruct | Generated via the Liquid `ifstruct` package (`vendor/ifstruct`) | Liquid in-house |
| gpqa_diamond | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) (Diamond config) | Public (gated, CC BY 4.0) |
| math_500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | Public (MIT) |
| mme | VLMEvalKit pinned `MME.tsv` | Public (no license; academic/non-commercial terms) |
| mmstar | VLMEvalKit pinned `MMStar.tsv` | Public (no license; inherits CC-BY-NC[-SA] sources) |
| ocrbench | VLMEvalKit pinned `OCRBench.tsv` | Public (MIT code; mixed image sources) |
| mmmu_val | VLMEvalKit pinned `MMMU_DEV_VAL.tsv` (VAL rows) | Public (Apache-2.0 card; uncleared images) |
| realworldqa | VLMEvalKit pinned `RealWorldQA.tsv` | Public (**CC-BY-ND-4.0**) |
| mmbench_en_dev | VLMEvalKit pinned `MMBench_DEV_EN_V11.tsv` | Public (Apache-2.0 repo; uncleared images) |
| pope | VLMEvalKit pinned `POPE.tsv` | Public (MIT annotations; COCO/Flickr images) |

All 7 VLM datasets are built by `packages/pipette-scores/scripts/build_<eval>_dataset.py`, never committed here, and kept for internal use only in the organization's private storage; they are not redistributable in transformed form. Licensing details and distribution rules: [dataset-licenses.md](../../../docs/dataset-licenses.md).

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
- **`dataset_name`** — the pinned dataset version (`ifbench` → `2026.06.1`, `ifstruct` → `release_v1_0`, `gpqa_diamond` → `2026.06.1`, `math_500` → `2026.06.1` and `2026.10.1`, all 7 VLM evals → `2026.10.1`).
- **Committed vs materialized** — only the Liquid-owned `ifstruct` dataset is committed to the repo. The third-party datasets (`ifbench`, `gpqa_diamond`, `math_500`, and the 7 VLM evals) are gitignored and **materialized at deploy** from their pinned upstream revision by `packages/pipette-scores/scripts/build_<eval>_dataset.py` (a `content_sha256` in the metadata is self-asserted at build; the text-eval loaders re-verify it at load).

## 3. Sample IDs

All sample IDs are deterministic 12-character hex strings produced by SHA256 hashing a key derived from the sample content. The hash input varies per eval to ensure uniqueness. For evals with `metadata.repeats = N`, the samples endpoint additionally serves salted `<id>#0 … <id>#(N-1)` variants of each base id.

## 4. Prompt vs Scoring Samples

Each sample exists in two forms: a **prompt** version containing only the chat messages for inference (no answers), and a **scoring** version containing the full ground truth. Both share the same ID, which links a model completion back to its expected answer.

**Prompt profiles (VLM evals).** By default every model gets the same prompt: the dataset's own VLMEvalKit prompt, stored with the dataset. VLMEvalKit itself prompts some model families differently, some model cards ask for a particular prompt when benchmarking, and the published numbers come from those prompts. A prompt profile, asked for with `?prompt_profile=` on the samples endpoint, rebuilds each prompt's text from the row's `raw_doc` the way VLMEvalKit's wrapper or the model card does for that family. Only the text changes: images, their placement, the dataset and the scoring stay the same. Each profile is one short function in [`dataset_catalog/prompt_profiles.py`](../pipette_scores/dataset_catalog/prompt_profiles.py) citing what it copies (VLMEvalKit `8eca8e5` lines, or the model card), and the tests spell out every profile's text.

| Profile | Copies the prompts of | What changes |
|---------|------|------|
| `internvl` | InternVL3.5 | multiple choice ends "Answer with the option's letter from the given choices directly."; MME and OCRBench ask for a single word or phrase |
| `internvl_r1` | InternVL3.5 with thinking | the `internvl` prompts, after InternVL's R1 system prompt (its thinking mode has no chat-template switch) |
| `granite_vision` | Granite Vision (IBM's wrapper, written for 3.x) | the same multiple-choice prompt; OCRBench asks for one word, a short phrase or a full address |
| `minicpm_v` | MiniCPM-V, answering directly | multiple choice ends with the letter instruction; OCRBench asks for a single word or phrase |
| `minicpm_v_cot` | MiniCPM-V, reasoning step by step | step-by-step instructions ending in an "Answer: ..." line |
| `lfm_vl` | LFM2-VL, LFM2.5-VL-1.6B | "Please answer directly with only the final answer, do not give any explanation." after every prompt |
| `lfm2_5_vl_3b` | LFM2.5-VL-3B | MME prompts by category; OCRBench asks for a concise answer; the direct-answer line on MMStar, RealWorldQA and MMBench |
| `qwen3_5` | Qwen3.5, from its model card's benchmarking best practices | multiple choice ends "Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`."; the scorer reads that field |
| `glm_v` | GLM-4.6V-Flash (VLMEvalKit's `GLMThinking`) | the dataset prompts; POPE adds a line "Please answer yes or no." |
| `kimi_vl` | Kimi-VL-A3B-Thinking-2506 (VLMEvalKit's `KimiVL`) | on MMMU and MMStar the closing "Please select the correct answer..." line gives way to Kimi's step-by-step instruction ending in `'Answer: $\boxed{LETTER}$'`, appended to every piece of text (MMMU's question is split at its images); the other benchmarks keep the dataset prompts. The scorer reads the text after the model's `◁think▷…◁/think▷` like any other reasoning and, on MMMU and MMStar, its last `\boxed{}` |
| `nemotron_think` | Nemotron-Nano-12B-v2-VL with thinking | the dataset prompts, each starting with a newline after the image (on MMMU, whose images sit at their `<image N>` placeholders, before the first image), after the system message `/think` (its thinking switch; the chat template ignores `enable_thinking`) |
| `phi4_rv_think` | Phi-4-reasoning-vision-15B with thinking forced | the dataset prompts, no system message, and the conversation ends with an assistant message `<think>` that the server continues (the model card's forced-thinking form; its template has no switch). The scorer knows the reply continues a thought the prompt opened ([scoring.md §7](scoring.md#7-reasoning-markup-and-cut-off-reasoning)). Needs a runtime that continues a final assistant message, as llama-server does by default |
| `mimo_vl` | MiMo-VL-7B-RL-2508 (optional) | the dataset prompts under the system message "You are a helpful assistant." |

---

## 5. ifbench

Scoring: [ifbench](scoring.md#2-ifbench)

The upstream IFBench test set — **300** single-turn prompts, each pairing a user-style request with one or more machine-checkable constraints (keyword counts, format/casing rules, line structure, etc.). Run verbatim, read from the [allenai/IFBench](https://github.com/allenai/IFBench) submodule at a pinned commit. The verifier code (the revision liquid_evals uses, see [scoring](scoring.md#2-ifbench)) is Apache-2.0; the dataset rows are ODC-By-1.0 and should be used in accordance with Ai2's Responsible Use Guidelines. Served with `metadata.repeats = 5` (1500 attempt ids), reported as pass@1.

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

[HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) — the **500**-problem test split (from OpenAI's PRM800K work) of the Hendrycks MATH benchmark, spanning algebra, geometry, number theory, precalculus, and more at difficulty levels 1–5. Open-ended (not MCQ): the model reasons step by step and gives its final answer in `\boxed{}`. MIT-licensed but, like the other third-party datasets, materialized from a pinned revision rather than committed. Reported as pass@1.

Two versions hold the same 500 problems and the upstream `answer` column as gold. Each reproduces one liquid_evals track, prompt and grader together:

| Version | liquid_evals track | Prompt (one user message) | Repeats | `metadata.grader` |
|---------|--------------------|---------------------------|---------|-------------------|
| `2026.06.1` | `math500_std` (Artificial Analysis methodology) | "Solve the following math problem step by step. Put your answer inside `\boxed{}`." / problem / "Remember to put your answer inside `\boxed{}`." | 5 (2500 attempt ids) | absent, so `math_generic` |
| `2026.10.1` | `math500`, the MATH-500 on Liquid's model cards | problem / "Please reason step by step, and put your final answer within `\boxed{}`." | 5 (2500 attempt ids) | `liquid_evals_math500` |

liquid_evals' `math500` task takes its gold from `extract_math_answer(solution)`; the builder checks that this equals `answer` on all 500 rows. One `math500` run is one pass over the 500 problems, and the model-card number is the mean of 5 runs, so `2026.10.1` serves 5 repeats: their pass@1 is that mean.

**Dataset versions:** `2026.06.1`, `2026.10.1` (materialized at deploy; not committed).

---

## 9. mme

Scoring: [PORTING_NOTES.md](../pipette_scores/scoring/vlm_upstream/PORTING_NOTES.md)

**2374** yes/no perception + cognition questions over 14 categories. Built from VLMEvalKit's pinned `MME.tsv` by `scripts/build_mme_dataset.py`; the prompt is the raw question and `raw_doc.image_path` pairs the two questions per image for `acc_plus`. No declared license — the authors' terms are academic-research-only / non-commercial (self-assess, especially commercial use).

**Dataset version:** `2026.10.1` (materialized at deploy; not committed).

---

## 10. mmstar

Scoring: [PORTING_NOTES.md](../pipette_scores/scoring/vlm_upstream/PORTING_NOTES.md)

**1500** vision-indispensable 4-option MCQs. Built from VLMEvalKit's pinned `MMStar.tsv` by `scripts/build_mmstar_dataset.py` with the VLMEvalKit MCQ prompt. No declared license; inherits CC-BY-NC (SEED-Bench) and CC-BY-NC-SA (ScienceQA) content terms.

**Dataset version:** `2026.10.1` (materialized at deploy; not committed).

---

## 11. ocrbench

Scoring: [PORTING_NOTES.md](../pipette_scores/scoring/vlm_upstream/PORTING_NOTES.md)

**1000** OCR questions over 10 categories (Final Score / 1000). Built from VLMEvalKit's pinned `OCRBench.tsv` by `scripts/build_ocrbench_dataset.py`; bare-question prompt, `answer` kept as the upstream stringified list, `category` case-exact. MIT benchmark code over ~29 mixed-terms image sources.

**Dataset version:** `2026.10.1` (materialized at deploy; not committed).

---

## 12. mmmu_val

Scoring: [PORTING_NOTES.md](../pipette_scores/scoring/vlm_upstream/PORTING_NOTES.md)

The **900** validation rows of VLMEvalKit's pinned `MMMU_DEV_VAL.tsv` (the 150 dev rows are excluded), built by `scripts/build_mmmu_val_dataset.py`. MCQ prompt with option columns A–I; open questions keep no options and are rewritten by the scorer (`MMMU_preproc`). Multi-image questions keep their `<image N>` placeholders; the loader interleaves images at those positions. Apache-2.0 card over uncleared textbook/online images (takedown clause).

**Dataset version:** `2026.10.1` (materialized at deploy; not committed).

---

## 13. realworldqa

Scoring: [PORTING_NOTES.md](../pipette_scores/scoring/vlm_upstream/PORTING_NOTES.md)

**765** real-world MCQs (xAI). Built from VLMEvalKit's pinned `RealWorldQA.tsv` by `scripts/build_realworldqa_dataset.py`. **CC-BY-ND-4.0**: the built parquet is Adapted Material and must not be redistributed in any form.

**Dataset version:** `2026.10.1` (materialized at deploy; not committed).

---

## 14. mmbench_en_dev

Scoring: [PORTING_NOTES.md](../pipette_scores/scoring/vlm_upstream/PORTING_NOTES.md)

MMBench EN dev v1.1 incl. **all circular-rotation rows** (rotations of a base question share `index % 1e6`; circular scoring needs every row). Built from VLMEvalKit's pinned `MMBench_DEV_EN_V11.tsv` by `scripts/build_mmbench_en_dev_dataset.py`; MCQ prompt with optional `Hint:` line; rotation rows' image pointers are resolved to actual bytes. Apache-2.0 repo over uncleared public/internet images.

**Dataset version:** `2026.10.1` (materialized at deploy; not committed).

---

## 15. pope

Scoring: [PORTING_NOTES.md](../pipette_scores/scoring/vlm_upstream/PORTING_NOTES.md)

Yes/no object-hallucination probes over COCO val2014 images (adversarial / popular / random subsets; overlapping questions carry comma-joined categories). Built from VLMEvalKit's pinned `POPE.tsv` by `scripts/build_pope_dataset.py`; bare-question prompt; headline metric is F1. MIT annotations over COCO/Flickr per-photo image terms (no redistribution).

**Dataset version:** `2026.10.1` (materialized at deploy; not committed).

## 16. ASR evals (Open ASR Leaderboard English, FLEURS)

16 evals, `asr_esb_*` (the Open ASR Leaderboard's 9 public English sets) and
`asr_fleurs_<locale>` (7 locales: `en_us`, `es_419`, `pt_br`, `de_de`, `fr_fr`,
`it_it`, `nl_nl`), version `2026.10.1`, built by
`scripts/build_asr_datasets.py` from pinned Hugging Face sources in their split's
row order. Each row is one clip: `sample_id`, `doc_id` (its position in the
source split), `reference`, `language` (ISO 639-3),
`session` / `chunk_index` (chunked Earnings22), the source `audio` bytes and
their `audio_format`, `audio_sha256`, `sample_rate`, `channels`, `num_frames`,
and `raw_doc` (the source row's other fields plus `language_name`,
`language_alpha_2`, `audio_subtype`). There is no stored prompt: a prompt
profile builds the request. `/samples` streams a set's requests whole for one
profile, each clip as a 16 kHz mono WAV (16-bit PCM where its samples are 16-bit
values, 32-bit float otherwise), under the digest `/samples/digest` reports. `metadata.json`'s `content_sha256` is the
audio-inclusive fingerprint of the stored files, re-checked at load. The catalog
pins `source_sha256`, the same digest over the rows as the source shipped them,
because the builder's resampling rounds per CPU architecture. The output is
built outside the checkout (the builder refuses it inside) and is gitignored and
dockerignored. Sources, pins, integrity, profiles and metrics:
[asr-evals.md](asr-evals.md).
