# VLM scorer porting notes (VLMEvalKit @ 8eca8e5)

Answer extraction and metrics for the 7 VLM evals are vendored from
**VLMEvalKit** (commit `8eca8e5`), deterministic-only: no LLM judge, no random
fallback. On the multiple-choice and yes/no evals, the vendored rule
(`can_infer` or `YOrN_Extraction`) reads each answer first. The deterministic
fallbacks in `scoring/vlm_answer.py` then read an answer that the rule cannot.
An answer that neither reads counts as **wrong** (the deterministic equivalent
of VLMEvalKit's `exact_matching` policy). The "Scoring" lines below name the
vendored rule only.

Module provenance:

| module | ported from |
|---|---|
| `matching_util.py` | `vlmeval/utils/matching_util.py` (`can_infer`, `can_infer_option`, `can_infer_text`) |
| `yorn.py` | `vlmeval/dataset/utils/yorn.py` (`YOrN_Extraction`, `MME_rating`, `POPE_rating`) + `vlmeval/smp/misc.py` (`process_punctuation`) |
| `multiple_choice.py` | `vlmeval/dataset/utils/multiple_choice.py` (`build_choices`, `MMMU_preproc`, vanilla `eval_vanilla`/`report_acc` semantics) |
| `ocrbench.py` | `vlmeval/dataset/utils/ocrbench.py` (`OCRBench_eval`) |

## Metric contract (pipette-mgmt)

Every scorer's `context` includes:

- `"accuracy"` — 0-1 float, the primary comparable metric read by
  pipette-mgmt. MCQ evals → hit rate; **POPE → F1** (0-1); **MME → total
  score / 2800**; **OCRBench → Final Score / 1000**.
- `"leaderboard_score"` — the conventional human-facing scale: MCQ acc x100,
  POPE F1 x100, MME total (0-2800), OCRBench Final Score (0-1000).

## Per-benchmark raw_doc + prompt requirements

The scorer reads the parquet `prompt` column (what the model saw) and the
JSON `raw_doc` column. Dataset regeneration must produce VLMEvalKit-style
prompts (`ImageMCQDataset.build_prompt`) and these `raw_doc` fields:

### MCQ prompt format (mmbench_en_dev, mmstar, mmmu_val, realworldqa)

```
[Hint: {hint}\n]            # only if the row has a hint (MMBench)
Question: {question}\n
Options:\nA. {A}\nB. {B}\n...   # only option letters present in the row
Please select the correct answer from the options above. \n
```

(For MMMU, `<image N>` placeholders in the question are interleaved with the
images — `MMMUDataset.split_MMMU`; text content is unchanged.)

### mmbench_en_dev
- raw_doc: `question`, option columns `A`-`D` (some rows have fewer), `answer`
  (letter), `index` (int; circular rotations of one base question share
  `index % 1e6` — the dataset MUST contain the rotation rows from
  `MMBench_DEV_EN_V11.tsv` for circular scoring to be meaningful), `category`,
  optional `hint`, `l2-category`.
- Scoring: `can_infer` per rotation; base-question hit = all rotations
  correct; headline = base-question hit rate.

### mmstar
- raw_doc: `question`, `A`-`D`, `answer` (letter), `category`, `l2_category`
  (or `l2-category`), `index`.
- Scoring: `can_infer`; headline = micro mean(hit) (VLMEvalKit `report_acc`
  Overall — not the macro per-L2-category mean).

### mmmu_val
- raw_doc: `question`, option columns `A`-`I` (absent/None for open
  questions), `answer` (letter for MCQ, free text for open), `category`
  (subject), `index`.
- Scoring: `MMMU_preproc` — open questions (no `A` option) become 2-choice
  MCQ with `A = gold answer`, `B = "Other Answers"`, GT = `A`; then
  `can_infer`. Headline = micro accuracy.

### realworldqa
- raw_doc: `question`, `A`-`D`, `answer` (letter), `index`.
- Scoring: `can_infer` (VLMEvalKit ImageMCQ path). Headline = micro accuracy.

### mme
- prompt: the raw MME question (ends with "Please answer yes or no.").
- raw_doc: `question`, `answer` ("Yes"/"No"), `category` (one of the 14 MME
  categories, exact strings incl. `OCR` capitalization), and a **pairing
  key** identifying the image: `image_path` (VLMEvalKit) or `question_id`
  (lmms-eval export, where it is the image's relative path). The two yes/no
  questions of one image MUST share the pairing key, else `acc_plus`
  collapses and the score is deflated.
- Scoring: `YOrN_Extraction`; per category `acc + acc_plus`; perception
  (10 cats, max 2000) + cognition (4 cats, max 800) = total (max 2800).

### pope
- prompt: the raw POPE question ("Is there a ... in the image?").
- raw_doc: `question`, `answer` ("Yes"/"No"), `category` (subset name:
  adversarial / popular / random; may be comma-joined — it is exploded).
- Scoring: `YOrN_Extraction`; headline = **F1** over yes/no predictions
  ('Unknown' counts as a "No" prediction); precision / recall / accuracy in
  context.

### ocrbench
- prompt: the raw OCRBench question.
- raw_doc: `question`, `answer` (list of acceptable strings, or a stringified
  list, or a single string), `category` (VLMEvalKit name; the lmms-eval field
  `question_type` holds the same values and is accepted) — one of the 10
  OCRBench categories, **exact case**.
- Scoring: substring match of any gold answer in the prediction; lowercase
  for all categories EXCEPT `Handwritten Mathematical Expression Recognition`,
  which is case-sensitive with all whitespace removed. Headline = Final Score
  (sum correct, /1000).

## Coverage and denominator policy

The `/score` API scores a VLM eval only when the request answers every
dataset sample once, none with `stop_reason: failure`; anything else is a
400 (pipette-mgmt refuses such a run before it calls, and pipette's runner
retries a failed sample and never submits a run without an answer to each).
The scorers below hold the full-dataset line regardless, for any direct
caller of the scoring functions. Every headline metric (`context["accuracy"]` / `context["leaderboard_score"]`)
is computed over the **full dataset**, never over the submitted subset:

- mmstar / realworldqa / mmmu_val: micro accuracy with `len(samples)` as the
  denominator — an unsubmitted sample counts as wrong.
- mmbench_en_dev: circular base groups are enumerated from the dataset; a
  group with any unsubmitted rotation is a miss.
- pope: the confusion matrix covers all dataset samples; a missing completion
  counts as the wrong answer (a false negative or a false positive), not as
  `Unknown`: that reads as "No", which would make a missing negative a free
  true negative that F1 never sees.
- mme: a missing completion contributes 0 to both `acc` and `acc_plus`
  (denominators are per-category dataset counts; total is /2800).
- ocrbench: Final Score /1000 regardless of coverage.

Each scorer also reports `context["samples_missing"]` — the number of dataset
samples with no submitted completion (0 for a complete run). Diagnostics named
`per_sample_accuracy` / `per_rotation_accuracy` remain submitted-only and are
labeled as such in code.

Rationale: a subset must never be able to inflate a leaderboard number, at
any layer. This matches the text `math_500` / `gpqa_diamond` convention of
dividing by `len(samples)`.

## Scoring parity (verified against VLMEvalKit's own code, 2026-10-19)

The port was checked answer by answer against VLMEvalKit `8eca8e5` itself,
unmodified, on 49,626 stored answers: LFM2.5-VL-3B (no thinking), Qwen3.5-4B
and Gemma-4-E2B (thinking), each on all seven benchmarks. VLMEvalKit built its
datasets from the seven pinned TSVs (MD5 checked) and scored with its exact
matching (`ImageMCQDataset.evaluate`, MMBench circular, `MMMUDataset`,
`ImageYORNDataset` for MME and POPE, `OCRBench.evaluate`), no LLM judge. Both
scorers graded the same text: the completion with its reasoning removed.

- With the fallbacks below switched off, pipette gives VLMEvalKit's verdict on
  every answer and its headline on all 21 runs. The extracted letter or yes/no
  agrees on all 44,256 multiple-choice and yes/no rows, and the verdict on all
  3,000 OCRBench rows. No porting difference was found.
- Every disagreement with the fallbacks on is a fallback reading an answer
  exact matching could not (1,557 of 49,626): the last answer statement,
  a line that opens with an option, the last line. VLMEvalKit hands those to
  its LLM judge, which pipette never uses. LFM2.5-VL-3B, which answers with a
  letter, has none; a reasoning model that explains its answer has many.
- A reply that names both "yes" and "no", such as "I cannot answer yes or no",
  is not read as an answer.
- The fallbacks also read the format that Qwen3.5's card prompt asks for,
  `"answer": "C"`, in three more shapes: the key
  in backticks (`` `answer`: "C" ``), the value alone (`"B"`, `"a"`, an
  unclosed `"B`), and a quoted option line (`"B. Not the same"`). Each is read
  only in an anchored shape: the field as a whole line; the value alone as
  the whole answer, or as a last line (capital letter or exact option) after
  text that names no option letter. Prose that describes the format, refuses
  to choose, or quotes two options is not read. With these shapes read,
  Qwen3.5-4B scores 79.3 on RealWorldQA (42.0 without them; its card: 79.5),
  and Qwen3.5-2B scores 77.7 on MMBench (58.0 without them). No other family's
  score changes.

## Prompt parity (verified against VLMEvalKit source)

Without a prompt profile, the served prompts match VLMEvalKit's dataset-default
`build_prompt` verbatim, with NO extra instruction beyond what upstream emits:

- **MMStar / MMBench_DEV_EN_V11 / RealWorldQA** use the base `ImageMCQDataset.build_prompt`:
  `Question:` + `Options:\nA. …` + the soft directive
  `"Please select the correct answer from the options above. \n"`. There is NO
  stricter `"Answer with the option's letter … directly"` suffix — that belongs
  to *other* datasets (MMMU-Pro, GMAI, LEGO, VisualPuzzles), not these. Adding
  it would diverge from upstream and shift scores.
- **MMMU_DEV_VAL** = the same base prompt + `split_MMMU` image interleaving.
- **POPE / MME** use the raw question (bare prompt). VLMEvalKit only appends
  `"\nPlease answer yes or no."` for AMBER (not run here); MME's own question
  text already ends with that sentence.
- **OCRBench** uses the raw question.

Verbose models that describe the image instead of emitting the option letter
are handled by `can_infer` and then the `vlm_answer` fallbacks, on the
completion with its reasoning removed — not by a stricter prompt.

## Generation length

The scorers grade the text that the client sends. Allow enough output tokens
for the model to state its answer. The reference setting is **1024** output
tokens for all 7 evals. In VLMEvalKit, generation length is a per-*model*
serving default, not a per-dataset constant, so there is no single upstream
value to copy. The limit is a ceiling, not a target. Models that answer first
stop after a few tokens. Reasoning models that state their answer last need
the full budget, and a lower limit cuts their answer off.
