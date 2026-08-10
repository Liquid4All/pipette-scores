# Scoring Logic

Each eval has a dedicated scorer in `pipette_scores/scoring/`. The common flow is:

1. **Dispatch** — `scoring.score()` routes to the per-eval `score()` function based on `eval_id`.
2. **Build lookup** — the scorer indexes dataset samples by ID for fast matching.
3. **Evaluate** — each completion is compared to its ground-truth sample, producing a `SampleScore(id, is_correct)`.
4. **Enrich** — the API layer joins scores with the original prompts and completions and returns the full `ScoreResponse` (see `httpapi.md`).

A completion whose `id` does not match any dataset sample will raise a `KeyError`. Clients should only submit IDs returned by the samples endpoint.

The scoreable evals are exactly the members of `pipette_scores.types.EvalId`: `ifbench`, `ifstruct`, `gpqa_diamond`, and `math_500`.

## 1. Per-eval return shape

Each per-eval `score()` returns `list[SampleScore]`:

```python
[
    SampleScore(id=str, is_correct=bool),
    ...
]
```

Totals and per-sample enrichment (messages, completion) are assembled at the API layer — scorers stay focused on correctness judgement.

For evals served with `metadata.repeats = N`, the samples endpoint expands each base id `<id>` into `N` salted ids `<id>#0 … <id>#(N-1)`; the scorer treats each `#k` as an ordinary sample, and the headline **pass@1** is the mean correctness across the repeats per base sample.

---

## 2. ifbench

Dataset: [ifbench](datasets.md#5-ifbench) | Module: `pipette_scores/scoring/ifbench/` (scorer + bundled upstream adapter)

Precise instruction following on out-of-distribution constraints, scored with the upstream IFBench **loose** checker.

Thinking tags are stripped and degenerate (repetition-collapse) output is rejected before scoring. Each completion is built into an IFBench `InputExample` and run through `test_instruction_following_loose`; the sample is correct only if **every** attached constraint passes under at least one of the upstream's 8 response transformations. Served with `metadata.repeats = 5`, reported as pass@1.

---

## 3. ifstruct

Dataset: [ifstruct](datasets.md#6-ifstruct) | Module: `pipette_scores/scoring/ifstruct.py`

Structured format (JSON/YAML) generation, validated against the per-sample schema by the vendored `ifstruct` validator (`validate_response`).

For each completion the validator checks: output-format parse (JSON or YAML), the fenced-code-block and no-commentary requirements when set, top-level structure (bare list vs wrapped object) and wrapper-key name, all fields against the JSON schema, and the top-level item count. The sample is correct only when there are no validation errors. Reasoning tags are **not** stripped, so visible scratch-work fails the no-commentary check. Served once per prompt (no repeats).

---

## 4. gpqa_diamond

Dataset: [gpqa_diamond](datasets.md#7-gpqa_diamond) | Module: `pipette_scores/scoring/gpqa_diamond/` (scorer + `mcq` verifier)

Graduate-level science multiple choice (A–D), Artificial Analysis methodology.

Thinking tags are stripped, then the chosen letter is extracted with the AA MCQ extractor (`score_mcq(..., valid_options="ABCD")` — an `Answer: X` pattern with markdown-tolerant fallbacks, taking the last match) and compared to the correct option. This is **generative + regex extraction**; extraction is restricted to A-D, so an unparseable response counts as incorrect. Served with `metadata.repeats = 5`, reported as pass@1.

---

## 5. math_500

Dataset: [math_500](datasets.md#8-math_500) | Module: `pipette_scores/scoring/math_500/` (scorer + `math_generic`/`prm800k` verifier)

Competition mathematics; the model reasons step by step and gives its final answer in `\boxed{}`.

Graded with `score_math_generic` from `pipette_scores/scoring/math_500/math_generic.py`: thinking tags are stripped, the boxed answer is extracted, then graded by the **PRM800K** symbolic grader (sympy equivalence — so `1/2` matches `0.5`, or reordered expressions match) with a **robust-boxed** numeric-normalization fallback. The raw completion is passed through; the verifier does its own extraction. Served with `metadata.repeats = 5`, reported as pass@1.
