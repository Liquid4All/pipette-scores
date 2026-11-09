# Scoring Logic

Each eval has a dedicated scorer in `pipette_scores/scoring/`. The common flow is:

1. **Dispatch** — `scoring.score()` routes to the per-eval `score()` function based on `eval_id`.
2. **Build lookup** — the scorer indexes dataset samples by ID for fast matching.
3. **Evaluate** — each completion's reasoning markup is rewritten (see [§7](#7-reasoning-markup-and-cut-off-reasoning)), then it is compared to its ground-truth sample, producing a `SampleScore(id, is_correct)`.
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

Dataset: [ifbench](datasets.md#5-ifbench) | Module: `pipette_scores/scoring/ifbench/` (scorer + adapter for the vendored checkers)

Precise instruction following on out-of-distribution constraints, scored with the IFBench **loose** checker.

The checkers are allenai/IFBench @ `1a902f9`, the revision liquid_evals scores its `ifbench_aa` task with, vendored at `vendor/ifbench-checkers`. Four local changes are listed in that directory's README. They keep scoring offline and deterministic, and count a check that raises as not followed for that response; in loose mode the other response variants are still tried. None of them changes a verdict on an input where liquid_evals gives a stable one.

Reasoning markup is rewritten ([§7](#7-reasoning-markup-and-cut-off-reasoning)) and thinking tags are stripped before scoring. There is no repetition filter here; the client stops runaway repetition, and the scorer evaluates the text as submitted. Each completion is built into an IFBench `InputExample` and run through `test_instruction_following_loose`; the sample is correct only if **every** attached constraint passes under at least one of the upstream's 8 response transformations. Served with `metadata.repeats = 5`, reported as pass@1.

---

## 3. ifstruct

Dataset: [ifstruct](datasets.md#6-ifstruct) | Module: `pipette_scores/scoring/ifstruct.py`

Structured format (JSON/YAML) generation, validated against the per-sample schema by the vendored `ifstruct` validator (`validate_response`).

For each completion the validator checks: output-format parse (JSON or YAML), the fenced-code-block and no-commentary requirements when set, top-level structure (bare list vs wrapped object) and wrapper-key name, all fields against the JSON schema, and the top-level item count. The sample is correct only when there are no validation errors. The completion first goes through the markup rewrite in [§7](#7-reasoning-markup-and-cut-off-reasoning). Before these checks the validator then removes thinking blocks, once, as liquid_evals does: `<think>…</think>`, `[THINK]…[/THINK]`, orphan opening or closing tags, and a GPT-OSS `assistantfinal` preamble. Any other visible scratch-work makes an unfenced response fail to parse. With a fenced block, it fails only samples that set `require_no_commentary`. Served once per prompt (no repeats).

---

## 4. gpqa_diamond

Dataset: [gpqa_diamond](datasets.md#7-gpqa_diamond) | Module: `pipette_scores/scoring/gpqa_diamond/` (scorer + `mcq` verifier)

Graduate-level science multiple choice (A–D), Artificial Analysis methodology.

Reasoning markup is rewritten ([§7](#7-reasoning-markup-and-cut-off-reasoning)) and thinking tags are stripped, then the chosen letter is extracted with the AA MCQ extractor (`score_mcq(..., valid_options="ABCD")` — an `Answer: X` pattern with markdown-tolerant fallbacks, taking the last match) and compared to the correct option. This is **generative + regex extraction**; extraction is restricted to A-D, so an unparseable response counts as incorrect. Served with `metadata.repeats = 5`, reported as pass@1.

---

## 5. math_500

Dataset: [math_500](datasets.md#8-math_500) | Module: `pipette_scores/scoring/math_500/` (scorer + `math_generic`/`prm800k` verifier + vendored `liquid_evals_parser`)

Competition mathematics; the model reasons step by step and gives its final answer in `\boxed{}`.

The dataset version picks the grader through `metadata.grader`. Either way, reasoning markup is rewritten ([§7](#7-reasoning-markup-and-cut-off-reasoning)) and thinking tags are stripped first; the grader does its own answer extraction from what is left.

- **`2026.06.1`, `math_generic`** (no `grader` key): `score_math_generic` from `pipette_scores/scoring/math_500/math_generic.py`. The boxed answer is extracted, then graded by the **PRM800K** symbolic grader (sympy equivalence — so `1/2` matches `0.5`, or reordered expressions match) with a **robust-boxed** numeric-normalization fallback. Served with `metadata.repeats = 5`. The context counts each grading path as `math_500_method/{prm800k,robust_boxed,no_match,failed}`.
- **`2026.10.1`, `liquid_evals_math500`**: liquid_evals' `extract_math_answer` and `verify_math_answer`, vendored verbatim from `liquid_evals/parser.py` @ `d5534cf` as `pipette_scores/scoring/math_500/liquid_evals_parser.py`, called as `tasks/math500.py` calls them. `extract_math_answer` takes the last `\boxed{}` (brace-counted, so nested braces and boxes keep the outer content), else a `###`/`####` line, `ANSWER:`, "answer is …", "final answer" patterns, else the last number. `verify_math_answer` accepts an exact match, else tries `check_math_result` (Hendrycks' mathd normalization, or PRM800K-style sympy equivalence) on the extracted answer, then on the whole response with a leading "final answer:"/"answer:"/`###` removed, then on each of the response's last 150 lines. Sympy is skipped for expressions over 3000 characters, with more than two unknown letters, or with `^{`, `^(` or multi-digit exponents, and LaTeX with more than 100 macros is not parsed; there is no time limit. Served with `metadata.repeats = 5`. The context counts the pattern the answer came from as `math_500_extraction/{boxed,hash,answer_colon,answer_is,other,numeric,none}`; `math_500_unparsed_rate` is the `none` share.

---

## 6. Parity with liquid_evals

The internal eval scheduler runs these evals through [Liquid4All/liquid_evals](https://github.com/Liquid4All/liquid_evals). Another internal harness has keys with the same names; those are a different implementation and are not reproduced here. Each scorer here reproduces the matching liquid_evals task, so the same completion gets the same verdict on both sides, apart from the differences listed below:

| Eval | liquid_evals task | Same on both sides |
|------|-------------------|--------------------|
| `ifbench` | `ifbench_aa` | Prompts, loose prompt-level verdict, checker revision (allenai `1a902f9`; local patches in `vendor/ifbench-checkers/README.md`), nltk 3.9.4, thinking-tag removal |
| `ifstruct` | `ifstruct` | Test set (byte-identical), validator verdicts, thinking-tag removal |
| `gpqa_diamond` | `gpqa_diamond_aa` | Prompt and option shuffle, AA answer extraction |
| `math_500` `2026.06.1` | `math500_std` | `MathGenericFormatter` prompt, gold answers, `score_math_generic` without the LLM fallback |
| `math_500` `2026.10.1` | `math500` | Prompt, gold answers (`extract_math_answer(solution)`, identical to `answer`), `extract_math_answer` + `verify_math_answer` code, sympy 1.14.0 and pylatexenc 2.10 |

Differences that remain by design:

- **Generation is client-side.** The service serves prompts only. With no overrides, liquid_evals (in-process vLLM) samples at temperature 0.6, top_p 1.0 and top_k -1, with a fresh random seed per generate call. vLLM's repetition guard is on: a 2–128-token pattern repeated 10 times stops generation. Up to 32768 new tokens are allowed within a 32768-token context, capped to the model's base context. It applies the model's chat template with no system prompt. Pipette's clients set their own policy (temperature 0.6, no fixed seed) and stop runaway repetition with their own detector.
- **Repeats.** `math500_std` generates once per problem, while `math_500` `2026.06.1` is served with 5 repeats. The expected pass@1 is the same, and the variance is lower. `math500` also runs once per problem, but the model-card number is the mean of 5 `math500` runs, each with a fresh seed. `2026.10.1` serves those 5 passes as 5 repeats in one cell, and pass@1 over the 2500 attempts equals the mean of the 5 per-pass scores.
- **Crashes.** Where a liquid_evals checker raises and the whole run gets no score, the scorer here usually returns a verdict. For example, a JSON integer literal over 4300 digits fails the ifstruct sample, and a raising IFBench check counts as not followed for that response. Pathologically nested JSON (RecursionError) still fails on both sides; here `/score` returns HTTP 500 for the batch.
- **IFBench key 286.** liquid_evals reads that row's `n_start=0` as unset and picks a random span start on every call, so a correct answer passes about 1 time in 42. Here it always passes, which can raise ifbench pass@1 by up to 0.33 pp (5 of 1500 attempts).
- **Denominators and dataset pins.** ifstruct's pass rate here counts only the completions submitted. liquid_evals counts every sample, so a missing completion is a fail there. The ifbench, gpqa_diamond and math_500 pass@1 keys here already count every sample. liquid_evals reads its Hugging Face datasets without a revision pin, while the datasets here are pinned by content hash. They are identical today.
- **Reasoning markup.** liquid_evals strips only `<think>`, `[THINK]` and gpt-oss channels. The rewrite in [§7](#7-reasoning-markup-and-cut-off-reasoning) also reads Gemma 4, Cohere North and ERNIE markup. A completion without that markup gets the same verdict on both sides, whatever its stop reason, including one cut off inside its reasoning.
- **Scale.** Scores here are 0–1 fractions; the internal eval scheduler shows them ×100.

---

## 7. Reasoning markup and cut-off reasoning

Module: `pipette_scores/scoring/completion_text.py` (`grading_text`)

Every scorer grades `grading_text`'s output instead of the raw completion. The thinking-tag rules then run as before: `remove_thinking_tags` for ifbench, gpqa_diamond and math_500, and the validator's own copy for ifstruct, which therefore strips once.

**Markup rewrite** (`normalize_reasoning_markup`). Some models mark their reasoning or their answer with tokens the thinking-tag rules do not know. These are rewritten first:

| Model | Markup in the completion | Rewrite |
|-------|--------------------------|---------|
| Gemma 4 (llama.cpp) | `<\|channel>thought…<channel\|>`, then the answer | Each channel becomes `<think>…</think>`. An unclosed channel becomes an open `<think>`, which the orphan-open rule removes to the end. A `<channel\|>` with no opener is dropped. This reads the markup as the Gemma 4 chat template's `strip_thinking` does |
| Cohere North (vLLM, or llama.cpp with the markers rendered) | `<\|START_THINKING\|>`, `<\|END_THINKING\|>` | `<think>`, `</think>` |
| Cohere North | `<\|START_TEXT\|>…<\|END_TEXT\|>` around the answer | Dropped |
| ERNIE-4.5-Thinking (also behind vLLM's `ernie45` parser) | `<response>…</response>` around the answer | Dropped |

An answer wrapper's opening tag is dropped only where a section starts: at the top of the completion, or right where its reasoning closes, after the first `</think>` (or the first `[/THINK]` when there is no `</think>`). Its closing tag is dropped only at the very end. A wrapper nested in another is dropped too. The same tags anywhere else are answer text and stay. A completion without this markup is unchanged.

Gemma 4 on vLLM without `--reasoning-parser gemma4` loses both channel tokens, which are special tokens, so its completions arrive as `thought\n<reasoning><answer>` with no delimiter left to cut on.

**Cut-off reasoning.** A completion is graded on its rewritten text whatever its `stop_reason`, as liquid_evals grades every response through `remove_thinking_tags`, with no stop-reason rule. For a completion cut off before its reasoning closed, that means:

- When the chat template opens the reasoning block in the prompt, the completion carries no tag at all and is graded whole: an answer the reasoning already states counts.
- A completion that opened `<think>` or `[THINK]` itself loses everything from the tag on, by the orphan-open rule; what is graded is the text before the tag, usually nothing. After the rewrite this also covers an unclosed Gemma 4 thought channel and North's `<|START_THINKING|>`.
- gpt-oss cut inside its analysis channel has no `assistantfinal` transition and is graded whole.
- A completion whose reasoning closed is graded on the answer after the close, even if the answer was cut.

**Strict view.** Scorer `2026.10.12-552094f` graded such completions as empty instead. Each eval still reports that view in the response context:

- `<eval_id>_blanked_truncated` counts the completions it blanks. A completion is counted when the request carries `enable_thinking: true`, its `stop_reason` is `truncated` (the output-token cap or a full context) or `doom_loop` (a repetition stop: vLLM's `repetition_detection`, or an older client's detector), its rewritten text has no reasoning close (no `</think>`, `[/THINK]` or gpt-oss `assistantfinal` transition), and that text is not empty. A missing `stop_reason`, or `eos`, `failure` or `unknown`, is never counted.
- `<eval_id>_pass_at_1_blanked` is the eval's pass@1 with those completions counted wrong: the headline that scorer reported. For ifstruct, which is served once per prompt, it is `pass_rate` with those completions counted wrong. Every eval fails an empty completion, so counting them wrong is the same as grading them empty.

Content that a server-side reasoning parser already split (vLLM with `--reasoning-parser`, or gpt-oss's Harmony final channel) is the answer alone with no close. With thinking on, a cut answer there is counted even if its reasoning finished, so the strict view understates those cells.

### VLM evals

Module: `split_reasoning`, `normalize_vlm_reasoning_markup`. The VLM scorers (mme, mmstar, ocrbench, mmmu_val, realworldqa, mmbench_en_dev, pope) read each completion with `split_reasoning`. The text evals (ifbench, gpqa_diamond, math_500, ifstruct) keep `grading_text` and the rules above.

**Markup rewrite** (`normalize_vlm_reasoning_markup`): the table above, plus:

| Model | Markup in the completion | Rewrite |
|-------|--------------------------|---------|
| Ministral-3-Reasoning | `<thinking>…</thinking>` after its `[/THINK]` | `<think>`, `</think>` |
| Kimi-VL-A3B-Thinking-2506 | `◁think▷…◁/think▷`, plain text in its vocabulary, so the server prints it | `<think>`, `</think>`. A thought that never closes is removed to the end, as Kimi's own `extract_summary` returns "" for it, unless the model stopped with a complete `<answer>…</answer>` in it (see `unclosed_stop` below) |

**Reasoning, however it was opened.** A VLM scorer reads a completion once, and gets the text to grade, a status and the strict-view flag. Reasoning counts as **opened** when any of these holds: its opener is in the text (`<think>`, `[THINK]`, and after the rewrite `◁think▷`, `<thinking>`, North, Gemma channels); the run's prompt profile opens it for the model (`REASONING_OPENERS` in `prompt_profiles.py`: `phi4_rv_think` by an assistant prefill, `nemotron_think` by its `/think` system message and template, `kimi_vl` and `mimo_vl` because the model always thinks; not `internvl_r1`, whose R1 prompt has the model write `<think>` itself); `enable_thinking` is true; or a close (`</think>`, `[/THINK]`, the gpt-oss `assistantfinal` transition) shows it was opened out of sight, as with a hidden control token such as Ministral's `[THINK]`, or a template's `<think>\n`. The run does not have to have recorded `enable_thinking`: a client refuses it for a template with no switch.

| Status | Meaning |
|--------|---------|
| `closed` | a close was read, and the opener was known, or is in the text with no `</think>` or `[/THINK]` before it |
| `orphan_close` | a close was read, the opener was not known, and no opener is in the text or a `</think>` or `[/THINK]` comes before the first one |
| `unclosed_cut` | opened, no close, generation cut (`truncated` or `doom_loop`) |
| `unclosed_stop` | opened, no close, the model stopped on its own |
| `no_reasoning` | no sign of reasoning |

The context counts them as `reasoning_status`, a dict with all five keys. A profile whose wrapper reduces the reply to its answer (`minicpm_v_cot`, [httpapi](httpapi.md)) is read after that reduction, as main's strict view reads it: a `minicpm_v_cot` reply cut inside its thought has become its answer, so it counts as `unclosed_cut` and blanked only through `enable_thinking`, which its runs record. With a server-side reasoning parser the content is the answer alone, so a run that declared `enable_thinking` reads `unclosed_stop`; the count flags it and changes no grade.

**Text graded.** The grading is liquid_evals', as above: a completion with no tag is graded whole, whatever its status. Complete blocks are removed (a block ends at the first close after its opener, and an opener inside it is part of it); the text through the first close that has no opener is reasoning (the first such `</think>`, else the first such `[/THINK]`, as `remove_thinking_tags` cuts), and a later stray close stays in the text; an opener left at the end removes the rest; then the gpt-oss transition and trailing harmony markers go. The two tag families pair with each other (`[THINK]…</think>`). Beyond that:

- **Answer before close.** A reply that opens with a one-word answer, closes a block it never opened, and goes on to reasoning that gives no answer (`yes[/THINK][THINK]I need to determine…`, Ministral-3-Reasoning on yes/no questions) is graded on that word, when no opener was known. More than one word before the close is reasoning, as Ministral writes in `So the answer should be "yes".[/THINK]`, and such a reply is graded empty. Counted as `reasoning_answer_before_close`. The rules of liquid_evals grade such a reply empty.
- **`unclosed_stop`.** When the opener is known from the profile or in the text and the model stopped without closing, a complete `<answer>…</answer>` in the reply is graded (the last one); without it the reply is graded as above.
- **`<answer>` wrapper.** An `<answer>` that opens the text after a close and its closing tag are dropped, like the `<response>` and North wrappers of §7's table.

**Strict view.** `blanked_truncated` counts a completion whose reasoning was opened by its own opener, by the run's prompt profile, or by `enable_thinking: true`, whose `stop_reason` is `truncated` or `doom_loop`, whose rewritten text has no close, and whose text is not empty, whatever the row recorded: a reply that opened its own thought and was cut counts even under `enable_thinking: false`. A reply cut inside reasoning that nothing shows was opened (a hidden control token, a profile or `enable_thinking` that does not say so) cannot be told from an answer, and is graded and counted as one.

## 8. ASR evals (Open ASR Leaderboard English, FLEURS)

`scoring/asr/`: the Open ASR Leaderboard's English WER (its normalizer, row
filter, long-form merge and kaldialign alignment) and FLEURS scored with its
multilingual scoring (its multilingual normalizer, which spells numbers out in
the language so that digits and number words compare equal, its compound-pair
step and kaldialign) applied to FLEURS's own `transcription` references, with
the leaderboard's code vendored unchanged (`scoring/asr/NOTICE.md`). The scorer cleans each completion with the request's
prompt profile, normalizes and aligns it with the dataset's recipe, and reports
the corpus rate (summed edits over summed reference units). A result is an
answer to every sample, or no result: a run with a missing or failed sample is
refused (`/score` answers 400), never scored with the gap as an empty
transcript. The
strict view of §7 does not apply: ASR runs with thinking off. Parity with the
public reference implementations (the leaderboard's own code, with num2words
0.5.14, and `datasets` audio decoding) is tested in
`tests/test_asr_public_reference.py`; the suite headlines' definition is
`scoring/asr/suite.py`. Details: [asr-evals.md](asr-evals.md).
