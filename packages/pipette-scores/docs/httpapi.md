# API Reference

Base URL: `http://localhost:8000` (default uvicorn)

## 1. Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PIPETTE_SCORES_DATA_DIR` | Root directory for dataset files (see [Data Storage](datasets.md)) | Parent of the `pipette_scores` package |
| `PIPETTE_SCORES_VERSION` | Value reported as `runtime_version` in the `POST /score` response | `dev` |
| `PIPETTE_SCORES_PACKAGES` | The eval packages' store, `s3://<bucket>/<prefix>` or a directory: a set's digest is read from its index for `PIPETTE_SCORES_VERSION` rather than computed ([packages.md](packages.md#3-index)) | unset: every digest is computed |

## 2. Caching

The server caches loaded datasets in memory for the lifetime of the process, and the prompt samples of the four `(eval_id, dataset_name, prompt_profile)` combinations it served last: a VLM set holds its base64 images (POPE about 1.1 GB) and every profile is a set of its own, so an older one is rebuilt from its parquet when asked for again. The samples' digest ([§3.3](#33-get-evalseval_iddatasetsdataset_namesamplesdigest)) is cached too, but alone: the digest route builds the samples, hashes them and drops them, so a caller probing many combinations keeps a few bytes per combination resident rather than its samples. Cold builds run one at a time per process, on a thread of their own, so a burst of them after a deploy neither multiplies memory nor holds up `POST /score`.

---

## 3. Endpoints

### 3.1. `GET /health`

Health check.

**Response** `200 OK`

```json
{"status": "ok"}
```

---

### 3.2. `GET /evals/{eval_id}/datasets/{dataset_name}/samples`

Get prompt samples for an eval/dataset. Returns the prompts a client should use for inference — no ground truth is included.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `eval_id` | string | Eval identifier (e.g. `math_500`) |
| `dataset_name` | string | Dataset to load |

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt_profile` | string, optional | VLM evals: replaces each prompt's text with the prompt one model family is evaluated with (VLMEvalKit's, or its model card's): `internvl`, `internvl_r1`, `granite_vision`, `minicpm_v`, `minicpm_v_cot`, `lfm_vl`, `lfm2_5_vl_3b`, `qwen3_5`, `glm_v`, `kimi_vl`, `nemotron_think`, `phi4_rv_think`, `mimo_vl` ([datasets.md §4](datasets.md#4-prompt-vs-scoring-samples)); omitted, every model gets the dataset's own prompt. **ASR evals: required**, the family whose request each clip becomes: `gemma4`, `qwen25_omni`, `voxtral`, `minicpm_o`, `lfm25_audio`, `lfm3` ([asr-evals.md §2](asr-evals.md#2-requests-the-prompt-profiles)). |

**ASR evals: one stream.** An ASR set's samples carry their audio (SPGISpeech's
are gigabytes), so the server never builds them whole: it first learns the
set's digest (from the digest route's cache, or one pass over the set,
[§3.3](#33-get-evalseval_iddatasetsdataset_namesamplesdigest)), sends it as
`X-Samples-Digest`, then writes the body one sample at a time, each sample as
the canonical JSON its digest hashes (so a client can hash the body's samples
as they arrive). A body that would not hash to its header is cut off before its
closing `]}`: a client then holds JSON it cannot parse, never a complete body
under the wrong digest. A sample's `id` is its position: the sample at index
`i` is `sha256("<eval_id>|<i>")[:12]`, which the server checks before serving.
Every served clip is re-hashed against the dataset's fingerprint and served as
a 16 kHz mono WAV, resampled once from the source's rate (e.g. MonsoonASR's
44.1 kHz) when the dataset was built, as the Open ASR Leaderboard's
`datasets.Audio(sampling_rate=16000)` does, so no clip is resampled when served: **16-bit PCM** (chunks `fmt `,
`data`) when every sample is a 16-bit value, which decodes to exactly the
samples a float WAV would carry, and **32-bit float** (chunks `fmt `, `fact`,
`data`) otherwise (a resampled or downmixed clip, a float source with finer
values); a clip is the same bytes on every call. ASR samples are not paged: an
`offset` or `limit` on an ASR request is a 400. Details: `asr-evals.md` §6.

**Response** `200 OK` — `SamplesResponse`, with an `X-Samples-Digest` header carrying the body's digest ([§3.3](#33-get-evalseval_iddatasetsdataset_namesamplesdigest)), so a client that keeps a copy knows which content it took without a second request.

```json
{
  "samples": [
    {
      "id": "a1b2c3d4e5f6",
      "messages": [
        {"role": "user", "content": "Solve the following math problem..."}
      ]
    }
  ]
}
```

**`samples[*]` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Deterministic sample ID (SHA256-based, 12 chars) |
| `messages` | list[ChatMessage] | Chat messages to send to the model |
| `chat_template_kwargs` | object, optional | Arguments for the server's chat template, sent with the request (ASR `voxtral`: `{"transcription": true, "language": "en"}`). Omitted when unused |

An ASR sample (here `voxtral` on `asr_fleurs_nl_nl`):

```json
{
  "samples": [
    {
      "id": "3f1c0d9a2b7e",
      "messages": [
        {"role": "user", "content": [
          {"type": "input_audio", "input_audio": {"data": "<base64 WAV>", "format": "wav"}}
        ]}
      ],
      "chat_template_kwargs": {"transcription": true, "language": "nl"}
    }
  ]
}
```

**Error Responses:**

| Status | Condition |
|--------|-----------|
| 400 | Unknown `prompt_profile`, or one given for a text eval; an ASR request without an ASR profile, or with `offset` or `limit` |
| 404 | `eval_id` not found or dataset directory does not exist |
| 500 | Error loading samples from parquet; an ASR dataset that fails its integrity checks (fingerprint, a sample out of position, a smoke build). A clip whose bytes no longer match its recorded hash is found while streaming, so it cuts the body off instead |

**Examples:**

```bash
curl "http://localhost:8000/evals/math_500/datasets/2026.06.1/samples"
curl "http://localhost:8000/evals/mmstar/datasets/2026.10.1/samples?prompt_profile=internvl"
curl -o voxpopuli.json "http://localhost:8000/evals/asr_esb_voxpopuli_cleaned_aa/datasets/2026.10.1/samples?prompt_profile=voxtral"
```

---

### 3.3. `GET /evals/{eval_id}/datasets/{dataset_name}/samples/digest`

The digest of what [`/samples`](#32-get-evalseval_iddatasetsdataset_namesamples) serves for the same path and `prompt_profile`, without the samples: a few bytes for a caller that only needs to know whether its copy is still current. It takes the same parameters and returns the same errors as `/samples`.

**Response** `200 OK` — `SamplesDigestResponse`

```json
{"digest": "sha256:1bcdef92af9cb06377f51f075c4fb1e4da9d1b32004c72922b2933cac15a2ea3", "sample_count": 5127}
```

| Field | Type | Description |
|-------|------|-------------|
| `digest` | string | `sha256:` and the hex SHA-256 of the served samples, each as its JSON with keys sorted, no whitespace and non-ASCII as UTF-8, followed by `\n`, in served order. Any change to what is served (an id, a prompt's wording, an image, the order) changes it; nothing else does, so it is stable across restarts and pinnable. A client can recompute it from a `/samples` body. |
| `sample_count` | integer | Number of samples served |

The first request for a combination builds and hashes its samples once (3 s for POPE's 5127 samples, about 1 GB of JSON), then keeps only the digest and count; later requests are served from that in microseconds. For an ASR set the first request streams the set through once, decoding every clip, and learns the digest of every ASR profile in that one pass (each clip is encoded once per distinct cut), so the other five profiles' requests are served from the cache. With `PIPETTE_SCORES_PACKAGES` set, a set this server holds and its packages index lists is not built at all: the first request reads every profile's digest from the index ([packages.md](packages.md#3-index)).

```bash
curl "http://localhost:8000/evals/pope/datasets/2026.10.1/samples/digest"
```

---

### 3.4. `POST /score`

Score completions against ground truth. Returns per-sample scores enriched with the original prompt messages and completions so the caller can persist full audit records without a second round trip.

**Request Body** — `ScoreRequest`

```json
{
  "eval_id": "math_500",
  "dataset_name": "2026.06.1",
  "enable_thinking": true,
  "completions": [
    {"id": "a1b2c3d4e5f6", "completion": "The answer is \\boxed{42}", "stop_reason": "eos"}
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `eval_id` | string | Eval identifier |
| `dataset_name` | string | Dataset to score against |
| `enable_thinking` | bool, optional | The job's `enable_thinking` model flag. Omit it when the job does not set the flag. ASR evals run with thinking off: `true` is a 400 |
| `prompt_profile` | string, optional | The run's prompt profile. VLM evals: `scored_samples[*].messages` carry the prompts that profile produced, so stored results show what the model was sent. Grading is the same, except where the profile's VLMEvalKit wrapper cuts each reply down to its answer before the benchmark's scoring: `minicpm_v_cot` grades what MiniCPM-V-4.6's `extract_answer` (minicpm_v.py:1504-1529 at 8eca8e5) returns, the last stated `Answer:` even inside an unclosed reasoning block, on every benchmark but OCRBench; `kimi_vl` grades, on MMMU and MMStar, the last `\boxed{}` after Kimi-VL's thought ([scoring/profile_answers.py](../pipette_scores/scoring/profile_answers.py)). `scored_samples[*].completion` is the reply as sent either way. ASR evals: **required**; each completion is cleaned as that profile's source says before it is scored, and the messages are shown without their audio (each audio part emptied in place). An unknown profile, or one on a text eval, is a 400 |
| `samples_digest` | string, optional | The digest of the samples the run was given (`sha256:<hex>`, as [§3.3](#33-get-evalseval_iddatasetsdataset_namesamplesdigest) reports it). When given, scoring answers `409` if this server no longer serves those samples for the eval, dataset and profile, so answers are never graded and stored against prompts the model was not sent. Any other shape is a `422` |
| `completions` | list[SampleCompletion] | Completions to score |
| `completions[*].id` | string | Must match a sample ID from the samples endpoint |
| `completions[*].completion` | string | Raw model output text |
| `completions[*].stop_reason` | string, optional | How the client's generation ended, as plan-types names it: `eos`, `truncated`, `doom_loop`, `failure` or `unknown`. Omit it when the client reported none |

Unknown keys are ignored, so a caller may send more per-sample fields than these.

Before grading, every eval rewrites reasoning markup that the thinking-tag rules do not know: Gemma 4 thought channels, Cohere North's thinking and text markers, and ERNIE's `<response>` wrapper, and on the VLM evals Kimi-VL's `◁think▷` and `<thinking>` (see [scoring](scoring.md#7-reasoning-markup-and-cut-off-reasoning)). Each completion is then graded on its text, as liquid_evals grades it. On the text evals neither `enable_thinking` nor `stop_reason` changes a verdict; together they select the completions that the strict view in `context` counts wrong. On the VLM evals they, and a `prompt_profile` that opens the reasoning, also decide how a few reasoning shapes are read (scoring.md §7, VLM evals).

**Response** `200 OK` — `ScoreResponse`

`context` is a flat `string → JSON value` map of eval-specific aggregates (numbers typical, other scalars allowed). Every eval but the ASR evals (whose `context` is below) reports two keys for the strict view:

- `<eval_id>_blanked_truncated`: the number of completions it counts wrong. These are the completions with `enable_thinking: true`, a `stop_reason` of `truncated` or `doom_loop`, no reasoning close and some text.
- `<eval_id>_pass_at_1_blanked`: the eval's pass@1 with those completions counted wrong.

When either request field is missing, the count is 0 and the strict pass@1 equals the graded one.

```json
{
  "runtime_version": "2026.03.15-abcdef0",
  "context": {"accuracy_EN": 0.95, "accuracy_FR_FR": 0.82},
  "scored_samples": [
    {
      "id": "a1b2c3d4e5f6",
      "messages": [{"role": "user", "content": "Solve..."}],
      "completion": "The answer is \\boxed{42}",
      "is_correct": true
    }
  ]
}
```

Aggregates (`total`, `correct`, accuracy) are derivable from `scored_samples` and are not returned — callers compute them as needed.

| Field | Type | Description |
|-------|------|-------------|
| `runtime_version` | string | Value of `PIPETTE_SCORES_VERSION` env var (default `"dev"`) |
| `scored_samples` | list[ScoredSample] | Per-sample results |
| `scored_samples[*].id` | string | Matches the request `completions[*].id` |
| `scored_samples[*].messages` | list[ChatMessage] | Prompt from the dataset, its text only: `image_url` parts are left out, as the caller holds the images (sending them back would copy a VLM corpus, about 1 GB for POPE, with every score). ASR: each `input_audio` part stays in its place with its `format` and an empty `data`, so the turn order is auditable |
| `scored_samples[*].completion` | string | Echoed from the request |
| `scored_samples[*].is_correct` | bool | Whether the completion was scored correct (ASR: its unit scored with no error; not an ASR metric) |
| `scored_samples[*].measures` | object, optional | ASR only: the sample's part in its scoring unit (below). Omitted elsewhere |
| `scored_samples[*].chat_template_kwargs` | object, optional | The template arguments the sample was served with (ASR `voxtral`). Omitted when none |

**Error Responses:**

| Status | Condition |
|--------|-----------|
| 400 | A completion id twice or not in the dataset; or, for a VLM or ASR eval, a sample with no completion or a completion with `stop_reason: failure`: a VLM or ASR eval is scored only with an answer to every sample (an ASR sample the row filter leaves out of the score included); a missing or unknown ASR `prompt_profile`; `enable_thinking: true` on an ASR eval |
| 404 | `eval_id` not found or dataset directory does not exist |
| 409 | `samples_digest` given, and this server serves other samples for the eval, dataset and profile |
| 422 | Malformed body (e.g. missing `eval_id` / `dataset_name`) |
| 500 | Error loading dataset (an ASR dataset that fails its integrity checks included), scoring failure, or scorer returned an id not present in request completions |

**Example:**

```bash
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "eval_id": "math_500",
    "dataset_name": "2026.06.1",
    "completions": [
      {"id": "a1b2c3d4e5f6", "completion": "Step 1: ... \\boxed{42}"}
    ]
  }'
```

**ASR evals** (`asr_*`): a request is scored only with an answer to every
dataset sample (excluded ones included) and no `stop_reason: "failure"`;
anything else is the 400 above, never a result with the gap scored as an empty
transcript. `context` is the dataset's result and its counts:
`error_unit` (`"word"`: every ASR eval is scored by word error rate), `error_rate`, `result`, `hits`, `substitutions`,
`deletions`, `insertions`, `reference_units`, `units_scored`,
`samples_total`, `samples_excluded`, `stop_reasons`, `prompt_profile` and
`scoring`. `error_rate` and `result` are null only for a dataset with no
reference unit. Each `scored_samples[*]` adds `measures`, sum-safe (a unit's
counts and normalized texts on one sample; an Earnings22 call's on its first
chunk, the other chunks `{"unit": <call>}`; `{"excluded": true}` for a sample
the row filter leaves out). Field by field:
[asr-evals.md §3](asr-evals.md#3-metrics).

An abridged response on `asr_esb_voxpopuli_cleaned_aa` (illustrative values):

```json
{
  "context": {"error_unit": "word",
              "error_rate": 0.04, "result": 4.0, "hits": 965,
              "substitutions": 20, "deletions": 15, "insertions": 5, "reference_units": 1000,
              "units_scored": 50, "samples_total": 50, "samples_excluded": 0,
              "stop_reasons": {"eos": 50},
              "prompt_profile": "qwen25_omni",
              "scoring": {"normalizer": "open_asr_english", "aligner": "kaldialign_compound", "open_asr_exclusions": true}},
  "scored_samples": [
    {"id": "4d42027687cc",
     "messages": [{"role": "system", "content": "You are a speech recognition model."},
                  {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "", "format": "wav"}},
                                               {"type": "text", "text": "Transcribe the English audio into text without any punctuation marks."}]}],
     "completion": "CEFOPs role is to contribute to the shaping …", "is_correct": false,
     "measures": {"unit": "4d42027687cc", "hits": 18, "substitutions": 1, "deletions": 1, "insertions": 0,
                  "reference_units": 20, "reference": "cedefop s role is to contribute to the shaping …",
                  "hypothesis": "cefops role is to contribute to the shaping …"}}
  ]
}
```
