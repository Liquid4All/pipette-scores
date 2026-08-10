# API Reference

Base URL: `http://localhost:8000` (default uvicorn)

## 1. Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PIPETTE_SCORES_DATA_DIR` | Root directory for dataset files (see [Data Storage](datasets.md)) | Parent of the `pipette_scores` package |
| `PIPETTE_SCORES_VERSION` | Value reported as `runtime_version` in the `POST /score` response | `dev` |

## 2. Caching

The server caches loaded datasets and prompt samples in memory using `functools.cache`. Once an `(eval_id, dataset_name)` pair is loaded, subsequent requests for the same pair are served from cache. The cache persists for the lifetime of the process.

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

**Response** `200 OK` — `SamplesResponse`

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

**Error Responses:**

| Status | Condition |
|--------|-----------|
| 404 | `eval_id` not found or dataset directory does not exist |
| 500 | Error loading samples from parquet |

**Example:**

```bash
curl "http://localhost:8000/evals/math_500/datasets/2026.06.1/samples"
```

---

### 3.3. `POST /score`

Score completions against ground truth. Returns per-sample scores enriched with the original prompt messages and completions so the caller can persist full audit records without a second round trip.

**Request Body** — `ScoreRequest`

```json
{
  "eval_id": "math_500",
  "dataset_name": "2026.06.1",
  "completions": [
    {"id": "a1b2c3d4e5f6", "completion": "The answer is \\boxed{42}"}
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `eval_id` | string | Eval identifier |
| `dataset_name` | string | Dataset to score against |
| `completions` | list[SampleCompletion] | Completions to score |
| `completions[*].id` | string | Must match a sample ID from the samples endpoint |
| `completions[*].completion` | string | Raw model output text |

**Response** `200 OK` — `ScoreResponse`

`context` is a flat `string → JSON value` map of eval-specific aggregates (numbers typical, other scalars allowed).

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
| `scored_samples[*].messages` | list[ChatMessage] | Prompt from the dataset |
| `scored_samples[*].completion` | string | Echoed from the request |
| `scored_samples[*].is_correct` | bool | Whether the completion was scored correct |

**Error Responses:**

| Status | Condition |
|--------|-----------|
| 404 | `eval_id` not found or dataset directory does not exist |
| 422 | Malformed body (e.g. missing `eval_id` / `dataset_name`) |
| 500 | Error loading dataset, scoring failure, or scorer returned an id not present in request completions |

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
