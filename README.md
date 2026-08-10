# pipette — edge-evals workspace

Python monorepo hosting two packages for LLM edge-evaluation:

| Package | Role |
|---|---|
| [**`pipette-scores`**](packages/pipette-scores/) | HTTP service that serves eval samples and scores client completions |
| [**`pipette-calibration`**](packages/pipette-calibration/) | Offline pipeline for building and calibrating eval datasets |

Plus shared infrastructure:

- **`datasets/`** — LFS-tracked canonical eval datasets (parquet/jsonl).
- **`vendor/ifstruct/`** — submodule: standalone IFStruct evaluator (`Liquid4All/ifstruct`).
- **`vendor/ifbench/`** — submodule: upstream IFBench instruction checkers (`allenai/IFBench`).
- **`docker/pipette-scores.Dockerfile`** — production image for the scores service.

## Supported evals

| Eval | Type | Description |
|------|------|-------------|
| `ifbench` | Generation | Instruction-following benchmark (backed by `vendor/ifbench`) |
| `ifstruct` | Generation | Structured format (JSON/YAML) generation (backed by `vendor/ifstruct`) |
| `gpqa_diamond` | Generation | Graduate-level science MCQ (A–D), generative + regex extraction |
| `math_500` | Generation | Competition math, `\boxed{}` answer graded by `score_math_generic` |

## Quick start

```bash
# Clone with submodules + LFS
git clone --recurse-submodules https://github.com/Liquid4All/pipette-scores.git
cd pipette-scores
git lfs pull

# Install the workspace
uv sync --extra dev

# Run the pipette-scores service (from workspace root)
uv run --package pipette-scores uvicorn pipette_scores.api.app:app --reload

# Run all tests
uv run pytest

# Run lint / format
uv run ruff check
uv run ruff format --check
```

### Running the tests

The full test suite runs with **no credentials and no network access**. All 152
tests pass offline:

```bash
uv sync --extra dev
uv run pytest
```

No test requires a HuggingFace token. The suite never contacts the HuggingFace
Hub: `packages/pipette-scores/tests/conftest.py` materializes small local
parquet/jsonl fixtures in `tmp_path` for every eval, and the GPQA fixture builds
a synthetic two-row stand-in whose `content_sha256` matches what the loader
verifies — so the gated upstream dataset is never downloaded. The tests in
`test_integration_scoring.py` are "integration" only in the sense that they
exercise the real loaders and scorers end to end; their inputs are the checked-in
fixtures under `tests/fixtures/`, the LFS-tracked `datasets/` tree, and
`vendor/ifbench/data/IFBench_test.jsonl` from the submodule.

HuggingFace access is needed only **outside** the test suite, to materialize the
gated GPQA-Diamond dataset for an actual eval run:

```bash
HF_TOKEN=... uv run --project packages/pipette-scores \
    python packages/pipette-scores/scripts/build_gpqa_diamond_dataset.py
```

That script is the only consumer of `HF_TOKEN` in the repository. It downloads
the gated [`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa)
dataset, which requires accepting the dataset's terms on HuggingFace first — see
[datasets](packages/pipette-scores/docs/datasets.md). Contributors who only need
to run tests, lint, or the service against the other evals do not need it.

## Documentation

Per-package docs:

- Scores: [api reference](packages/pipette-scores/docs/httpapi.md) · [docker deployment](packages/pipette-scores/docs/docker.md) · [scoring logic](packages/pipette-scores/docs/scoring.md) · [datasets](packages/pipette-scores/docs/datasets.md)
- Calibration: [architecture](packages/pipette-calibration/docs/architecture.md) · [creating a dataset](packages/pipette-calibration/docs/creating-a-dataset.md) · [representative selection](packages/pipette-calibration/docs/representative-selection.md) · [verification](packages/pipette-calibration/docs/verification.md) · [slurm runs](packages/pipette-calibration/docs/slurm.md)

## License

This repository's original work is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details and third-party attributions.
