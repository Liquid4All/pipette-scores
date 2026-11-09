# pipette — edge-evals workspace

Python monorepo hosting two packages for LLM edge-evaluation:

| Package | Role |
|---|---|
| [**`pipette-scores`**](packages/pipette-scores/) | HTTP service that serves eval samples and scores client completions |
| [**`pipette-calibration`**](packages/pipette-calibration/) | Offline pipeline for building and calibrating eval datasets |

Plus shared infrastructure:

- **`datasets/`** — the committed `ifstruct` dataset (LFS). Builder scripts create the other eval datasets in the same layout.
- **`vendor/ifstruct/`** — submodule: standalone IFStruct evaluator (`Liquid4All/ifstruct`).
- **`vendor/ifbench/`** — submodule: upstream IFBench test set (`allenai/IFBench`).
- **`vendor/ifbench-checkers/`** — the IFBench instruction checkers (`allenai/IFBench` @ `1a902f9`, the revision liquid_evals scores with), vendored with small service patches.
- **`docker/pipette-scores.Dockerfile`** — production image for the scores service.

## Supported evals

| Eval | Type | Description |
|------|------|-------------|
| `ifbench` | Generation | Instruction-following benchmark (test set from `vendor/ifbench`, checkers from `vendor/ifbench-checkers`) |
| `ifstruct` | Generation | Structured format (JSON/YAML) generation (backed by `vendor/ifstruct`) |
| `gpqa_diamond` | Generation | Graduate-level science MCQ (A–D), generative + regex extraction |
| `math_500` | Generation | Competition math, `\boxed{}` answer graded by the grader that the dataset version records |
| VLM evals | Vision | `mme`, `mmstar`, `ocrbench`, `mmmu_val`, `realworldqa`, `mmbench_en_dev`, `pope` (see [datasets](packages/pipette-scores/docs/datasets.md)) |
| ASR evals | Speech | `asr_esb_*` (Open ASR Leaderboard English sets) and `asr_fleurs_*` (FLEURS locales), scored by word error rate (see [ASR evals](packages/pipette-scores/docs/asr-evals.md)) |

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

The full test suite runs with **no credentials and no network access**. The
live ASR reference tests are skipped unless `PIPETTE_ASR_REFERENCE_ENV` is set:

```bash
uv sync --extra dev
uv run pytest
```

No test requires a HuggingFace token. The suite never contacts the HuggingFace
Hub: `packages/pipette-scores/tests/conftest.py` materializes small local
parquet/jsonl fixtures in `tmp_path` for the text evals, and the GPQA fixture builds
a synthetic two-row stand-in whose `content_sha256` matches what the loader
verifies — so the gated upstream dataset is never downloaded. The VLM and ASR
tests build their fixtures in their own test files. The tests in
`test_integration_scoring.py` are "integration" only in the sense that they
exercise the real loaders and scorers end to end; their inputs are the checked-in
fixtures under `tests/fixtures/`, the LFS-tracked `datasets/` tree, and
`vendor/ifbench/data/IFBench_test.jsonl` from the submodule.

HuggingFace access is needed only **outside** the test suite, to build the gated
datasets for an actual eval run. For example, GPQA-Diamond:

```bash
HF_TOKEN=... uv run --project packages/pipette-scores \
    python packages/pipette-scores/scripts/build_gpqa_diamond_dataset.py
```

That script downloads the gated
[`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa) dataset.
Accept the dataset's terms on HuggingFace first — see
[datasets](packages/pipette-scores/docs/datasets.md). The ASR builder
(`scripts/build_asr_datasets.py`) also reads gated sets, such as GigaSpeech.
Contributors who only run tests, lint, or the service against already built
datasets do not need a token.

## Documentation

Per-package docs:

- Scores: [api reference](packages/pipette-scores/docs/httpapi.md) · [docker deployment](packages/pipette-scores/docs/docker.md) · [scoring logic](packages/pipette-scores/docs/scoring.md) · [datasets](packages/pipette-scores/docs/datasets.md) · [ASR evals](packages/pipette-scores/docs/asr-evals.md) · [eval packages](packages/pipette-scores/docs/packages.md)
- Calibration: [architecture](packages/pipette-calibration/docs/architecture.md) · [creating a dataset](packages/pipette-calibration/docs/creating-a-dataset.md) · [representative selection](packages/pipette-calibration/docs/representative-selection.md) · [verification](packages/pipette-calibration/docs/verification.md) · [slurm runs](packages/pipette-calibration/docs/slurm.md)

## License

This repository's original work is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details and third-party attributions.
