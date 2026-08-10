# Contributing

Thanks for your interest in contributing to `pipette-scores`.

## Getting Started

Clone the repository with submodules and LFS objects:

```bash
git clone --recurse-submodules https://github.com/Liquid4All/pipette-scores.git
cd pipette-scores
git lfs pull
```

Install the workspace and run the default checks:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check
uv run ruff format --check
```

The test suite is expected to run without HuggingFace credentials or network access.

## Pull Requests

Keep changes focused on one problem at a time.

Include tests when changing scoring behavior, dataset loading, API behavior, or calibration logic.

Update documentation when changing public commands, API shapes, dataset formats, scoring methodology, or deployment behavior.

Do not commit generated datasets, private credentials, model outputs, local caches, or machine-specific paths.

## Licensing

By contributing, you agree that your contribution is submitted under the Apache License, Version 2.0, unless you explicitly state otherwise.

Retain third-party copyright and license notices when adding or adapting external code, datasets, or assets.
