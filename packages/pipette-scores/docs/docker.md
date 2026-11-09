# Docker

The repository includes a production-like multi-stage Docker image that runs
the API with Gunicorn and `uvicorn.workers.UvicornWorker`.

## 1. Build

Run from the **workspace root** (one level above `packages/pipette-scores/`):

```bash
docker build -f docker/pipette-scores.Dockerfile -t pipette-scores .
```

The build context is the workspace so the Dockerfile can pull in the workspace
lockfile, the `pipette-scores` package source under `packages/pipette-scores/`,
and `vendor/` (the `ifstruct` and `ifbench` submodules and the vendored
`ifbench-checkers`). Clone with `--recurse-submodules` and run `git lfs pull`
before you build.

## 2. Run

The image bundles the workspace's canonical `datasets/` tree under
`${PIPETTE_SCORES_DATA_DIR}/datasets` (default `/datasets/datasets`). The
bundle holds only the committed `ifstruct` dataset. To serve `ifstruct`, run
the image with no volume mount:

```bash
docker run --rm -p 8000:8000 pipette-scores
```

To serve the other evals or a different datasets tree,
override `${PIPETTE_SCORES_DATA_DIR}/datasets` with a read-only mount. Either
shadow the bundled path directly:

```bash
docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/datasets:/datasets/datasets:ro" \
  pipette-scores
```

Or point `PIPETTE_SCORES_DATA_DIR` at a different prefix:

```bash
docker run --rm \
  -p 8000:8000 \
  -e PIPETTE_SCORES_DATA_DIR=/srv \
  -v "$(pwd)/datasets:/srv/datasets:ro" \
  pipette-scores
```

The app always looks at `${PIPETTE_SCORES_DATA_DIR}/datasets` — point `-v` at
whatever host path holds the JSONL/parquet files for the evals you care about.

## 3. Published Image

CI publishes multi-arch images (`linux/amd64`, `linux/arm64`) to
`ghcr.io/liquid4all/pipette-scores`:

- A push to a `release/<train>` branch publishes the tag
  `<train>-<commits since main>-g<commit>` and creates a GitHub release with
  that tag.
- A push to `main` publishes the tag `<commit>`.
- A pull request builds the image but does not publish it.

`ci/version.sh` prints the tag that the current commit publishes. Example:

```bash
docker pull ghcr.io/liquid4all/pipette-scores:2026.08.1-3-ga1b2c3d4ab
```

## 4. Notes

- The image runs as a non-root user.
- The runtime image contains the built virtualenv (with the application) and
  the bundled `datasets/` tree.
- Gunicorn bind host, port, and worker count are controlled by
  `PIPETTE_SCORES_HOST`, `PIPETTE_SCORES_PORT`, and
  `PIPETTE_SCORES_WORKERS`.
- The image bakes the release version into `PIPETTE_SCORES_VERSION`.
  `POST /score` returns that value as `runtime_version`. A local build without
  `--build-arg APP_VERSION=<version>` reports `dev`.
