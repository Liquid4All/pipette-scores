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
and the shared `vendor/` submodule content.

## 2. Run

The image bundles the workspace's canonical `datasets/` tree under
`${PIPETTE_SCORES_DATA_DIR}/datasets` (default `/datasets/datasets`), so the
simplest run needs no volume mount:

```bash
docker run --rm -p 8000:8000 pipette-scores
```

To serve a different datasets tree (e.g. private or newer than the bundle),
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

CI publishes versioned images to:

```bash
liquidai/pipette-scores:<release-version>
```

Example:

```bash
docker pull liquidai/pipette-scores:2026.03.15-abcdef0
```

## 4. Notes

- The image runs as a non-root user.
- The runtime image only contains the built virtualenv and application code.
- Gunicorn bind host, port, and worker count are controlled by
  `PIPETTE_SCORES_HOST`, `PIPETTE_SCORES_PORT`, and
  `PIPETTE_SCORES_WORKERS`.
- The image bakes the release version into `PIPETTE_SCORES_VERSION`, and
  `GET /health` returns that value.
