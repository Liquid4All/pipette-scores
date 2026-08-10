#!/bin/bash
# Dev convenience: run pipette-scores under gunicorn locally.
# NOTE: the prod invocation lives in docker/pipette-scores.Dockerfile's CMD.
# Keep the two in sync if you change the app path or add flags.
exec uv run gunicorn pipette_scores.api.app:app \
    --bind "${PIPETTE_SCORES_HOST:-0.0.0.0}:${PIPETTE_SCORES_PORT:-8000}" \
    --workers "${PIPETTE_SCORES_WORKERS:-4}" \
    --worker-class uvicorn.workers.UvicornWorker
