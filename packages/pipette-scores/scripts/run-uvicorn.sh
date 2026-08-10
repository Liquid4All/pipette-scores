#!/bin/bash
exec uv run uvicorn pipette_scores.api.app:app \
    --host "${PIPETTE_SCORES_HOST:-0.0.0.0}" \
    --port "${PIPETTE_SCORES_PORT:-8000}" \
    --reload
