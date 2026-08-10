FROM python:3.12-slim-trixie AS builder

ARG APP_VERSION=dev

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# Workspace manifest + per-package manifests for dep resolution. Calibration's
# source isn't installed (--package pipette-scores scopes the sync), but its
# manifest is required so uv's workspace member discovery succeeds.
# README + the third-party attribution force-included into the wheel.
COPY pyproject.toml uv.lock README.md NOTICE THIRD_PARTY_LICENSES.md ./
COPY packages/pipette-scores/pyproject.toml ./packages/pipette-scores/pyproject.toml
COPY packages/pipette-calibration/pyproject.toml ./packages/pipette-calibration/pyproject.toml
COPY vendor ./vendor

# Install only pipette-scores' runtime deps, not the project itself yet.
RUN uv sync --package pipette-scores --frozen --no-dev --no-install-project --no-editable

# Then bring in the source and install the project.
COPY packages/pipette-scores/pipette_scores ./packages/pipette-scores/pipette_scores
RUN uv sync --package pipette-scores --frozen --no-dev --no-editable

# Pre-download NLTK corpora into a location bundled with the venv. Without
# this, concurrent gunicorn workers race on first-request lazy downloads and
# corrupt the target zips. /app/.venv/nltk_data is on nltk.data.path by
# default, so runtime needs no extra config.
RUN /app/.venv/bin/python -c "import nltk; [nltk.download(r, download_dir='/app/.venv/nltk_data', quiet=True) for r in ('punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger_eng')]"


FROM python:3.12-slim-trixie

ARG APP_VERSION=dev

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH \
    PIPETTE_SCORES_VERSION=${APP_VERSION} \
    PIPETTE_SCORES_HOST=0.0.0.0 \
    PIPETTE_SCORES_PORT=8000 \
    PIPETTE_SCORES_WORKERS=4 \
    PIPETTE_SCORES_DATA_DIR=/datasets

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
# Bundled datasets. PIPETTE_SCORES_DATA_DIR is /datasets and the resolver
# appends /datasets/, so the tree lives at /datasets/datasets. A volume
# mount on /datasets/datasets at runtime overrides the bundled copy.
COPY --chown=appuser:appuser datasets /datasets/datasets

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "gunicorn pipette_scores.api.app:app --bind ${PIPETTE_SCORES_HOST}:${PIPETTE_SCORES_PORT} --workers ${PIPETTE_SCORES_WORKERS} --worker-class uvicorn.workers.UvicornWorker"]
