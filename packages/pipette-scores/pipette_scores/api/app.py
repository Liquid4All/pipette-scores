import asyncio
import functools
import html
import logging
import os
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from pipette_scores import scoring
from pipette_scores import dataset_catalog
from pipette_scores.dataset_catalog import _DEFAULT_ROOT
from pipette_scores.types import (
    DatasetSample,
    EvalId,
    EvalSample,
    ScoredSample,
    ScoreRequest,
    ScoreResponse,
    SamplesResponse,
)
from pipette_scores.memory import rss_mb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(process)d] %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_APP_VERSION = os.environ.get("PIPETTE_SCORES_VERSION", "dev")

_INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    datasets_dir = _DEFAULT_ROOT / "datasets"
    if not datasets_dir.is_dir() or not any(datasets_dir.iterdir()):
        logger.warning(
            "No datasets found in %s. "
            "Set the PIPETTE_SCORES_DATA_DIR environment variable to the directory containing your datasets.",
            datasets_dir,
        )
    yield


app = FastAPI(
    title="Edge Evals Server",
    description="Stateless API for LLM evaluations",
    version=_APP_VERSION,
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@functools.cache
def _load_prompts_cached(eval_id: str, dataset_name: str) -> list[EvalSample]:
    return dataset_catalog.load_prompt_samples(eval_id, dataset_name)


@functools.cache
def _load_dataset_cached(eval_id: str, dataset_name: str) -> list[DatasetSample]:
    return dataset_catalog.load_eval_samples(eval_id, dataset_name)


def _validate_eval_id(eval_id: str) -> None:
    try:
        EvalId(eval_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Eval not found: {eval_id}")


def _format_id_error(label: str, ids: list[str], max_show: int = 10) -> str:
    head = ", ".join(ids[:max_show])
    suffix = f" (and {len(ids) - max_show} more)" if len(ids) > max_show else ""
    return f"{label}: {head}{suffix}"


def _load_prompts_or_error(eval_id: str, dataset_name: str) -> list[EvalSample]:
    try:
        return _load_prompts_cached(eval_id, dataset_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("loading prompts failed eval=%s dataset=%s", eval_id, dataset_name)
        raise HTTPException(status_code=500, detail=f"Error loading prompts: {e}")


def _load_dataset_or_error(eval_id: str, dataset_name: str) -> list[DatasetSample]:
    try:
        return _load_dataset_cached(eval_id, dataset_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("loading dataset failed eval=%s dataset=%s", eval_id, dataset_name)
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {e}")


@app.get("/evals/{eval_id}/datasets/{dataset_name}/samples")
async def get_samples_endpoint(eval_id: str, dataset_name: str) -> SamplesResponse:
    _validate_eval_id(eval_id)
    return SamplesResponse(samples=_load_prompts_or_error(eval_id, dataset_name))


@app.post("/score")
async def score_endpoint(request: ScoreRequest) -> ScoreResponse:
    eval_id = request.eval_id
    dataset_name = request.dataset_name

    completions_size = sum(len(c.completion) for c in request.completions)
    logger.info(
        "[mem] score start: rss=%.0fMB completions=%d total_chars=%d eval=%s dataset=%s",
        rss_mb(),
        len(request.completions),
        completions_size,
        eval_id,
        dataset_name,
    )

    _validate_eval_id(eval_id)

    # Reject duplicate ids up front — no dataset needed for this check, and
    # the per-id dicts below would silently drop all but the last.
    duplicate_ids = sorted(cid for cid, n in Counter(c.id for c in request.completions).items() if n > 1)
    if duplicate_ids:
        raise HTTPException(
            status_code=400,
            detail=_format_id_error("duplicate completion ids", duplicate_ids),
        )

    samples = _load_dataset_or_error(eval_id, dataset_name)
    logger.info("[mem] dataset loaded: rss=%.0fMB samples=%d", rss_mb(), len(samples))

    # Validate every completion id has a matching dataset sample; otherwise the
    # scorer would KeyError deep in user code. Report as 400 with the offenders.
    known_ids = {s.id for s in samples}
    unknown_ids = sorted({c.id for c in request.completions if c.id not in known_ids})
    if unknown_ids:
        raise HTTPException(
            status_code=400,
            detail=_format_id_error("completions reference ids not in dataset", unknown_ids),
        )

    try:
        # Off-load CPU-bound scoring to a worker thread so the event loop
        # remains free to service other requests on this uvicorn worker.
        scores, context = await asyncio.to_thread(scoring.score, eval_id, request.completions, samples)
    except Exception as e:
        logger.exception("scoring failed eval=%s dataset=%s", eval_id, dataset_name)
        raise HTTPException(status_code=500, detail=f"Error scoring: {e}")

    # Prompts are only needed for per-sample enrichment, so load them after
    # scoring — saves a parquet read on the fail-fast paths above.
    prompts = _load_prompts_or_error(eval_id, dataset_name)
    prompts_by_id = {p.id: p for p in prompts}
    completions_by_id = {c.id: c for c in request.completions}

    # Scorer-side guard: unreachable under the ingress checks above (scorers
    # iterate request.completions, so output ids ⊆ request ids ⊆ dataset ids),
    # but kept as a tripwire for scorer-implementation drift.
    hallucinated_ids = [s.id for s in scores if s.id not in prompts_by_id or s.id not in completions_by_id]
    if hallucinated_ids:
        raise HTTPException(
            status_code=500,
            detail=f"Scorer returned unknown ids: {hallucinated_ids}",
        )

    scored_samples = [
        ScoredSample(
            id=s.id,
            messages=prompts_by_id[s.id].messages,
            completion=completions_by_id[s.id].completion,
            is_correct=s.is_correct,
        )
        for s in scores
    ]

    logger.info("[mem] scoring done: rss=%.0fMB", rss_mb())
    return ScoreResponse(
        runtime_version=_APP_VERSION,
        scored_samples=scored_samples,
        context=context,
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    return HTMLResponse(_INDEX_HTML.replace("{version}", html.escape(_APP_VERSION)))


# 204 to suppress the browser-auto-request 404.
@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
async def health():
    return {"status": "ok"}
