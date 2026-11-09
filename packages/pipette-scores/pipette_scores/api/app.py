import asyncio
import functools
import hashlib
import html
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from pipette_scores import packages, scoring
from pipette_scores import dataset_catalog
from pipette_scores.dataset_catalog import _DEFAULT_ROOT, asr_prompt_profiles, prompt_profiles
from pipette_scores.scoring import profile_answers
from pipette_scores.hashing import canonical_sample, samples_digest
from pipette_scores.types import (
    ASR_EVAL_IDS,
    ChatMessage,
    DatasetSample,
    EvalId,
    EvalSample,
    ScoredSample,
    ScoreRequest,
    ScoreResponse,
    SamplesDigestResponse,
    SamplesResponse,
)
from pipette_scores.memory import rss_mb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(process)d] %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_APP_VERSION = os.environ.get("PIPETTE_SCORES_VERSION", "dev")
# The eval packages' store (packages.py); its index for this version holds the
# sets' digests, so a restart need not recompute them.
_PACKAGES = (
    packages.open_store(os.environ["PIPETTE_SCORES_PACKAGES"]) if "PIPETTE_SCORES_PACKAGES" in os.environ else None
)

_INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

# An ASR set's samples are streamed, never built whole: they carry their audio
# (SPGISpeech's are gigabytes). The body is `{"samples": [...]}` written one
# sample at a time from its canonical JSON, the bytes its digest hashes.
_ASR_BODY_OPEN = b'{"samples":['
_ASR_BODY_CLOSE = b"]}"


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


# Prompt sets a worker keeps: a VLM set holds its base64 images (POPE about
# 1.1 GB), and each (eval, dataset, profile) is a set of its own, so an unbounded
# cache grows with every profile served. The few most recent are kept; another
# is rebuilt from its parquet (seconds) when asked for again.
_PROMPT_SETS = 4


@functools.lru_cache(maxsize=_PROMPT_SETS)
def _load_prompts_cached(eval_id: str, dataset_name: str, prompt_profile: str | None = None) -> list[EvalSample]:
    # The uncached loader: the catalog's own cache would keep every set anyway.
    return dataset_catalog.load_prompt_samples.__wrapped__(eval_id, dataset_name, prompt_profile=prompt_profile)


# Digest and sample count per (eval, dataset, profile): all the digest route
# keeps resident. A cache that pins a copy of each set probes every pinned set
# every minute, and holding their samples instead would keep every one of them
# in memory for good (POPE's alone are a gigabyte). Datasets are immutable
# within a process (a new build ships under a new name, and a deploy restarts),
# so a digest, once known, stays true.
_DIGESTS: dict[tuple[str, str, str | None], tuple[str, int]] = {}
# The digest of each list /samples served, by the list's identity: the cached
# list is the same object on every request, so it is hashed once.
_SERVED: dict[tuple[str, str, str | None], tuple[int, str]] = {}
# Cold builds run here, one at a time: each holds a whole sample set while it
# is hashed, after a deploy every pinned set is cold at once, and waiting on a
# thread of its own keeps them from holding the threads /score runs on. A
# second request for a set queued behind its build finds the digest done.
_DIGEST_BUILDER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="samples-digest")


def _build_prompts(eval_id: str, dataset_name: str, prompt_profile: str | None) -> list[EvalSample]:
    """The samples ``/samples`` serves, built afresh outside every cache."""
    return dataset_catalog.load_prompt_samples.__wrapped__(eval_id, dataset_name, prompt_profile=prompt_profile)


def _samples_digest(eval_id: str, dataset_name: str, prompt_profile: str | None) -> tuple[str, int]:
    """The digest and count of a set, built once per process and dropped once
    hashed. Runs on ``_DIGEST_BUILDER``. An ASR set is streamed through once for
    every profile at a time (dataset_catalog.asr_prompt_digests), so the
    probes for its other profiles find their digests done. A set this server
    holds and the packages index lists for this version
    (``PIPETTE_SCORES_PACKAGES``) is not built: the index gives the digest of
    each of its profiles."""
    key = (eval_id, dataset_name, prompt_profile)
    # Only a VLM or ASR set on disk: the index also lists versions uploaded since this server synced its datasets.
    if (
        key not in _DIGESTS
        and _PACKAGES is not None
        and packages.profiles(eval_id)
        and dataset_catalog._dataset_dir(eval_id, dataset_name).exists()
    ):
        index = _PACKAGES.read(packages.index_key(_APP_VERSION, eval_id, dataset_name))
        if index is not None:
            for entry in json.loads(index)["profiles"]:
                _DIGESTS.setdefault(
                    (eval_id, dataset_name, entry["prompt_profile"]), (entry["samples_digest"], entry["sample_count"])
                )
    if (known := _DIGESTS.get(key)) is None and eval_id in ASR_EVAL_IDS:
        for profile, digest in dataset_catalog.asr_prompt_digests(EvalId(eval_id), dataset_name).items():
            _DIGESTS.setdefault((eval_id, dataset_name, profile), digest)
        return _DIGESTS[key]
    if known is None:
        built = _build_prompts(eval_id, dataset_name, prompt_profile)
        # A /samples request may have stored the digest meanwhile; it hashed
        # the body it served, so it stands.
        known = _DIGESTS.setdefault(key, (samples_digest(built), len(built)))
        del built
    return known


def _served_digest(eval_id: str, dataset_name: str, prompt_profile: str | None, samples: list[EvalSample]) -> str:
    """The digest of the body ``/samples`` is about to send, hashed from that
    very list (once per list); the digest route then reports it too."""
    key = (eval_id, dataset_name, prompt_profile)
    served = _SERVED.get(key)
    if served is not None and served[0] == id(samples):
        return served[1]
    digest = samples_digest(samples)
    _SERVED[key] = (id(samples), digest)
    known = _DIGESTS.get(key)
    if known is not None and known[0] != digest:
        logger.warning(
            "samples changed under a cached digest eval=%s dataset=%s profile=%s was=%s now=%s",
            eval_id,
            dataset_name,
            prompt_profile,
            known[0],
            digest,
        )
    _DIGESTS[key] = (digest, len(samples))
    return digest


@functools.cache
def _load_dataset_cached(eval_id: str, dataset_name: str) -> list[DatasetSample]:
    return dataset_catalog.load_eval_samples(eval_id, dataset_name)


def _validate_eval_id(eval_id: str) -> None:
    try:
        EvalId(eval_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Eval not found: {eval_id}")


def _validate_prompt_profile(eval_id: str, prompt_profile: str | None) -> None:
    if eval_id in ASR_EVAL_IDS:
        # An ASR dataset holds no prompt: the profile is the request.
        known = ", ".join(asr_prompt_profiles.PROFILES)
        if prompt_profile is None:
            raise HTTPException(status_code=400, detail=f"{eval_id} needs a prompt_profile (known: {known})")
        if prompt_profile not in asr_prompt_profiles.PROFILES:
            raise HTTPException(
                status_code=400, detail=f"Unknown ASR prompt profile: {prompt_profile} (known: {known})"
            )
        return
    if prompt_profile is None:
        return
    if prompt_profile not in prompt_profiles.PROFILES:
        known = ", ".join(prompt_profiles.PROFILES)
        raise HTTPException(status_code=400, detail=f"Unknown prompt profile: {prompt_profile} (known: {known})")
    if eval_id not in prompt_profiles.BENCHMARKS:
        raise HTTPException(status_code=400, detail=f"{eval_id} has no prompt profiles")


def _refuse_paging(eval_id: str, request: Request) -> None:
    """ASR samples are not paged; ``offset``/``limit`` are refused rather than
    ignored, so a caller expecting a page does not silently get the whole set."""
    if eval_id in ASR_EVAL_IDS and {"offset", "limit"} & set(request.query_params):
        raise HTTPException(
            status_code=400,
            detail=f"{eval_id}'s samples are not paged: GET /samples streams the whole set, with its digest",
        )


def _without_images(message: ChatMessage) -> ChatMessage:
    """The message as a scored sample carries it: its text without its images.
    The caller holds the corpus they came from; sending them back with every
    score would copy it whole (about 1 GB for POPE)."""
    if isinstance(message.content, str):
        return message
    return message.model_copy(update={"content": [part for part in message.content if part.get("type") != "image_url"]})


def _without_audio(message: ChatMessage) -> ChatMessage:
    """An ASR message as a scored sample carries it: its text, and each audio
    part kept as a placeholder with its format and no data, so the turn order
    stays auditable. The caller holds the clips (gigabytes for some sets)."""
    if isinstance(message.content, str):
        return message
    return message.model_copy(
        update={
            "content": [
                {**part, "input_audio": {**part.get("input_audio", {}), "data": ""}}
                if part.get("type") == "input_audio"
                else part
                for part in message.content
            ]
        }
    )


def _format_id_error(label: str, ids: list[str], max_show: int = 10) -> str:
    head = ", ".join(ids[:max_show])
    suffix = f" (and {len(ids) - max_show} more)" if len(ids) > max_show else ""
    return f"{label}: {head}{suffix}"


def _load_prompts_or_error(eval_id: str, dataset_name: str, prompt_profile: str | None = None) -> list[EvalSample]:
    try:
        return _load_prompts_cached(eval_id, dataset_name, prompt_profile)
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


async def _samples_digest_or_error(eval_id: str, dataset_name: str, prompt_profile: str | None) -> tuple[str, int]:
    if (known := _DIGESTS.get((eval_id, dataset_name, prompt_profile))) is not None:
        return known
    try:
        # Off the event loop: the first digest of a large VLM set hashes ~1 GB,
        # of an ASR set every clip of it.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_DIGEST_BUILDER, _samples_digest, eval_id, dataset_name, prompt_profile)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("digesting samples failed eval=%s dataset=%s", eval_id, dataset_name)
        raise HTTPException(status_code=500, detail=f"Error digesting samples: {e}")


def _asr_body(eval_id: str, dataset_name: str, prompt_profile: str, digest: str, count: int) -> Iterator[bytes]:
    """The streamed ``/samples`` body of an ASR set, re-hashed as it is written.
    Its ``X-Samples-Digest`` went out before the first byte, so a body that does
    not hash to it is cut off before its closing bracket: a client then holds
    unparseable JSON, never a complete body under the wrong digest."""
    hasher = hashlib.sha256()
    served = 0
    yield _ASR_BODY_OPEN
    for sample in dataset_catalog.iter_asr_prompt_samples(EvalId(eval_id), dataset_name, prompt_profile):
        line = canonical_sample(sample)
        hasher.update(line)
        hasher.update(b"\n")
        yield line if served == 0 else b"," + line
        served += 1
    if (f"sha256:{hasher.hexdigest()}", served) != (digest, count):
        logger.error(
            "streamed samples do not match their digest eval=%s dataset=%s profile=%s sent=%s/%d streamed=%s/%d",
            eval_id,
            dataset_name,
            prompt_profile,
            digest,
            count,
            f"sha256:{hasher.hexdigest()}",
            served,
        )
        raise RuntimeError(f"{eval_id}/{dataset_name} streamed other samples than its digest {digest}")
    yield _ASR_BODY_CLOSE


@app.get("/evals/{eval_id}/datasets/{dataset_name}/samples")
async def get_samples_endpoint(
    eval_id: str, dataset_name: str, request: Request, response: Response, prompt_profile: str | None = None
) -> SamplesResponse:
    """``prompt_profile`` swaps the dataset's prompt for a model family's
    VLMEvalKit prompt (dataset_catalog/prompt_profiles.py), or, for an ASR eval,
    names the family whose request each clip becomes (required). The
    ``X-Samples-Digest`` header is the body's digest, so a client that keeps a
    copy knows which content it took without a second request.

    An ASR set comes streamed, one sample at a time, after its digest is known
    (from the digest route's cache, or one pass over the set): its samples carry
    their audio, so they are never held whole."""
    _validate_eval_id(eval_id)
    _validate_prompt_profile(eval_id, prompt_profile)
    _refuse_paging(eval_id, request)
    if eval_id in ASR_EVAL_IDS:
        assert prompt_profile is not None  # _validate_prompt_profile requires it for ASR
        digest, count = await _samples_digest_or_error(eval_id, dataset_name, prompt_profile)
        return StreamingResponse(  # type: ignore[return-value]
            _asr_body(eval_id, dataset_name, prompt_profile, digest, count),
            media_type="application/json",
            headers={"X-Samples-Digest": digest},
        )
    samples = _load_prompts_or_error(eval_id, dataset_name, prompt_profile)
    try:
        digest = await asyncio.to_thread(_served_digest, eval_id, dataset_name, prompt_profile, samples)
    except Exception as e:
        logger.exception("digesting samples failed eval=%s dataset=%s", eval_id, dataset_name)
        raise HTTPException(status_code=500, detail=f"Error digesting samples: {e}")
    response.headers["X-Samples-Digest"] = digest
    return SamplesResponse(samples=samples)


@app.get("/evals/{eval_id}/datasets/{dataset_name}/samples/digest")
async def get_samples_digest_endpoint(
    eval_id: str, dataset_name: str, request: Request, prompt_profile: str | None = None
) -> SamplesDigestResponse:
    """The digest of what ``/samples`` serves for the same request, without
    the samples: a few bytes for a caller that only checks its copy is current.
    The samples are built to be hashed and not kept."""
    _validate_eval_id(eval_id)
    _validate_prompt_profile(eval_id, prompt_profile)
    _refuse_paging(eval_id, request)
    digest, count = await _samples_digest_or_error(eval_id, dataset_name, prompt_profile)
    return SamplesDigestResponse(digest=digest, sample_count=count)


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
    _validate_prompt_profile(eval_id, request.prompt_profile)
    if eval_id in ASR_EVAL_IDS and request.enable_thinking:
        raise HTTPException(
            status_code=400,
            detail=f"{eval_id} runs with thinking off: enable_thinking=true is not comparable",
        )
    if request.samples_digest is not None:
        served, _ = await _samples_digest_or_error(eval_id, dataset_name, request.prompt_profile)
        if served != request.samples_digest:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"the run was given samples {request.samples_digest}, but this server serves "
                    f"{served} for them now: its answers are not to these prompts"
                ),
            )

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

    # A VLM or ASR eval is scored only on one answer to each of its samples:
    # its headline is defined over the whole dataset, and a run that lacks an
    # answer is not a result, however its gaps would score. pipette-mgmt
    # refuses such a run first; this holds the line for any caller. (An ASR
    # sample the leaderboard's row filter leaves out of the score still needs
    # its answer.)
    kind = "a VLM eval" if dataset_catalog.is_vlm_eval(eval_id) else "an ASR eval" if eval_id in ASR_EVAL_IDS else None
    if kind is not None:
        unanswered = sorted(known_ids - {c.id for c in request.completions})
        failed = sorted(c.id for c in request.completions if c.stop_reason == "failure")
        if unanswered or failed:
            gaps = [
                _format_id_error(label, ids) for label, ids in (("unanswered", unanswered), ("failed", failed)) if ids
            ]
            raise HTTPException(
                status_code=400,
                detail=f"{kind} is scored only with an answer to every sample; {'; '.join(gaps)}",
            )

    try:
        # Off-load CPU-bound scoring to a worker thread so the event loop
        # remains free to service other requests on this uvicorn worker.
        # A profile whose VLMEvalKit wrapper cuts each reply down to its answer
        # is scored on what the wrapper hands on (scoring/profile_answers.py);
        # the per-sample rows below keep the reply as sent.
        graded = profile_answers.apply(request.prompt_profile, eval_id, request.completions)
        scores, context = await asyncio.to_thread(
            scoring.score,
            EvalId(eval_id),
            graded,
            samples,
            enable_thinking=request.enable_thinking,
            prompt_profile=request.prompt_profile,
        )
    except Exception as e:
        logger.exception("scoring failed eval=%s dataset=%s", eval_id, dataset_name)
        raise HTTPException(status_code=500, detail=f"Error scoring: {e}")

    # Prompts are only needed for per-sample enrichment, so load them after
    # scoring — saves a parquet read on the fail-fast paths above. With the
    # run's prompt profile, so each sample shows the prompt the model was sent.
    prompts = _load_prompts_or_error(eval_id, dataset_name, request.prompt_profile)
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

    # The prompts' text only: the caller holds the media.
    text_only = _without_audio if eval_id in ASR_EVAL_IDS else _without_images
    scored_samples = [
        ScoredSample(
            id=s.id,
            messages=[text_only(m) for m in prompts_by_id[s.id].messages],
            completion=completions_by_id[s.id].completion,
            is_correct=s.is_correct,
            measures=s.measures,
            chat_template_kwargs=prompts_by_id[s.id].chat_template_kwargs,
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
