import atexit
import json
import logging
import os
import pathlib
import shutil
import tempfile
import time

from pipette_scores import dataset_catalog
from pipette_scores.types import EvalSample, SampleCompletion

from pipette_calibration.inference.spec import GenerationParams, get_generation_params

logger = logging.getLogger(__name__)


def _get_choice_token_ids(tokenizer, choices: list[str]) -> list[int]:
    token_ids: set[int] = set()
    for choice in choices:
        for variant in [choice, f" {choice}", choice.lower(), f" {choice.lower()}"]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.add(ids[0])
    return sorted(token_ids)


def _shard_list(items, shard: int, num_shards: int):
    return items[shard::num_shards]


def _patched_tokenizer_dir(model: str) -> str | None:
    """Return a local dir with a normalized tokenizer, or None if not needed.

    Some LiquidAI models set ``tokenizer_class="TokenizersBackend"`` (an
    internal name, not a real transformers class) and ship no ``.py`` files
    to define it — so both ``trust_remote_code`` paths fail. ``tokenizer.json``
    itself is valid HF fast-tokenizer format, so if we strip the bogus class
    name and a few LiquidAI-internal keys, ``AutoTokenizer`` falls back to
    ``PreTrainedTokenizerFast`` and loads correctly.

    This is a workaround for a model-repo publishing issue. The proper fix is
    to clean up the affected model repos on HuggingFace; this patcher should
    be removed once that lands.
    """
    from huggingface_hub import hf_hub_download

    tok_config_path = hf_hub_download(model, "tokenizer_config.json")
    with open(tok_config_path) as f:
        cfg = json.load(f)
    if cfg.get("tokenizer_class") != "TokenizersBackend":
        return None

    # Safety: `model_specific_special_tokens` is the only stripped key that
    # *could* carry semantically load-bearing info. Refuse to proceed if a
    # future model populates it — better a loud crash than silent output
    # drift.
    mst = cfg.get("model_specific_special_tokens")
    if mst not in (None, {}, []):
        raise RuntimeError(
            f"Tokenizer patcher for {model} refuses to strip non-empty "
            f"model_specific_special_tokens={mst!r}. Upstream the fix to the "
            "model repo's tokenizer_config.json instead of extending this patcher."
        )

    # Keep patched dirs under $HOME, not /tmp — same reasoning as the
    # torch-inductor / triton cache redirects in run_eval.sh: shared /tmp
    # on compute nodes fills up unpredictably.
    parent = pathlib.Path.home() / ".cache" / "pipette-calibration" / "patched-tokenizers"
    parent.mkdir(parents=True, exist_ok=True)
    patched = pathlib.Path(tempfile.mkdtemp(prefix=f"pipette-tok-{model.replace('/', '__')}-", dir=str(parent)))
    atexit.register(shutil.rmtree, str(patched), True)

    for fname in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
        try:
            src = hf_hub_download(model, fname)
            shutil.copy(src, patched / fname)
        except Exception:
            pass  # optional files

    with open(patched / "tokenizer_config.json") as f:
        cfg = json.load(f)
    for key in ("tokenizer_class", "backend", "is_local", "model_specific_special_tokens"):
        cfg.pop(key, None)
    with open(patched / "tokenizer_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    logger.warning(
        "Patched tokenizer_config.json for %s (stripped LiquidAI-specific keys tokenizer_class/backend/is_local/model_specific_special_tokens) → %s",
        model,
        patched,
    )
    return str(patched)


def load_model(
    *,
    model: str,
    max_model_len: int,
    hf_cache: pathlib.Path | None = None,
    gpu_memory_utilization: float = 0.9,
):
    from huggingface_hub import hf_hub_download
    from vllm import LLM

    if hf_cache:
        os.environ["HF_HOME"] = str(hf_cache)

    config_path = hf_hub_download(model, "config.json")
    with open(config_path) as f:
        hf_config = json.load(f)
    model_max = hf_config.get("max_position_embeddings")
    if model_max and max_model_len > model_max:
        logger.info("Capping max_model_len from %d to %d (model limit)", max_model_len, model_max)
        max_model_len = model_max

    # auto_map can live in either config.json (custom model) or
    # tokenizer_config.json (custom tokenizer). Detect in either.
    trust_remote_code = "auto_map" in hf_config
    if not trust_remote_code:
        try:
            tok_config_path = hf_hub_download(model, "tokenizer_config.json")
        except Exception:
            tok_config_path = None
        if tok_config_path is not None:
            with open(tok_config_path) as f:
                tok_config = json.load(f)
            trust_remote_code = "auto_map" in tok_config
    if trust_remote_code:
        logger.info("Model or tokenizer has custom code (auto_map), enabling trust_remote_code")

    tokenizer_override = _patched_tokenizer_dir(model)

    t0 = time.time()
    logger.info("Loading model %s (max_model_len=%d)...", model, max_model_len)
    llm_kwargs: dict = {}
    if tokenizer_override is not None:
        llm_kwargs["tokenizer"] = tokenizer_override
    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
        **llm_kwargs,
    )
    tokenizer = llm.get_tokenizer()
    logger.info("Model ready in %.1fs", time.time() - t0)

    return llm, tokenizer


def write_completions(
    completions: list[SampleCompletion], *, shard: int | None, num_shards: int | None, output_dir: pathlib.Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".shard{shard}" if num_shards else ""
    path = output_dir / f"completions{suffix}.jsonl"
    with open(path, "w") as f:
        for c in completions:
            f.write(json.dumps({"id": c.id, "completion": c.completion}) + "\n")
    logger.info("Saved %d completions to %s", len(completions), path)


def _filter_by_token_length(prompts, sample_ids, tokenizer, max_model_len, max_output_tokens):
    """Split prompts into valid/skipped based on token length.

    A prompt is accepted if it fits in the context window with at least 1 token
    of room for output.  vLLM will generate up to min(max_output_tokens, remaining)
    tokens, so prompts that fit but leave less than max_output_tokens still get
    a (shorter) response rather than being skipped entirely.
    """
    # Accept any prompt that fits in the context window.
    max_input_len = max_model_len - 1

    logger.info(
        "Context budget: max_model_len=%d, max_output_tokens=%d, max_input_len=%d",
        max_model_len,
        max_output_tokens,
        max_input_len,
    )

    valid_prompts = []
    valid_ids = []
    valid_indices = []
    skipped = []
    token_lengths = []
    capped_count = 0

    for i, (prompt, sid) in enumerate(zip(prompts, sample_ids)):
        n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        token_lengths.append(n_tokens)
        if n_tokens > max_input_len:
            skipped.append((i, sid, n_tokens))
        else:
            remaining = max_model_len - n_tokens
            if remaining < max_output_tokens:
                capped_count += 1
            valid_prompts.append(prompt)
            valid_ids.append(sid)
            valid_indices.append(i)

    if token_lengths:
        logger.info(
            "Prompt token lengths: min=%d, max=%d, median=%d, mean=%d",
            min(token_lengths),
            max(token_lengths),
            sorted(token_lengths)[len(token_lengths) // 2],
            sum(token_lengths) // len(token_lengths),
        )

    logger.info(
        "Filter result: %d/%d accepted, %d skipped (exceeded %d input tokens)",
        len(valid_prompts),
        len(prompts),
        len(skipped),
        max_input_len,
    )
    if capped_count:
        logger.info(
            "  %d prompts will have output capped below max_output_tokens=%d due to context size",
            capped_count,
            max_output_tokens,
        )

    if skipped and len(skipped) <= 10:
        for _, sid, n in skipped:
            logger.warning("  Skipped %s: %d tokens (limit %d)", sid, n, max_input_len)
    elif skipped:
        for _, sid, n in skipped[:5]:
            logger.warning("  Skipped %s: %d tokens (limit %d)", sid, n, max_input_len)
        logger.warning("  ... and %d more", len(skipped) - 5)

    return valid_prompts, valid_ids, valid_indices, skipped


def run_mcq(
    *,
    gen: GenerationParams,
    samples: list[EvalSample],
    llm,
    tokenizer,
    shard: int | None = None,
    num_shards: int | None = None,
) -> list[SampleCompletion]:
    from vllm import SamplingParams

    total = len(samples)

    if num_shards:
        samples = _shard_list(samples, shard, num_shards)
        logger.info("%d samples, shard %d/%d -> %d samples", total, shard, num_shards, len(samples))
    else:
        logger.info("%d samples", len(samples))

    choices = gen.mcq_choices
    allowed_ids = _get_choice_token_ids(tokenizer, list(choices))
    logger.info("Constrained to %d token IDs for choices %s", len(allowed_ids), choices)

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        allowed_token_ids=allowed_ids,
    )

    prompts = [s.messages[0].content for s in samples]
    sample_ids = [s.id for s in samples]

    max_model_len = llm.llm_engine.model_config.max_model_len
    valid_prompts, valid_ids, valid_indices, skipped = _filter_by_token_length(
        prompts,
        sample_ids,
        tokenizer,
        max_model_len,
        max_output_tokens=1,
    )

    completions: list[SampleCompletion] = [None] * len(samples)

    n_generated = 0
    n_empty_generated = 0
    if valid_prompts:
        t0 = time.time()
        outputs = llm.generate(valid_prompts, sampling_params)
        elapsed = time.time() - t0
        n_generated = len(outputs)
        logger.info("Generated %d completions in %.1fs (%.1f samples/s)", n_generated, elapsed, n_generated / elapsed)
        for idx, sid, out in zip(valid_indices, valid_ids, outputs):
            text = out.outputs[0].text.strip().upper()
            if not text:
                n_empty_generated += 1
            completions[idx] = SampleCompletion(id=sid, completion=text)

    for _, sid, _ in skipped:
        completions[sample_ids.index(sid)] = SampleCompletion(id=sid, completion="")

    logger.info(
        "MCQ summary: %d total, %d generated (%d empty), %d skipped (context)",
        len(samples),
        n_generated,
        n_empty_generated,
        len(skipped),
    )

    return completions


def generate_completions(
    *,
    eval_id: str,
    dataset: str,
    model: str,
    output_dir: pathlib.Path,
    hf_cache: pathlib.Path | None = None,
    gpu_memory_utilization: float = 0.9,
    shard: int | None = None,
    num_shards: int | None = None,
) -> None:
    """Load data, run inference, and write per-sample completions for one eval."""

    if (shard is None) != (num_shards is None):
        raise ValueError("shard and num_shards must be used together")
    if num_shards and not (0 <= shard < num_shards):
        raise ValueError(f"shard must be in [0, {num_shards})")

    gen = get_generation_params(eval_id)

    public = dataset_catalog.load_prompt_samples(eval_id, dataset)

    llm, tokenizer = load_model(
        model=model,
        max_model_len=gen.max_tokens * 3,
        hf_cache=hf_cache,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    model_slug = model.replace("/", "--")
    model_output_dir = output_dir / eval_id / model_slug / dataset

    logger.info("=" * 60)
    logger.info("Eval: %s", eval_id)
    logger.info("Model: %s", model)
    if num_shards:
        logger.info("Shard: %d/%d", shard, num_shards)
    logger.info("Output: %s", model_output_dir)
    logger.info("=" * 60)

    if gen.mcq_choices is not None:
        completions = run_mcq(
            gen=gen,
            samples=public,
            llm=llm,
            tokenizer=tokenizer,
            shard=shard,
            num_shards=num_shards,
        )
    else:
        completions = run_text(
            gen=gen,
            samples=public,
            llm=llm,
            tokenizer=tokenizer,
            shard=shard,
            num_shards=num_shards,
        )

    write_completions(completions, shard=shard, num_shards=num_shards, output_dir=model_output_dir)
    logger.info("Done. Completions in %s/", model_output_dir)


def run_text(
    *,
    gen: GenerationParams,
    samples: list[EvalSample],
    llm,
    tokenizer,
    shard: int | None = None,
    num_shards: int | None = None,
) -> list[SampleCompletion]:
    from vllm import SamplingParams

    total = len(samples)

    if num_shards:
        samples = _shard_list(samples, shard, num_shards)
        logger.info("%d samples, shard %d/%d -> %d samples", total, shard, num_shards, len(samples))
    else:
        logger.info("%d samples", len(samples))

    sampling_params = SamplingParams(max_tokens=gen.max_tokens, temperature=gen.temperature)

    prompts = [
        tokenizer.apply_chat_template([m.model_dump() for m in s.messages], tokenize=False, add_generation_prompt=True)
        for s in samples
    ]
    sample_ids = [s.id for s in samples]

    max_model_len = llm.llm_engine.model_config.max_model_len
    valid_prompts, valid_ids, valid_indices, skipped = _filter_by_token_length(
        prompts,
        sample_ids,
        tokenizer,
        max_model_len,
        max_output_tokens=gen.max_tokens,
    )

    completions: list[SampleCompletion] = [None] * len(samples)

    n_generated = 0
    n_empty_generated = 0
    if valid_prompts:
        t0 = time.time()
        outputs = llm.generate(valid_prompts, sampling_params)
        elapsed = time.time() - t0
        n_generated = len(outputs)
        logger.info("Generated %d completions in %.1fs (%.1f samples/s)", n_generated, elapsed, n_generated / elapsed)
        for idx, sid, out in zip(valid_indices, valid_ids, outputs):
            text = out.outputs[0].text
            if not text.strip():
                n_empty_generated += 1
            completions[idx] = SampleCompletion(id=sid, completion=text)

    for _, sid, _ in skipped:
        completions[sample_ids.index(sid)] = SampleCompletion(id=sid, completion="")

    logger.info(
        "Text summary: %d total, %d generated (%d empty), %d skipped (context)",
        len(samples),
        n_generated,
        n_empty_generated,
        len(skipped),
    )

    return completions
