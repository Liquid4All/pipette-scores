"""Scoring facade — dispatches to per-eval score functions."""

import logging
from typing import Any

from pipette_scores.types import ASR_EVAL_IDS, EvalId, SampleScore
from pipette_scores.memory import rss_mb
from pipette_scores.scoring import (
    gpqa_diamond,
    ifbench,
    ifstruct,
    math_500,
    mmbench_en_dev,
    mme,
    mmmu_val,
    mmstar,
    ocrbench,
    pope,
    realworldqa,
)
from pipette_scores.scoring.asr import evaluate as asr

logger = logging.getLogger(__name__)


def list_eval_ids() -> list[EvalId]:
    """Return all registered eval IDs (used by the CLI)."""
    return list(EvalId)


class _ProgressList(list):
    """A list wrapper that logs progress and RSS when iterated."""

    def __init__(self, items: list, label: str, log_every: int = 50):
        super().__init__(items)
        self._label = label
        self._log_every = log_every

    def __iter__(self):
        total = len(self)
        for i, item in enumerate(super().__iter__()):
            if i > 0 and i % self._log_every == 0:
                logger.info(
                    "[%s] %d/%d samples  rss=%.0fMB",
                    self._label,
                    i,
                    total,
                    rss_mb(),
                )
            yield item


def score(
    eval_id: EvalId,
    completions: list[Any],
    samples: list[Any],
    *,
    enable_thinking: bool | None = None,
    label: str | None = None,
    prompt_profile: str | None = None,
) -> tuple[list[SampleScore], dict[str, Any]]:
    """``prompt_profile`` reaches the ASR scorer, which cleans each answer as that
    profile says, and tells the VLM scorers whether it opens the model's reasoning
    (prompt_profiles.REASONING_OPENERS), which sets their status, answer and strict
    view; the text evals score the same either way."""
    if EvalId(eval_id) in ASR_EVAL_IDS:
        scores, context = asr.score(
            completions, samples, enable_thinking=enable_thinking, prompt_profile=prompt_profile
        )
        logger.info(
            "[%s] Done: WER %s over %d units",
            f"{eval_id}/{label}" if label else eval_id,
            "n/a" if context["result"] is None else f"{context['result']:.2f}%",
            context["units_scored"],
        )
        return scores, context
    from pipette_scores.dataset_catalog.prompt_profiles import BENCHMARKS, REASONING_OPENERS

    scorer = {
        EvalId.IFBENCH: ifbench.score,
        EvalId.IFSTRUCT: ifstruct.score,
        EvalId.GPQA_DIAMOND: gpqa_diamond.score,
        EvalId.MATH_500: math_500.score,
        EvalId.MME: mme.score,
        EvalId.MMSTAR: mmstar.score,
        EvalId.OCRBENCH: ocrbench.score,
        EvalId.MMMU_VAL: mmmu_val.score,
        EvalId.REALWORLDQA: realworldqa.score,
        EvalId.MMBENCH_EN_DEV: mmbench_en_dev.score,
        EvalId.POPE: pope.score,
    }[EvalId(eval_id)]
    progress_label = f"{eval_id}/{label}" if label else eval_id
    total = len(completions)
    logger.info("[%s] Scoring %d completions...", progress_label, total)
    logged_completions = _ProgressList(completions, progress_label, log_every=500)
    vlm = {"opens_reasoning": prompt_profile in REASONING_OPENERS} if EvalId(eval_id) in BENCHMARKS else {}
    scores, context = scorer(logged_completions, samples, enable_thinking=enable_thinking, **vlm)
    correct = sum(1 for v in scores if v.is_correct)
    logger.info(
        "[%s] Done: %d/%d correct (%.1f%%)",
        progress_label,
        correct,
        total,
        (correct / total * 100) if total > 0 else 0.0,
    )
    return scores, context
