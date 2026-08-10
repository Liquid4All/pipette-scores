import math
from collections import defaultdict
from typing import Any

from pipette_scores.repeats import logical_id
from pipette_scores.scoring.completion_text import remove_thinking_tags

from pipette_scores.scoring.ifbench.upstream import (
    InputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)

from pipette_scores.types import IFBenchSample, SampleCompletion, SampleScore


def score(
    completions: list[SampleCompletion], samples: list[IFBenchSample]
) -> tuple[list[SampleScore], dict[str, Any]]:
    """Score completions with upstream IFBench's strict and loose checkers.

    Mirrors upstream ``run_eval.py``: every response is scored under both
    ``test_instruction_following_strict`` and ``test_instruction_following_loose``.
    The strict pass runs first on the shared ``InputExample`` — it strips
    ``None`` kwargs in place — so the loose pass then sees the same stripped
    kwargs, exactly as upstream's two-pass loop does.

    The paper reports **prompt-level loose accuracy** as the headline metric, so
    each sample's ``is_correct`` reflects the loose pass; both strict and loose
    accuracies are emitted in the context for reference. The context mirrors
    upstream ``print_report``: prompt-level and instruction-level aggregates,
    tier0 accuracy per category prefix (``strict/<category>``), and tier1
    accuracy per full instruction id (``strict/<category>:<name>``) — each for
    both the strict and loose passes.

    Reasoning chains are removed first via ``remove_thinking_tags``. Upstream
    strips these during generation; because this scorer ingests raw completions,
    it does the equivalent cleanup here before applying the unmodified checkers.

    When the dataset is run with repeats (``#k`` salted ids), the context also
    carries ``ifbench_pass_at_1`` — the loose prompt-level pass rate aggregated
    over the N·R sample rows, with the denominator taken from ``samples`` so a
    missing attempt counts as a fail — plus its binomial ``_stderr`` and a
    ``_per_sample`` breakdown keyed by logical id. ``repeats: 1`` (no ``#k``
    suffix) is transparent: each id is its own logical id.
    """
    by_id = {s.id: s for s in samples}
    scores: list[SampleScore] = []

    strict_prompt_correct = 0
    strict_inst_correct = 0
    loose_prompt_correct = 0
    loose_inst_correct = 0
    inst_total = 0

    strict_iid_passed: dict[str, int] = defaultdict(int)
    loose_iid_passed: dict[str, int] = defaultdict(int)
    per_iid_total: dict[str, int] = defaultdict(int)

    for c in completions:
        sample = by_id[c.id]
        text = remove_thinking_tags(c.completion)

        inp = InputExample(
            key=sample.key,
            instruction_id_list=list(sample.instruction_id_list),
            prompt=sample.prompt,
            kwargs=[dict(k) for k in sample.kwargs],
        )
        # Strict first: it filters None kwargs on `inp` in place, so the loose
        # pass below operates on the same stripped kwargs (upstream ordering).
        strict_out = test_instruction_following_strict(inp, {sample.prompt: text})
        loose_out = test_instruction_following_loose(inp, {sample.prompt: text})

        scores.append(SampleScore(id=c.id, is_correct=loose_out.follow_all_instructions))

        if strict_out.follow_all_instructions:
            strict_prompt_correct += 1
        if loose_out.follow_all_instructions:
            loose_prompt_correct += 1
        strict_inst_correct += sum(strict_out.follow_instruction_list)
        loose_inst_correct += sum(loose_out.follow_instruction_list)
        inst_total += len(strict_out.follow_instruction_list)

        for iid, strict_passed, loose_passed in zip(
            strict_out.instruction_id_list,
            strict_out.follow_instruction_list,
            loose_out.follow_instruction_list,
        ):
            per_iid_total[iid] += 1
            if strict_passed:
                strict_iid_passed[iid] += 1
            if loose_passed:
                loose_iid_passed[iid] += 1

    total = len(scores)
    context: dict[str, Any] = {
        "strict_prompt_accuracy": strict_prompt_correct / total if total else 0.0,
        "strict_instruction_accuracy": strict_inst_correct / inst_total if inst_total else 0.0,
        "loose_prompt_accuracy": loose_prompt_correct / total if total else 0.0,
        "loose_instruction_accuracy": loose_inst_correct / inst_total if inst_total else 0.0,
    }
    # Repeat-aware pass@1 (loose, prompt-level). Repeats are realized as #k salted
    # ids (see repeats.expand); group scored ids back by logical id. The
    # denominator is the SAMPLE count (N logical samples × R repeats) taken from
    # `samples`, not from received completions, so a missing attempt scores as a
    # fail — matching liquid_evals' fixed total_attempts and management's
    # "missing = incorrect". With one completion per #k id this equals the
    # dashboard's correct/total over the boolean rows.
    n_samples = len(samples)
    per_sample_total: dict[str, int] = defaultdict(int)
    for s in samples:
        per_sample_total[logical_id(s.id)] += 1
    per_sample_correct: dict[str, int] = defaultdict(int)
    for sc in scores:
        if sc.is_correct:
            per_sample_correct[logical_id(sc.id)] += 1
    pass_at_1 = loose_prompt_correct / n_samples if n_samples else 0.0
    context["ifbench_pass_at_1"] = pass_at_1
    context["ifbench_pass_at_1_stderr"] = math.sqrt(pass_at_1 * (1.0 - pass_at_1) / n_samples) if n_samples else 0.0
    context["ifbench_pass_at_1_per_sample"] = {
        lid: per_sample_correct[lid] / per_sample_total[lid] for lid in sorted(per_sample_total)
    }
    # tier0 derives from tier1: each instruction id belongs to exactly one
    # category (the prefix before ":"), so category totals are just the sum
    # over their members. Keys have no ":", e.g. "strict/count".
    strict_cat_passed: dict[str, int] = defaultdict(int)
    loose_cat_passed: dict[str, int] = defaultdict(int)
    per_cat_total: dict[str, int] = defaultdict(int)
    for iid in per_iid_total:
        cat = iid.split(":")[0]
        per_cat_total[cat] += per_iid_total[iid]
        strict_cat_passed[cat] += strict_iid_passed[iid]
        loose_cat_passed[cat] += loose_iid_passed[iid]
    for cat in sorted(per_cat_total):
        context[f"strict/{cat}"] = strict_cat_passed[cat] / per_cat_total[cat]
        context[f"loose/{cat}"] = loose_cat_passed[cat] / per_cat_total[cat]
    # tier1: per full instruction id (keys contain ":", e.g.
    # "strict/count:keywords_multiple").
    for iid in sorted(per_iid_total):
        context[f"strict/{iid}"] = strict_iid_passed[iid] / per_iid_total[iid]
        context[f"loose/{iid}"] = loose_iid_passed[iid] / per_iid_total[iid]
    return scores, context
