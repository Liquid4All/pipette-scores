"""The answer a model family's VLMEvalKit wrapper cuts out of a reply before the
benchmark's own scoring sees it, for the prompt profiles whose wrapper does.

VLMEvalKit's MiniCPM-V wrappers prompt step by step to an ``Answer:`` line (the
``minicpm_v_cot`` profile) and then reduce the reply to that answer: MiniCPM-V
4.6's ``extract_answer`` (vlmeval/vlm/minicpm_v.py:1504-1529 at 8eca8e5),
ported here line for line. MiniCPM-V 4.5's (:971-999) reads the last
``Answer:`` letter of the whole reply, which this rule also reads, since it
cuts at a closing ``</think>`` only when there is one. A reply that ends
inside an unclosed reasoning block therefore still gives its stated answer, as
it does under VLMEvalKit; without this the reasoning rules would drop the whole
reply, answer included.

Kimi-VL's wrapper (``KimiVL.generate_inner``, vlmeval/vlm/kimi_vl.py:140-143
at 8eca8e5) keeps the text after the first ``◁/think▷`` (nothing at all when
``◁think▷`` opened and never closed) and, on MMMU and MMStar, the content of the
last ``\\boxed{}`` in it. The first part is what the reasoning rules do for every
family; the boxed part is ported below.

Every other profile's wrapper leaves the reply as it is, and so does this
module. OCRBench is left whole on every profile, as upstream leaves it. The
reasoning of the other families is cut by the scorers' thinking-tag rules
(scoring.completion_text), not here.
"""

import re

from pipette_scores.types import EvalId, SampleCompletion

# DATASET_TYPE upstream: the multiple-choice benchmarks, and the yes/no ones.
_MCQ = frozenset({EvalId.MMSTAR, EvalId.REALWORLDQA, EvalId.MMBENCH_EN_DEV, EvalId.MMMU_VAL})
_YES_NO = frozenset({EvalId.MME, EvalId.POPE})

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_MCQ_ANSWER = re.compile(r"(?:Answer(?:\s+is|:)|答案(?:是|为)?[:：]?)\s*([A-Ja-j])(?![A-Za-z])", re.DOTALL)
_TAIL_ANSWER = re.compile(r"Answer:\s*(.*)\s*$", re.DOTALL)


def minicpm_v_cot(benchmark: EvalId, reply: str) -> str:
    """MiniCPM-V-4.6's ``extract_answer`` for a benchmark it prompts with CoT."""
    if "</think>" in reply:
        reply = reply.rsplit("</think>", 1)[-1].strip()
    if tag := _ANSWER_TAG.search(reply):
        return tag.group(1).strip()
    if benchmark in _MCQ:
        if matches := _MCQ_ANSWER.findall(reply):
            return matches[-1].strip()
    elif benchmark in _YES_NO:
        if match := _TAIL_ANSWER.search(reply):
            return match.group(1)
    return reply


def _last_boxed(answer: str) -> str:
    """Kimi's ``extract_boxed_content``: the content of the last ``\\boxed{``,
    braces balanced; the text itself when it has none or its braces do not
    balance."""
    start = answer.rfind("\\boxed{")
    if start == -1:
        return answer
    start += len("\\boxed{")
    depth = 1
    end = start
    while end < len(answer):
        if answer[end] == "{":
            depth += 1
        elif answer[end] == "}":
            depth -= 1
            if depth == 0:
                break
        end += 1
    return answer[start:end] if depth == 0 else answer


def kimi_vl(benchmark: EvalId, reply: str) -> str:
    """Kimi-VL-A3B-Thinking-2506's reply with, on MMMU and MMStar, the summary
    after its thought cut down to the boxed answer in it, as its VLMEvalKit wrapper
    reduces it (``extract_summary`` takes the text after the first ``◁/think▷``,
    then ``extract_boxed_content`` the last ``\\boxed{}``).

    The thought stays in the reply, closed, so the reasoning rules still see it
    and read the rest as they do for every family; text before its opener is
    dropped, since the summary is only what follows the close. A thought that
    never closes has no summary, and the reply is left for the rules to cut."""
    from pipette_scores.dataset_catalog.prompt_profiles import KIMI_VL_BENCHMARKS

    if "◁/think▷" not in reply:
        if benchmark not in KIMI_VL_BENCHMARKS:
            return reply
        return reply if "◁think▷" in reply else _last_boxed(reply)
    head, _, summary = reply.partition("◁/think▷")
    if "◁think▷" in head:
        head = head[head.index("◁think▷") :]
    if benchmark in KIMI_VL_BENCHMARKS:
        summary = _last_boxed(summary.strip())
    return f"{head}◁/think▷{summary}"


# Profile -> the wrapper's extraction. A profile not here leaves replies whole.
EXTRACTIONS = {
    "minicpm_v_cot": minicpm_v_cot,
    "kimi_vl": kimi_vl,
}


def apply(prompt_profile: str | None, benchmark: str, completions: list[SampleCompletion]) -> list[SampleCompletion]:
    """The completions as the profile's wrapper hands them to the benchmark's
    scoring: each reply reduced to its answer where the wrapper does that."""
    extract = EXTRACTIONS.get(prompt_profile or "")
    if extract is None:
        return completions
    try:
        eval_id = EvalId(benchmark)
    except ValueError:
        return completions
    if eval_id == EvalId.OCRBENCH:
        return completions
    return [c.model_copy(update={"completion": extract(eval_id, c.completion)}) for c in completions]
