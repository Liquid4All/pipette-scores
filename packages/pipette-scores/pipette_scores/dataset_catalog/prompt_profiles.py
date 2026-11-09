"""Model-specific prompts, copied from VLMEvalKit or a model card.

By default every model gets the same prompt for a benchmark: the dataset's own
VLMEvalKit prompt, stored with the dataset. A prompt profile replaces that text
with the prompt one model family is evaluated with — the one VLMEvalKit's
wrapper builds for it, or the one its model card asks for — so a run can
reproduce the numbers that family publishes. Only the text changes: the images
and where they go, the dataset and the scoring stay the same.

A profile may also set a system message (SYSTEM_PROMPTS). That is how a family
whose thinking is switched on by a prompt rather than by its chat template is
run with thinking: InternVL3.5 thinks only under its R1 system prompt, and
Nemotron-Nano-v2-VL thinks when a system message says ``/think``.

A profile may also end the conversation with an assistant message (PREFILLS),
which the server continues instead of starting a new turn. That is how a family
whose template has no thinking switch at all is forced to think: Phi-4-reasoning-vision
thinks when its reply is opened with ``<think>``.

Every string here is copied either from VLMEvalKit at commit 8eca8e5, from the
file and lines named in each profile (paths are under vlmeval/vlm/), or from
the model card a profile names. To add one, write a function from (benchmark,
the dataset row's raw document, the default prompt) to the prompt text, and add
it to PROFILES.
"""

import math
import re
import string
from collections.abc import Callable

from pipette_scores.types import EvalId

Profile = Callable[[EvalId, dict, str], str]

MULTIPLE_CHOICE = frozenset({EvalId.MMSTAR, EvalId.REALWORLDQA, EvalId.MMBENCH_EN_DEV, EvalId.MMMU_VAL})
# The benchmarks every profile covers: the VLM evals.
BENCHMARKS = MULTIPLE_CHOICE | {EvalId.MME, EvalId.POPE, EvalId.OCRBENCH}

SHORT_ANSWER = "Answer the question using a single word or phrase."


def _present(value) -> bool:
    """Upstream skips cells pandas reads as missing; the rows store them as null or NaN."""
    return value is not None and value != "" and not (isinstance(value, float) and math.isnan(value))


def _hint(doc: dict) -> str | None:
    return doc["hint"] if _present(doc.get("hint")) else None


def _options(doc: dict) -> dict[str, str]:
    return {letter: doc[letter] for letter in string.ascii_uppercase if _present(doc.get(letter))}


def _letter_choice(doc: dict) -> str:
    """A question answered with an option letter: internvl/utils.py:161-182,
    which granite_vision/granite_vision.py:113-139 copies."""
    question = doc["question"]
    if hint := _hint(doc):
        question = f"{hint}\n{question}"
    options = _options(doc)
    for letter, option in options.items():
        question += f"\n{letter}. {option}"
    chinese = re.search(r"[\u4e00-\u9fff]", question) is not None
    if options:
        return question + (
            "\n请直接回答选项字母。"
            if chinese
            else "\nAnswer with the option's letter from the given choices directly."
        )
    return question + ("\n请直接回答问题。" if chinese else "\nAnswer the question directly.")


def internvl(benchmark: EvalId, doc: dict, default: str) -> str:
    """InternVL3.5: internvl/internvl_chat.py:245-270."""
    if benchmark in MULTIPLE_CHOICE:
        return _letter_choice(doc)
    if benchmark == EvalId.MME:
        return f"{doc['question']} {SHORT_ANSWER}"
    if benchmark == EvalId.OCRBENCH:
        return f"{doc['question']}\n{SHORT_ANSWER}"
    return doc["question"]  # POPE


def granite_vision(benchmark: EvalId, doc: dict, default: str) -> str:
    """IBM's Granite Vision wrapper, granite_vision/granite_vision.py:82-139.
    Written for Granite Vision 3.x; the closest upstream recipe for 4.1."""
    if benchmark in MULTIPLE_CHOICE:
        return _letter_choice(doc)
    if benchmark == EvalId.OCRBENCH:
        return doc["question"] + "\nReply with only one word or a short phrase or a full address."
    return default  # MME, POPE


# minicpm_v.py:820-830
MINICPM_LETTER = "\nAnswer with the option's letter from the given choices directly."
MINICPM_NO_OPTIONS = "Carefully read the following question. Answer the question directly."
MINICPM_CHOICE_COT = (
    "Carefully read the following multichoice question, solve it step by step and finally pick the option "
    'associated with the correct answer in the format of "Answer: selected option\n\n'
)
MINICPM_SHORT_COT = (
    "Read the following question carefully, solve it step by step, and then output the final answer in the "
    'format of "Answer: single number or single word or phrase".\n\n'
)
MINICPM_OCR_COT = "Carefully observe the image and answer the OCR-related questions below. \n\n"


def minicpm_v(benchmark: EvalId, doc: dict, default: str) -> str:
    """MiniCPM-V answering directly: the prompts VLMEvalKit gives it on the
    benchmarks it runs without step-by-step reasoning, minicpm_v.py:1110-1149."""
    if benchmark in MULTIPLE_CHOICE:
        prompt = f"Hint: {hint}\n" if (hint := _hint(doc)) else ""
        prompt += f"Question: {doc['question']}\n"
        if options := _options(doc):
            return prompt + "Options:\n" + "".join(f"{k}. {v}\n" for k, v in options.items()) + MINICPM_LETTER
        # Only MMMU has questions without options; upstream puts this line first there.
        return f"{MINICPM_NO_OPTIONS}\n{prompt}"
    if benchmark == EvalId.OCRBENCH:
        # Upstream sends this instruction as a text item before the image.
        return f"{SHORT_ANSWER}\n{doc['question']}"
    return doc["question"]  # MME, POPE


def minicpm_v_cot(benchmark: EvalId, doc: dict, default: str) -> str:
    """MiniCPM-V reasoning step by step to an "Answer: ..." line: how VLMEvalKit
    prompts MiniCPM-V-4.6-Thinking on every benchmark and MiniCPM-V-4.5 on MMMU,
    MMBench, MMStar and OCRBench, minicpm_v.py:1150-1189."""
    if benchmark in MULTIPLE_CHOICE:
        prompt = f"Hint: {hint}\n" if (hint := _hint(doc)) else ""
        prompt += f"{doc['question']}\n"
        if options := _options(doc):
            return MINICPM_CHOICE_COT + prompt + "".join(f"{k}. {v}\n" for k, v in options.items())
        return MINICPM_SHORT_COT + prompt
    if benchmark == EvalId.OCRBENCH:
        return MINICPM_OCR_COT + doc["question"]
    return MINICPM_SHORT_COT + doc["question"]  # MME, POPE


# liquid.py:13-48
LFM_DIRECT_ANSWER = "\nPlease answer directly with only the final answer, do not give any explanation."
_YES_NO_WORD = "\nAnswer with a single word: Yes or No."
_YES_NO_ONLY = "\nOnly output Yes or No."
LFM_MME_BY_CATEGORY = {
    "ocr": "{question}",
    "artwork": "{question}" + _YES_NO_WORD,
    "celebrity": "This is a face recognition question about the person shown. Decide whether the name in the "
    "question matches the image. {question}" + _YES_NO_WORD,
    "color": "Check the visible color of the relevant object in the image. {question}" + LFM_DIRECT_ANSWER,
    "count": "{question}" + _YES_NO_ONLY,
    "existence": "{question}",
    "landmark": "{question}" + LFM_DIRECT_ANSWER,
    "position": "{question}" + _YES_NO_ONLY,
    "posters": "Look carefully at the image. {question}" + _YES_NO_WORD,
    "scene": "Look carefully at the image. {question}" + _YES_NO_WORD,
    "code_reasoning": "{question}" + _YES_NO_WORD,
    "commonsense_reasoning": "Use the image and common sense. {question}" + _YES_NO_WORD,
    "numerical_calculation": "Calculate carefully from the visible information. {question}" + _YES_NO_WORD,
    "text_translation": "{question}" + LFM_DIRECT_ANSWER,
}


def lfm_vl(benchmark: EvalId, doc: dict, default: str) -> str:
    """LFM2-VL and LFM2.5-VL-1.6B: the dataset's prompt, then LFM's direct-answer
    instruction, on every benchmark (liquid.py:224-228, 266-268)."""
    return default + LFM_DIRECT_ANSWER


def lfm2_5_vl_3b(benchmark: EvalId, doc: dict, default: str) -> str:
    """LFM2.5-VL-3B: liquid.py:129-133, 152-157, 224-232."""
    if benchmark == EvalId.MME:
        return LFM_MME_BY_CATEGORY[str(doc["category"]).lower()].format(question=doc["question"])
    if benchmark == EvalId.OCRBENCH:
        return default + "\nPlease answer concisely with short words or phrases when possible."
    if benchmark in (EvalId.MMMU_VAL, EvalId.POPE):
        return default
    return default + LFM_DIRECT_ANSWER  # MMStar, RealWorldQA, MMBench


# The Qwen3.5 model cards' "Best Practices" for benchmarking, item 3
# ("Standardize Output Format"), multiple-choice questions.
QWEN3_5_CHOICE = 'Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`.'


def qwen3_5(benchmark: EvalId, doc: dict, default: str) -> str:
    """Qwen3.5: the dataset's prompt, then the output format Qwen's model card
    asks for on multiple-choice questions. VLMEvalKit runs Qwen3.5 on its
    dataset prompts, so the other benchmarks keep them."""
    if benchmark in MULTIPLE_CHOICE and _options(doc):
        return f"{default.rstrip()}\n{QWEN3_5_CHOICE}"
    return default


# GLM-4.6V's VLMEvalKit wrapper, GLMThinking (cogvlm.py:55-242), has no custom prompt, so the
# dataset prompts stand, but it adds this as a text item of its own after POPE's
# question (cogvlm.py:131-132, 162-163).
GLM_V_YES_NO = "Please answer yes or no."


def glm_v(benchmark: EvalId, doc: dict, default: str) -> str:
    """GLM-4.6V-Flash: the dataset's prompt, and on POPE VLMEvalKit's extra
    yes/no line. The wrapper sends it as a second text item, which GLM's chat
    template renders on a line of its own, so one string with a newline before
    the line is the same prompt."""
    if benchmark == EvalId.POPE:
        return f"{default}\n{GLM_V_YES_NO}"
    return default


# Kimi-VL's VLMEvalKit wrapper, KimiVL.message_to_promptimg (kimi_vl.py:92-116), on MMMU and MMStar: it
# drops the dataset's closing line and appends this to every text item.
# The benchmarks the wrapper changes the prompt on, and reduces the reply to its boxed answer on.
KIMI_VL_BENCHMARKS = frozenset({EvalId.MMSTAR, EvalId.MMMU_VAL})
KIMI_VL_DROPPED = "Please select the correct answer from the options above. \n"
KIMI_VL_BOXED_COT = (
    "Answer the preceding question. The last line of your response should follow this format: "
    "'Answer: $\\boxed{LETTER}$' (without quotes), where LETTER is one of the options. "
    "Think step by step logically, considering all relevant information before answering."
)


def kimi_vl(benchmark: EvalId, doc: dict, default: str) -> str:
    """Kimi-VL-A3B-Thinking-2506: on MMMU and MMStar the dataset's closing
    "Please select..." line is replaced by Kimi's boxed step-by-step instruction,
    kimi_vl.py:92-116; the other benchmarks keep the dataset prompts.

    VLMEvalKit appends the instruction to each text item. MMMU's question reaches
    it split at its ``<image N>`` placeholders (split_MMMU), so every piece of
    text carries the instruction, an empty one included; the same split is made
    here, and the loader puts the images where the placeholders were. MMStar's
    prompt is one text item."""
    if benchmark not in KIMI_VL_BENCHMARKS:
        return default
    default = default.replace(KIMI_VL_DROPPED, "")
    if benchmark == EvalId.MMSTAR:
        return default + KIMI_VL_BOXED_COT
    head, *pieces = default.split("<image ")
    # A piece is "N>" and the text after it; one not shaped so is left for the
    # loader's own fallback, which sends the prompt whole.
    if not all(len(piece) >= 2 and piece[0].isdigit() and piece[1] == ">" for piece in pieces):
        return default + KIMI_VL_BOXED_COT
    return "".join([head + KIMI_VL_BOXED_COT, *(f"<image {piece}{KIMI_VL_BOXED_COT}" for piece in pieces)])


def nemotron_think(benchmark: EvalId, doc: dict, default: str) -> str:
    """Nemotron-Nano-12B-v2-VL: the dataset's prompt (VLMEvalKit has no wrapper
    for the model, so the dataset prompts stand), starting with a newline. The
    model's chat template writes ``<image>`` and a newline before the question;
    the image goes in a content part of its own, so the newline is the first
    character of the text part. On MMMU the loader puts the images at their
    ``<image N>`` placeholders, so the newline opens the text before the first image."""
    return "\n" + default


def dataset_prompt(benchmark: EvalId, doc: dict, default: str) -> str:
    """The dataset's prompts, for a profile that changes only the messages around them.

    Phi-4-reasoning-vision-15B: Microsoft ran the VLMEvalKit benchmarks "without any custom
    user-prompting" (technical report, section 6), under the model's own system prompt and chat
    template. Thinking is forced by PREFILLS.

    MiMo-VL-7B-RL-2508: what VLMEvalKit's Qwen2VLChat sends it (qwen2_vl/prompt.py:26-31,
    use_custom_prompt is False); thinking is on by default, so there is nothing to switch. The
    profile adds the system message and, as a REASONING_OPENERS entry, sets ``opens_reasoning``."""
    return default


# InternVL3.5's thinking mode: no chat-template switch, only this system
# prompt (internvl/internvl_chat.py:25-38, used by the -Thinking entries).
R1_SYSTEM_PROMPT = (
    "You are an AI assistant that rigorously follows this response protocol:\n\n"
    "1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and "
    "reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> "
    "tags.\n\n"
    "2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate "
    "the answer from the think section with a newline.\n\n"
    "Ensure that the thinking process is thorough but remains focused on the query. The final answer should be "
    "standalone and not reference the thinking section."
)

# Nemotron-Nano-12B-v2-VL's thinking switch: a ``/think`` flag in a system or user message, which the
# chat template strips from the text (the card's examples send ``/no_think`` as a system message, and
# the enable_thinking kwarg is not read).
NEMOTRON_THINK_FLAG = "/think"

# MiMo-VL's system prompt as VLMEvalKit's lmdeploy path gives it (the qwen2d5-vl template's
# default, lmdeploy v0.9.2 model.py:1076-1080) and as Xiaomi's own eval harness does (lmms-eval
# mimo_vl_eval branch, models/mivllm.py:117-123). Without a system message the model's chat
# template sends "You are MiMo, an AI assistant developed by Xiaomi.".
MIMO_VL_SYSTEM_PROMPT = "You are a helpful assistant."

# Phi-4-reasoning-vision's forced-thinking switch, model card section 2.4: end the generation prompt
# with ``<think>`` (``<nothink>`` forces the direct answer).
PHI4_RV_THINK_PREFILL = "<think>"

PROFILES: dict[str, Profile] = {
    "internvl": internvl,
    "internvl_r1": internvl,
    "granite_vision": granite_vision,
    "minicpm_v": minicpm_v,
    "minicpm_v_cot": minicpm_v_cot,
    "lfm_vl": lfm_vl,
    "lfm2_5_vl_3b": lfm2_5_vl_3b,
    "qwen3_5": qwen3_5,
    "glm_v": glm_v,
    "kimi_vl": kimi_vl,
    "nemotron_think": nemotron_think,
    "phi4_rv_think": dataset_prompt,
    "mimo_vl": dataset_prompt,
}


# A system message sent before the user turn, for the profiles that have one.
SYSTEM_PROMPTS: dict[str, str] = {
    "internvl_r1": R1_SYSTEM_PROMPT,
    "nemotron_think": NEMOTRON_THINK_FLAG,
    "mimo_vl": MIMO_VL_SYSTEM_PROMPT,
}

# An assistant message that ends the conversation, for the profiles that have one: the server
# continues it, so the reply is what follows it and does not repeat it. This needs a runtime that
# continues a final assistant message (llama-server does by default); one that ends it as a
# finished turn sends the model no opener.
PREFILLS: dict[str, str] = {"phi4_rv_think": PHI4_RV_THINK_PREFILL}

# The profiles whose model reasons from the first token of its reply, so a reply cut before the
# close shows no tag; the VLM scorers' strict view counts such a reply as thinking. internvl_r1 is
# not here: its R1 system prompt has the model write <think> itself, in the reply.
REASONING_OPENERS = frozenset({"nemotron_think", "phi4_rv_think", "kimi_vl", "mimo_vl"})


def profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown prompt profile {name!r}; known: {', '.join(PROFILES)}") from None
