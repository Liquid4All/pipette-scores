"""Deterministic MCQ answer matching, ported from VLMEvalKit.

Ported from VLMEvalKit @ 8eca8e5, ``vlmeval/utils/matching_util.py``
(functions ``can_infer_option``, ``can_infer_text``, ``can_infer``).

Deviations from upstream (behavior-preserving):
- Logging / ``VERBOSE`` env handling removed. Upstream's "A might be a
  quantifier" branch only fires when ``VERBOSE`` is set (default off), so it
  is dropped here to match upstream's default behavior.
- ``can_infer_text`` operates on a copy of ``choices`` instead of mutating
  the caller's dict (upstream lowercases values in place).

No LLM judge is used anywhere: when ``can_infer`` returns ``False`` the
caller must count the sample as WRONG (deterministic equivalent of
VLMEvalKit's ``exact_matching`` policy, which assigns option ``Z``).
"""

import copy as cp
import re
import string

# Matches verbose chain-of-thought answers such as:
#   "The correct answer is **B**. Here's why: ..."
#   "The answer is C."
_VERBOSE_ANSWER_RE = re.compile(r"(?i)(?:correct\s+)?answer\s+is\s+\**([ABCD])\**")


def can_infer_option(answer, choices):
    # Choices is a dictionary
    if "Failed to obtain answer via API" in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        "Cannot determine the answer",
    ]
    for err in reject_to_answer:
        if err in answer:
            return "Z"

    def count_choice(splits, choices, prefix="", suffix=""):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    chars = ".()[],:;!*#{}"
    for c in chars:
        answer_mod = answer_mod.replace(c, " ")

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if ch in splits and splits.index(ch) > (len(splits) - 5):
                return ch
    elif count == 0 and count_choice(splits, {"Z", ""}) == 1:
        return "Z"

    match = _VERBOSE_ANSWER_RE.search(answer or "")
    if match and match.group(1).upper() in choices:
        return match.group(1).upper()

    return False


def can_infer_text(answer, choices):
    answer = answer.lower()
    choices = {k: str(v).lower() for k, v in choices.items()}
    if len(answer) > 2 * sum(len(v) for v in choices.values()):
        return False
    for k in choices:
        assert k in string.ascii_uppercase
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)
