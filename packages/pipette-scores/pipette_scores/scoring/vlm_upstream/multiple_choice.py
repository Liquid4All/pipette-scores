"""MCQ scoring helpers, ported from VLMEvalKit.

Ported from VLMEvalKit @ 8eca8e5, ``vlmeval/dataset/utils/multiple_choice.py``:
- ``build_choices``   — collects the non-empty option columns A..Z of a row.
- ``mmmu_preproc_*``  — port of ``MMMU_preproc`` + the ``answer_map``
                        rewrite in ``mcq_vanilla_eval``: open questions are
                        re-formulated as 2-choice MCQ (A = gold answer,
                        B = 'Other Answers'), GT letter becomes 'A'.
- ``mcq_hit``         — port of the ``eval_vanilla`` exact-matching path:
                        ``can_infer`` the prediction; unparseable counts as
                        WRONG (upstream assigns 'Z' under ``exact_matching``).

Headline metric ("Overall" in upstream ``report_acc``) is the MICRO mean of
per-sample hits — not a macro average over categories.
"""

import string
from typing import Any

from .matching_util import can_infer

# 'Other Answers' literal from upstream MMMU_preproc.
MMMU_OTHER_ANSWERS = "Other Answers"


def _is_missing(v: Any) -> bool:
    """Equivalent of pandas ``isna`` for raw_doc values (+ empty strings,
    which a parquet/dict round-trip may produce instead of NaN)."""
    if v is None:
        return True
    if isinstance(v, float) and v != v:  # NaN
        return True
    s = str(v).strip()
    return s == "" or s.lower() == "nan"


def build_choices(item: dict) -> dict[str, str]:
    ret = {}
    for ch in string.ascii_uppercase:
        if ch in item and not _is_missing(item[ch]):
            ret[ch] = str(item[ch])
    return ret


def mmmu_preproc_choices(item: dict) -> dict[str, str]:
    """MMMU_preproc: open questions (no 'A' option) become a 2-choice MCQ."""
    choices = build_choices(item)
    if "A" not in choices:
        return {"A": str(item.get("answer", "")), "B": MMMU_OTHER_ANSWERS}
    return choices


def mmmu_preproc_gt(answer: Any) -> str:
    """GT letter after MMMU_preproc: non-letter (open) answers map to 'A'."""
    a = str(answer)
    return a if a in list(string.ascii_uppercase) else "A"


def mcq_hit(prediction: str, choices: dict[str, str], gt: str) -> bool:
    """Deterministic vanilla MCQ scoring: True iff ``can_infer`` extracts the
    ground-truth letter. No judge, no random fallback — unparseable = wrong."""
    ret = can_infer(prediction, choices)
    return bool(ret) and ret == str(gt).strip()
