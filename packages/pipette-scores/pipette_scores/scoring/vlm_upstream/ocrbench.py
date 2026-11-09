"""OCRBench matching + aggregation, ported from VLMEvalKit.

Ported from VLMEvalKit @ 8eca8e5, ``vlmeval/dataset/utils/ocrbench.py``
(``OCRBench_eval``), reshaped from a pandas loop into per-sample matching +
a pure aggregation.

The branch keys off the VLMEvalKit **category** string. For
'Handwritten Mathematical Expression Recognition' the substring match is
CASE-SENSITIVE with all whitespace removed; every other category lowercases
both sides (whitespace kept, newlines flattened).

The conventional headline is 'Final Score' out of 1000 (OCRBench has exactly
1000 questions, sum of correct); upstream also reports Final Score Norm =
score / 10 (a 0-100 scale).
"""

import ast
from typing import Any

OCRBENCH_CATEGORIES = [
    "Regular Text Recognition",
    "Irregular Text Recognition",
    "Artistic Text Recognition",
    "Handwriting Recognition",
    "Digit String Recognition",
    "Non-Semantic Text Recognition",
    "Scene Text-centric VQA",
    "Doc-oriented VQA",
    "Key Information Extraction",
    "Handwritten Mathematical Expression Recognition",
]

_TEXT_RECOGNITION_CATEGORIES = OCRBENCH_CATEGORIES[:6]

HME_CATEGORY = "Handwritten Mathematical Expression Recognition"


def parse_answers(answer: Any) -> list[str]:
    """Upstream stores the answer cell as a stringified list and ``eval``s it."""
    if isinstance(answer, (list, tuple)):
        return [str(a) for a in answer]
    s = str(answer)
    if s.strip().startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(a) for a in parsed]
        except (ValueError, SyntaxError):
            pass
    return [s]


def ocrbench_match(prediction: str, answers: list[str], category: str) -> bool:
    """Per-line matching loop from ``OCRBench_eval``."""
    predict = str(prediction)
    if category == HME_CATEGORY:
        # CASE-SENSITIVE, whitespace stripped from both sides.
        for ans in answers:
            answer = ans.strip().replace("\n", " ").replace(" ", "")
            pred = predict.strip().replace("\n", " ").replace(" ", "")
            if answer in pred:
                return True
    else:
        for ans in answers:
            answer = ans.lower().strip().replace("\n", " ")
            pred = predict.lower().strip().replace("\n", " ")
            if answer in pred:
                return True
    return False


def ocrbench_final_score(per_category_correct: dict[str, int]) -> dict[str, float]:
    """Aggregation from ``OCRBench_eval``: Final Score = sum correct (max 1000)."""
    get = per_category_correct.get
    ret: dict[str, float] = {}
    ret["Text Recognition"] = float(sum(get(c, 0) for c in _TEXT_RECOGNITION_CATEGORIES))
    ret["Scene Text-centric VQA"] = float(get("Scene Text-centric VQA", 0))
    ret["Doc-oriented VQA"] = float(get("Doc-oriented VQA", 0))
    ret["Key Information Extraction"] = float(get("Key Information Extraction", 0))
    ret[HME_CATEGORY] = float(get(HME_CATEGORY, 0))
    ret["Final Score"] = (
        ret["Text Recognition"]
        + ret["Scene Text-centric VQA"]
        + ret["Doc-oriented VQA"]
        + ret["Key Information Extraction"]
        + ret[HME_CATEGORY]
    )
    ret["Final Score Norm"] = ret["Final Score"] / 10
    return ret
