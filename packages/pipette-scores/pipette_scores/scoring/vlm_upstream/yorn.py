"""Yes/No answer extraction + MME / POPE ratings, ported from VLMEvalKit.

Ported from VLMEvalKit @ 8eca8e5:
- ``process_punctuation``  — ``vlmeval/smp/misc.py``
- ``YOrN_Extraction``      — ``vlmeval/dataset/utils/yorn.py``
- ``mme_rating``           — pure-python port of ``MME_rating``
                             (``vlmeval/dataset/utils/yorn.py``, pandas-free)
- ``pope_rating``          — pure-python port of ``POPE_rating``
                             (``vlmeval/dataset/utils/yorn.py``, pandas-free)

No LLM judge: upstream retries 'Unknown' extractions through a judge model;
here 'Unknown' deterministically stays 'Unknown' and therefore never matches
the 'Yes'/'No' ground truth (counts as wrong / as a "No" prediction in the
POPE confusion matrix, identical to upstream's ``exact_matching`` policy).
"""

import re
from collections import defaultdict


def process_punctuation(inText):
    outText = inText
    punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
    commaStrip = re.compile(r"(\d)(,)(\d)")
    periodStrip = re.compile(r"(?<!\d)\.(?!\d)")
    for p in punct:
        if (p + " " in inText or " " + p in inText) or (re.search(commaStrip, inText) is not None):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText


def YOrN_Extraction(output):
    s = str(output).lower()
    words = process_punctuation(s).split()
    if "yes" in words and "no" not in words:
        return "Yes"
    if "yes" not in words and "no" in words:
        return "No"
    return "Unknown"


# The 14 MME categories, grouped as in VLMEvalKit MME_rating `super_cates`.
# VLMEvalKit names the second super-category 'reasoning'; the conventional MME
# name is 'cognition' — same 4 categories, max 800 points.
MME_PERCEPTION_CATEGORIES = [
    "OCR",
    "artwork",
    "celebrity",
    "color",
    "count",
    "existence",
    "landmark",
    "position",
    "posters",
    "scene",
]
MME_COGNITION_CATEGORIES = ["code_reasoning", "commonsense_reasoning", "numerical_calculation", "text_translation"]


def mme_rating(records: list[dict]) -> dict[str, float]:
    """Pure-python port of VLMEvalKit ``MME_rating``.

    ``records``: one dict per question with keys ``category``, ``image_key``
    (the pairing key — the two yes/no questions of one image share it) and
    ``score`` (0.0 or 1.0).

    Per category: ``acc`` = mean score over all questions x100, ``acc_plus`` =
    mean over images of (score_q1 * score_q2) x100 (upstream: ``val[0] * val[1]``);
    category score = acc + acc_plus (max 200). Perception = sum over its 10
    categories (max 2000); cognition = sum over its 4 (max 800); total max 2800.
    """
    stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        stats[r["category"]][str(r["image_key"])].append(float(r["score"]))

    def acc(key, mode="normal"):
        res = stats[key]
        values = []
        for val in res.values():
            if mode == "normal":
                values.extend(val)
            elif mode == "plus":
                # Upstream indexes val[0] * val[1] (asserting pairs implicitly).
                # Tolerate unpaired images (subset scoring): they contribute 0
                # to acc_plus but still count in the denominator.
                values.append(val[0] * val[1] if len(val) >= 2 else 0.0)
        return sum(values) / len(values) * 100 if values else 0.0

    scores = {k: acc(k) + acc(k, "plus") for k in stats}

    ret: dict[str, float] = {}
    ret["perception"] = sum(scores.get(c, 0.0) for c in MME_PERCEPTION_CATEGORIES)
    # VLMEvalKit calls this super-category 'reasoning'.
    ret["cognition"] = sum(scores.get(c, 0.0) for c in MME_COGNITION_CATEGORIES)
    ret.update(scores)
    return ret


def _f1(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float]:
    """``cal_f1_score`` from VLMEvalKit ``POPE_rating`` — returns (f1, precision, recall)."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return f1_score, precision, recall


def pope_rating(records: list[dict]) -> dict[str, dict[str, float]]:
    """Pure-python port of VLMEvalKit ``POPE_rating``.

    ``records``: one dict per question with keys ``answer`` ('Yes'/'No'),
    ``extracted`` ('Yes'/'No'/'Unknown') and ``category`` (may be a
    comma-joined multi-category string; upstream explodes it).

    Returns ``{split: {f1, precision, recall, accuracy}}`` (all 0-1) with an
    'Overall' split plus one per exploded category. The POPE headline
    ('Overall' column upstream) is the **F1 score**, not accuracy.
    """
    exploded: list[tuple[str, str, str]] = []
    for r in records:
        cats = str(r.get("category") or "").split(",")
        for c in cats:
            exploded.append((r["answer"], r["extracted"], c.strip()))

    def rate(rows):
        y_true = [1 if a == "Yes" else 0 for a, _, _ in rows]
        y_pred = [1 if e == "Yes" else 0 for _, e, _ in rows]
        f1, precision, recall = _f1(y_true, y_pred)
        acc = sum(1 for a, e, _ in rows if a == e) / len(rows) if rows else 0.0
        return {"f1": f1, "precision": precision, "recall": recall, "accuracy": acc}

    res = {"Overall": rate(exploded)}
    for cat in sorted({c for _, _, c in exploded if c}):
        res[cat] = rate([row for row in exploded if row[2] == cat])
    return res
