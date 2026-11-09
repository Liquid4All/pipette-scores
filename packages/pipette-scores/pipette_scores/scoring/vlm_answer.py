"""Deterministic fallbacks for answers VLMEvalKit's exact matching cannot read.

A pipette extension, not part of the VLMEvalKit port: upstream hands these
answers to an LLM judge, and pipette never uses one. The vendored rule runs
first (``can_infer`` for options, ``YOrN_Extraction`` for yes/no), so every
answer it reads is scored exactly as upstream scores it. Only when it finds
nothing do these rules look for an answer the text commits to, in order:

1. the last explicit answer statement ("the correct answer is: **B**",
   "Final Answer: No", a JSON field `"answer": "C"`), or the last line that
   is that field alone, its key in backticks (`` `answer`: "C" ``),
2. the last ``\\boxed{...}``,
3. an answer given first and explained after: a first line that opens with
   an option letter ("B. Not the same") or a yes/no, or a line that is exactly
   an option ("C. painter"), unless the text walks through several options,
4. a quoted answer: the whole answer one quoted letter (`"B"`, the field's
   value given without its key), or, after text that names no option letter,
   a last line that is one quoted capital letter or one quoted option
   (`"B. Not the same"`),
5. the vendored rule again, on the last non-empty line alone.

Reasoning models are graded on the text after their reasoning is removed
(``grading_text`` and ``remove_thinking_tags``), so the same rules apply to
both. An answer none of them reads stays wrong.
"""

import re

from pipette_scores.scoring.vlm_upstream.matching_util import can_infer
from pipette_scores.scoring.vlm_upstream.yorn import YOrN_Extraction

# "answer", "final answer", "correct option", ... then an optional "is" and
# colon, and any markdown before the answer itself. Quotes around the key and
# the value read a JSON field, `"answer": "C"`, the format Qwen3.5's model card
# asks for on multiple-choice questions.
_STATEMENT = (
    r"(?i:\b(?:final\s+answer|correct\s+answer|best\s+answer|correct\s+option|"
    r"correct\s+choice|best\s+option|answer)\b"
    r"(?:\s+(?:is|would\s+be|should\s+be))?)"
    r"[\"']?\s*[:：]?\s*(?:>\s*)?[*_`\"']*\s*[(\[]?\s*"
)
# An option letter is uppercase and not the start of a word ("All of ...").
_OPTION_STATEMENT = re.compile(_STATEMENT + r"([A-Z])(?![A-Za-z0-9])")
# Not "answer yes or no": a reply that names both is saying it cannot choose.
_YES_NO_STATEMENT = re.compile(_STATEMENT + r"(?i:(yes|no))\b(?!\s*(?i:or|/)\s*(?i:yes|no)\b)")
_BOXED = re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*([^{}]*?)\s*\}?\s*\}")
_MARKDOWN = re.compile(r"[*_`>#]+")


# A line that opens with an option letter: "B. ...", "(C) ...", "**D**".
_OPTION_LINE = re.compile(r"^[*_`>#\s]*\(?([A-Z])(?:\)|\.|:|\*\*|$)\s*(.*)$")

# The shapes Qwen3.5 writes the `"answer": "C"` field its card asks for in, each
# read only as a whole line, so prose that mentions the format or refuses to
# choose is never one:
# - the field, its key quoted or in backticks: `` `answer`: "C" ``, `` `answer: "C"` ``;
_FIELD_LINE = r"^[\s{]*`?[\"']?answer[\"']?`?\s*[:：]?\s*[`\"'](VALUE)[`\"']\s*`?\s*[,}]?\s*$"
_ANSWER_FIELD_LINE = re.compile(_FIELD_LINE.replace("VALUE", "[A-Za-z]"), re.I | re.M)
_YES_NO_FIELD_LINE = re.compile(_FIELD_LINE.replace("VALUE", "yes|no"), re.I | re.M)
# - the field's value alone, the whole answer: `"B"`, `"a"`, an unclosed `"B`;
_QUOTED_LETTER = re.compile(r"^[\"'“‘]([A-Za-z])[\"'”’]?\.?$")
# - one option, quoted, as the last line: `"B. Not the same"`.
_QUOTED_OPTION_LINE = re.compile(r"^[\"'“‘]\s*([A-Z])[.)]\s*(.+?)\s*[\"'”’]\.?$")


def _last_commitment(text: str, patterns: tuple[re.Pattern[str], ...], valid) -> str | None:
    """The value of the last statement or answer field in the text, whichever
    comes last: a reply that revises its answer is read at its final word."""
    found = [(m.start(1), m.group(1)) for p in patterns for m in p.finditer(text) if valid(m.group(1))]
    return max(found)[1] if found else None


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.strip().splitlines() if ln.strip()]


def _option_lines(text: str, choices: dict[str, str]) -> list[tuple[str, str]]:
    """Each line that opens with an option letter, with the rest of the line."""
    out = []
    for line in _lines(text):
        match = _OPTION_LINE.match(line)
        if match and match.group(1) in choices:
            out.append((match.group(1), _MARKDOWN.sub(" ", match.group(2)).strip()))
    return out


def _leading_option(text: str, choices: dict[str, str]) -> str | None:
    """The option an answer opens with or states as a line of its own, when the
    text does not walk through several options."""
    lines = _option_lines(text, choices)
    if len({letter for letter, _ in lines}) != 1:
        return None
    first = _lines(text)[0]
    match = _OPTION_LINE.match(first)
    if match and match.group(1) in choices:
        return match.group(1)
    letter, rest = lines[0]
    return letter if rest and _words(rest) == _words(choices[letter]) else None


def _words(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return _MARKDOWN.sub(" ", lines[-1]).strip() if lines else ""


def _option_in(value: str, choices: dict[str, str]) -> str | None:
    match = re.match(r"\(?\s*([A-Z])(?![A-Za-z0-9])", value.strip())
    return match.group(1) if match and match.group(1) in choices else None


def option(text: str, choices: dict[str, str]) -> tuple[str | None, bool]:
    """The option letter the answer gives, and whether a fallback found it."""
    upstream = can_infer(text, choices)
    if upstream:
        return upstream, False
    stated = _last_commitment(text, (_OPTION_STATEMENT, _ANSWER_FIELD_LINE), lambda v: v.upper() in choices)
    if stated:
        return stated.upper(), True
    boxed = [o for b in _BOXED.findall(text) if (o := _option_in(b, choices))]
    if boxed:
        return boxed[-1], True
    leading = _leading_option(text, choices)
    if leading:
        return leading, True
    quoted = _quoted_answer(text, choices)
    if quoted:
        return quoted, True
    last = can_infer(_last_line(text), choices)
    return (last, True) if last and last in choices else (None, False)


def _quoted_answer(text: str, choices: dict[str, str]) -> str | None:
    """The option a quoted answer gives: the whole answer one quoted letter
    (any case), or a last line that is one quoted capital letter, or one quoted
    option, after text that names no option letter."""
    lines = _lines(text)
    if not lines:
        return None
    whole = _QUOTED_LETTER.match(text.strip())
    if whole:
        return whole.group(1).upper() if whole.group(1).upper() in choices else None
    if any(re.search(rf"\b{letter}\b", line) for line in lines[:-1] for letter in choices):
        return None
    letter = _QUOTED_LETTER.match(lines[-1])
    if letter and letter.group(1) in choices:
        return letter.group(1)
    option_line = _QUOTED_OPTION_LINE.match(lines[-1])
    if option_line and option_line.group(1) in choices:
        letter, rest = option_line.groups()
        return letter if _words(rest) == _words(choices[letter]) else None
    return None


def option_hit(text: str, choices: dict[str, str], gt: str) -> tuple[bool, bool]:
    """Whether the answer gives the ground-truth letter, and whether a fallback
    read it. Unreadable is wrong, as under upstream's exact matching."""
    letter, fallback = option(text, choices)
    return letter is not None and letter == str(gt).strip(), fallback


def yes_no(text: str) -> tuple[str, bool]:
    """'Yes', 'No' or 'Unknown', and whether a fallback read it."""
    upstream = YOrN_Extraction(text)
    if upstream != "Unknown":
        return upstream, False
    stated = _last_commitment(text, (_YES_NO_STATEMENT, _YES_NO_FIELD_LINE), lambda _: True)
    if stated:
        return stated.capitalize(), True
    boxed = [b.strip().lower() for b in _BOXED.findall(text)]
    boxed = [b for b in boxed if b in ("yes", "no")]
    if boxed:
        return boxed[-1].capitalize(), True
    first = re.match(r"[*_`>#\s]*(yes|no)[*_`]*\s*(?:[,.!:;]|$)", text, re.I | re.M)
    if first:
        return first.group(1).capitalize(), True
    last = YOrN_Extraction(_last_line(text))
    return (last, True) if last != "Unknown" else ("Unknown", False)
