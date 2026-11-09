"""Completion-text preprocessing shared across scorers."""

import re
from collections import Counter
from dataclasses import dataclass

from pipette_scores.types import SampleCompletion


# Vendored from Liquid4All/liquid_evals (liquid_evals/parser.py).
def remove_thinking_tags(text: str) -> str:
    """Remove thinking tags and their content, then trim whitespace.

    Handles:
    - Complete pairs: <think>...</think> or [THINK]...[/THINK]
    - Orphan closing tags: </think> or [/THINK] at start (when opening tag was in prompt)
    - Orphan opening tags: <think> or [THINK] at end (truncated response)
    - GPT-OSS harmony channel preamble: drop everything up through the
      ``assistantfinal`` channel transition (with optional ``<|start|>`` /
      ``<|channel|>`` / ``<|message|>`` control-token variants).
    """
    # First, remove complete pairs
    text = re.sub(
        r"<think>.*?</think>|\[THINK\].*?\[/THINK\]",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Handle orphan closing tag at start (opening tag was in the prompt)
    # Match: optional content followed by </think> or [/THINK]
    text = re.sub(
        r"^.*?</think>|^.*?\[/THINK\]",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Handle orphan opening tag at end (response was truncated)
    text = re.sub(
        r"<think>.*$|\[THINK\].*$",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # GPT-OSS harmony: drop the analysis channel and keep only what comes after
    # the channel transition into the final channel. vLLM's openai_gptoss parser
    # strips the literal <|channel|>/<|message|>/<|start|>/<|end|> control tokens
    # but leaves the bare role/channel names, producing
    # ``...analysis...assistantfinal<answer>...``. Also accept the control-token
    # literals in case a future vLLM version leaves them in.
    match = re.search(
        r"(?:<\|start\|>)?assistant(?:<\|channel\|>)?final(?:<\|message\|>)?",
        text,
    )
    if match:
        text = text[match.end() :]

    # Trim any trailing harmony end markers if vLLM didn't strip them.
    text = re.sub(r"<\|(?:end|return)\|>\s*$", "", text)

    return text.strip()


# Stop reasons (plan-types' snake_case names) for a generation that ended
# before the model did: `truncated` is the output-token cap or a full slot
# context; `doom_loop` is a runaway-repetition stop (vLLM's
# repetition_detection, finish_reason "repetition", or a legacy client's
# detector).
_CUT_STOP_REASONS = frozenset({"truncated", "doom_loop"})

# What remove_thinking_tags treats as the end of the reasoning: a think close
# tag or the gpt-oss harmony transition into the final channel.
_REASONING_CLOSE = re.compile(r"(?i:</think>|\[/THINK\])|assistant(?:<\|channel\|>)?final")

# The closes remove_thinking_tags' orphan-close rule cuts through, in its order:
# the first </think>, else the first [/THINK].
_ORPHAN_CLOSES = (re.compile(r"</think>", re.IGNORECASE), re.compile(r"\[/THINK\]", re.IGNORECASE))

# Wrappers around the answer that are chat-format markup, not answer text:
# ERNIE-4.5's <response> (vLLM's ernie45 parser leaves it in the content) and
# Cohere North's <|START_TEXT|>. Each pair is (opener at a section start,
# closer at the end of the completion).
_ANSWER_WRAPPERS = tuple(
    (re.compile(r"\s*(" + re.escape(opener) + ")"), re.compile(re.escape(closer) + r"\s*\Z"))
    for opener, closer in (("<response>", "</response>"), ("<|START_TEXT|>", "<|END_TEXT|>"))
)


# <thinking>...</thinking>, as Ministral-3-Reasoning writes it after its own [/THINK].
_THINKING_TAG = re.compile(r"<(/?)thinking>", re.IGNORECASE)


def _gemma4_thought_to_think(text: str) -> str:
    """Rewrite Gemma 4 thought channels as <think> blocks.

    Reads the markup the way the Gemma 4 chat template's strip_thinking does:
    each ``<|channel>`` up to the next ``<channel|>`` is reasoning. An unclosed
    channel becomes an open <think>, which remove_thinking_tags removes to the
    end; a ``<channel|>`` with no opener before it is a stray marker and is
    dropped.
    """
    if "<|channel>" not in text and "<channel|>" not in text:
        return text
    *closed, tail = text.split("<channel|>")
    out = []
    for part in closed:
        head, opened, thought = part.partition("<|channel>")
        out.append(head + "<think>" + thought.replace("<|channel>", "") + "</think>" if opened else part)
    head, opened, thought = tail.partition("<|channel>")
    out.append(head + "<think>" + thought.replace("<|channel>", "") if opened else tail)
    return "".join(out)


# A wrapper nests a handful deep at most. normalize_vlm_reasoning_markup bounds the
# peeling so a runaway repeating the opener, which would peel one copy per pass over
# the whole text, is read in linear time.
_MAX_PEELS = 16


def _unwrap_answer(text: str, max_peels: int | None = None) -> str:
    """Drop answer wrappers that open a section and close the completion.

    A section starts at the top of the completion or right where the
    reasoning closes (where remove_thinking_tags' orphan-close rule cuts); the
    same tags anywhere else are answer text and stay. Wrappers are peeled
    until none is left, so one nested in another goes too, or for at most
    ``max_peels`` passes.
    """
    peeled = True
    passes = 0
    while peeled and (max_peels is None or passes < max_peels):
        passes += 1
        peeled = False
        for opener, closer in _ANSWER_WRAPPERS:
            # Find the section starts afresh: a peeled opener moves them.
            close = next((m for c in _ORPHAN_CLOSES if (m := c.search(text))), None)
            section_starts = (0, close.end()) if close else (0,)
            spans = [m.span(1) for start in section_starts if (m := opener.match(text, start))]
            if not spans:
                continue
            for begin, end in reversed(spans):
                text = text[:begin] + text[end:]
            text = closer.sub("", text)
            peeled = True
    return text


def normalize_reasoning_markup(text: str) -> str:
    """Rewrite model-specific reasoning markup into the form remove_thinking_tags reads.

    - Gemma 4 thought channels (``<|channel>thought…<channel|>``) become
      <think> blocks; stray channel markers are dropped.
    - Cohere North's ``<|START_THINKING|>`` / ``<|END_THINKING|>`` become
      <think> / </think>.
    - An ERNIE ``<response>…</response>`` or North
      ``<|START_TEXT|>…<|END_TEXT|>`` wrapper around the answer is dropped.

    Text without this markup is returned unchanged, and a second pass changes
    nothing. The result
    still carries its reasoning, so it goes through the usual stripping next:
    remove_thinking_tags, or ifstruct's validator, which strips on its own.
    """
    text = _gemma4_thought_to_think(text)
    text = text.replace("<|START_THINKING|>", "<think>").replace("<|END_THINKING|>", "</think>")
    return _unwrap_answer(text)


def normalize_vlm_reasoning_markup(text: str) -> str:
    """normalize_reasoning_markup for the VLM evals (split_reasoning), which also rewrites:

    - ``<thinking>`` / ``</thinking>`` into <think> / </think>;
    - Kimi-VL's ``◁think▷`` / ``◁/think▷``, plain text in its vocabulary (the
      server prints them), into <think> / </think>. A reply that never closes
      its block loses everything from the opener on, as Kimi's own
      ``extract_summary`` gives "" for it.

    The text evals keep normalize_reasoning_markup. Text without this markup is
    returned unchanged, and a second pass changes nothing.
    """
    text = _gemma4_thought_to_think(text)
    text = text.replace("<|START_THINKING|>", "<think>").replace("<|END_THINKING|>", "</think>")
    text = text.replace("◁think▷", "<think>").replace("◁/think▷", "</think>")
    text = _THINKING_TAG.sub(lambda m: "</think>" if m.group(1) else "<think>", text)
    return _unwrap_answer(text, max_peels=_MAX_PEELS)


# normalize_vlm_reasoning_markup has rewritten every family onto these. One scan finds them all,
# in time linear in the text however many a runaway completion repeats.
_TAG = re.compile(r"<think>|</think>|\[THINK\]|\[/THINK\]", re.IGNORECASE)
_HARMONY_FINAL = re.compile(r"(?:<\|start\|>)?assistant(?:<\|channel\|>)?final(?:<\|message\|>)?")
_HARMONY_END = re.compile(r"<\|(?:end|return)\|>\s*$")
_ANSWER_TAG = re.compile(r"<answer>|</answer>", re.IGNORECASE)

CLOSED = "closed"
ORPHAN_CLOSE = "orphan_close"
UNCLOSED_CUT = "unclosed_cut"
UNCLOSED_STOP = "unclosed_stop"
NO_REASONING = "no_reasoning"
STATUSES = (CLOSED, ORPHAN_CLOSE, UNCLOSED_CUT, UNCLOSED_STOP, NO_REASONING)


@dataclass(frozen=True, slots=True)
class Reasoning:
    """One completion, split into its reasoning and its answer.

    ``answer`` is what a scorer grades. ``status`` is one of STATUSES. ``blanked``
    marks the samples the strict view grades as empty. ``answer_before_close`` marks
    an answer read from before a close, see split_reasoning.
    """

    answer: str
    status: str
    blanked: bool
    answer_before_close: bool = False


def _answer_wrapper_content(text: str) -> str | None:
    """The content of the last complete ``<answer>...</answer>`` in the text."""
    opened: int | None = None
    found: tuple[int, int] | None = None
    for m in _ANSWER_TAG.finditer(text):
        if m.group()[1] != "/":
            opened = m.end()
        elif opened is not None:
            found = (opened, m.start())
            opened = None
    return text[found[0] : found[1]] if found else None


def _peel_answer_tag(answer: str) -> str:
    """Drop an ``<answer>`` that opens the answer and its ``</answer>`` that ends it."""
    head = answer.lstrip()
    if head[:8].lower() != "<answer>":
        return answer
    body = head[8:].rstrip()
    return body[:-9] if body[-9:].lower() == "</answer>" else body


def split_reasoning(
    completion: SampleCompletion,
    enable_thinking: bool | None = None,
    *,
    opens_reasoning: bool = False,
) -> Reasoning:
    """Split a completion into its reasoning and the answer to grade, whatever
    opened the reasoning and however it ended.

    Reasoning counts as **opened** when its opener is in the text (<think>,
    [THINK], and after normalize_vlm_reasoning_markup the ◁think▷, <thinking>, North,
    Gemma channel forms), when the prompt opened it for the model (``opens_reasoning``:
    the run's prompt profile is registered in prompt_profiles.REASONING_OPENERS,
    which covers an assistant prefill, a system prompt, a template and a model
    that always thinks), when ``enable_thinking`` is true, or when a close
    (</think>, [/THINK], the gpt-oss ``assistantfinal`` transition) shows
    it was opened out of sight, as with a hidden control token or a template's
    ``<think>\\n``.

    ``status``:
    - ``closed``: a close was read, and the opener was known, or is visible with no
      </think> or [/THINK] before it.
    - ``orphan_close``: a close was read, the opener was not known, and no opener is
      visible or a </think> or [/THINK] comes before the first one.
    - ``unclosed_cut``: opened, no close, and generation was cut (``truncated`` or
      ``doom_loop``).
    - ``unclosed_stop``: opened, no close, and the model stopped on its own.
    - ``no_reasoning``: no sign of reasoning.

    ``answer`` is graded as the thinking-tag rules of liquid_evals' parser
    have always graded, so a completion with no tags is graded whole, whatever the
    status: complete blocks are removed (a block is closed by the first close after
    its opener; an opener inside a block is part of it); the text through the first
    close that has no opener is reasoning, the first such </think>, else the first such
    [/THINK], as remove_thinking_tags' orphan-close rule cuts (a later stray close stays
    as text); an
    opener left at the end removes the rest; then the gpt-oss transition and trailing
    harmony markers go. Beyond that, for the cases that parser grades empty or
    wrongly:
    - the two tag families pair with each other (``[THINK]...</think>``), as ``<thinking>``
      ends a ``[THINK]`` block in Ministral-3-Reasoning's replies;
    - **answer before close**: when the opener was not known, the reply opens with a
      one-word answer and a close (``yes[/THINK]``), and everything after is more reasoning,
      that word is the answer (``answer_before_close``); more words before the close are
      reasoning, as Ministral writes in ``So the answer should be "yes".[/THINK]``, and grade empty;
    - **unclosed_stop** with the opener from ``opens_reasoning`` or in the text (``enable_thinking``
      alone is not enough) grades the content of the last complete ``<answer>...</answer>`` if
      there is one, else as the rules above;
    - an ``<answer>`` that opens the answer and its closing tag are dropped.

    ``blanked`` is the strict view: reasoning was opened by the opener in the text, the
    prompt, or ``enable_thinking``, generation was cut, no close exists and there is
    text. It does not depend on the run having recorded ``enable_thinking``: a reply that
    opened its own thought and was cut counts even under ``enable_thinking`` false.
    """
    text = normalize_vlm_reasoning_markup(completion.completion)
    cut = completion.stop_reason in _CUT_STOP_REASONS

    blocks: list[tuple[int, int]] = []
    orphan_closes: list[tuple[int, int]] = []
    in_block, block_start = False, 0
    visible_open = any_close = False
    for m in _TAG.finditer(text):
        is_close = m.group()[1] == "/" or m.group()[1:3] == "[/"
        if is_close:
            any_close = True
        else:
            visible_open = True
        if in_block:
            if is_close:
                blocks.append((block_start, m.end()))
                in_block = False
        elif is_close:
            orphan_closes.append((m.start(), m.end()))
        else:
            in_block, block_start = True, m.start()
    if in_block:
        blocks.append((block_start, len(text)))

    def visible(start: int, end: int) -> str:
        """The text of text[start:end] outside every block."""
        out, at = [], start
        for b_start, b_end in blocks:
            if b_end <= start or b_start >= end:
                continue
            out.append(text[at : max(b_start, at)])
            at = max(at, b_end)
        out.append(text[at:end])
        return "".join(out)

    # remove_thinking_tags' orphan-close rule cuts through the first </think>, else the first [/THINK].
    cut_at = next(
        (end for start, end in orphan_closes if text[start] == "<"), orphan_closes[0][1] if orphan_closes else 0
    )
    answer = visible(cut_at, len(text))
    before = visible(0, orphan_closes[0][0]) if orphan_closes else ""
    harmony = _HARMONY_FINAL.search(answer)
    if harmony:
        answer = answer[harmony.end() :]
    any_close = any_close or _HARMONY_FINAL.search(text) is not None

    known = opens_reasoning or enable_thinking is True
    opened = known or visible_open or any_close
    if not opened:
        status = NO_REASONING
    elif any_close:
        # Only a close with no opener before it is the sign of reasoning opened out of sight.
        orphan_first = bool(orphan_closes) and (not blocks or orphan_closes[0][0] < blocks[0][0])
        status = ORPHAN_CLOSE if not known and (orphan_first or not visible_open) else CLOSED
    else:
        status = UNCLOSED_CUT if cut else UNCLOSED_STOP

    fallback = False
    if (
        status == ORPHAN_CLOSE
        and orphan_closes
        and (not blocks or orphan_closes[0][0] < blocks[0][0])
        and not harmony
        and not answer.strip()
        and len(before.split()) == 1
        and any(b_start >= orphan_closes[0][1] for b_start, _ in blocks)
    ):
        answer, fallback = before, True
    if status == UNCLOSED_STOP and (opens_reasoning or visible_open):
        wrapped = _answer_wrapper_content(text)
        if wrapped is not None:
            answer = wrapped
    answer = _HARMONY_END.sub("", answer)
    if status != NO_REASONING:
        answer = _peel_answer_tag(answer)

    return Reasoning(
        answer=answer.strip(),
        status=status,
        blanked=(known or visible_open) and cut and not any_close and bool(text.strip()),
        answer_before_close=fallback,
    )


class ReasoningTally:
    """What a scorer reports of its completions' reasoning: how many came out in
    each status, and how many answers were read from before a close."""

    def __init__(self) -> None:
        self.status: Counter[str] = Counter()
        self.answer_before_close = 0

    def add(self, reasoning: Reasoning) -> Reasoning:
        self.status[reasoning.status] += 1
        self.answer_before_close += reasoning.answer_before_close
        return reasoning

    def context(self) -> dict[str, object]:
        return {
            "reasoning_status": {status: self.status[status] for status in STATUSES},
            "reasoning_answer_before_close": self.answer_before_close,
        }


def grading_text(completion: SampleCompletion, enable_thinking: bool | None) -> tuple[str, bool]:
    """Return the text to grade for a completion, and whether the strict view blanks it.

    The text is the completion after normalize_reasoning_markup, whatever the
    stop reason. Grading it with the thinking-tag rules is what liquid_evals
    does with every response, cut or not: a completion whose reasoning the
    prompt opened and the cap cut carries no tag and is graded whole, and one
    that opened <think> itself loses everything from the tag on.

    The flag marks the samples the strict view grades as empty instead: with
    thinking on, the client cut generation (``truncated`` or ``doom_loop``) and
    the text has no reasoning close. A completion that is already blank is not
    flagged. Scorers count the flagged samples as ``<eval>_blanked_truncated``
    and report ``<eval>_pass_at_1_blanked``, their pass@1 with those samples
    counted wrong.
    """
    text = normalize_reasoning_markup(completion.completion)
    blanked = (
        bool(enable_thinking)
        and completion.stop_reason in _CUT_STOP_REASONS
        and not _REASONING_CLOSE.search(text)
        and bool(text.strip())
    )
    return text, blanked
