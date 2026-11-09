import pytest

from pipette_scores.scoring.completion_text import grading_text, normalize_reasoning_markup, remove_thinking_tags
from pipette_scores.types import SampleCompletion


@pytest.mark.parametrize(
    "raw, expected",
    [
        # No tags: passthrough (trimmed).
        ("  the answer is 42  ", "the answer is 42"),
        ("", ""),
        # Complete pairs removed.
        ("<think>reasoning</think>answer", "answer"),
        ("[THINK]reasoning[/THINK]answer", "answer"),
        ("a<think>x</think>b<think>y</think>c", "abc"),
        # Case-insensitive.
        ("<THINK>reasoning</THINK>answer", "answer"),
        # Multiline content (DOTALL).
        ("<think>line1\nline2</think>answer", "answer"),
        # Orphan closing tag at start (opening tag was in the prompt).
        ("leftover reasoning</think>answer", "answer"),
        ("leftover[/THINK]answer", "answer"),
        # Orphan opening tag at end (truncated response).
        ("answer<think>truncated reasoning", "answer"),
        ("answer[THINK]truncated", "answer"),
        # GPT-OSS harmony channels.
        ('<|start|>assistant<|channel|>final<|message|>{"x":1}<|return|>', '{"x":1}'),
        ('assistant<|channel|>final<|message|>{"x":1}<|end|>', '{"x":1}'),
        ('assistantfinal{"x":1}', '{"x":1}'),
        ('<think>a</think>assistantfinal{"y":2}', '{"y":2}'),
    ],
)
def test_remove_thinking_tags(raw, expected):
    assert remove_thinking_tags(raw) == expected


_J = '{"name": "Ada Lovelace", "born": 1815}'

# Completions as the runtimes deliver them, and what remove_thinking_tags leaves
# once normalize_reasoning_markup has run.
_MARKUP_CASES = [
    # Gemma 4 on llama.cpp: the thought channel, then the answer.
    pytest.param(
        "<|channel>thought\nThe user wants a JSON object with name and born, nothing else.\n"
        f"Draft: {_J}\nLooks right.<channel|>{_J}",
        _J,
        id="gemma4-closed",
    ),
    pytest.param(
        "<|channel>thought<channel|>The answer is $\\boxed{391}$.", "The answer is $\\boxed{391}$.", id="gemma4-empty"
    ),
    pytest.param(f"{_J}<|channel>thought\n<channel|>", _J, id="gemma4-trailing-empty"),
    # Cut inside the thought: nothing but reasoning.
    pytest.param(
        "<|channel>thought\nCompute 17*23: 17*20 = 340, 17*3 = 51, total 391. Now double-check whether",
        "",
        id="gemma4-unterminated",
    ),
    pytest.param(
        "<|channel><|channel>thought\nOption C fits the spectrum.<channel|>Answer: C",
        "Answer: C",
        id="gemma4-double-open",
    ),
    pytest.param("Answer: C<channel|>", "Answer: C", id="gemma4-stray-close"),
    # An answer, a second thought, then the answer again: the thought goes, as
    # the template's strip_thinking reads it.
    pytest.param(
        "I don't know\n<|channel>thought\nThe reply must be one word per line.<channel|>I\ndon't\nknow",
        "I don't know\nI\ndon't\nknow",
        id="gemma4-second-thought",
    ),
    # vLLM without a reasoning parser strips both channel tokens: no delimiter left.
    pytest.param(
        f"thought\nThe user wants name and born.{_J}", f"thought\nThe user wants name and born.{_J}", id="gemma4-vllm"
    ),
    # gpt-oss's <|channel|> is a different token.
    pytest.param(
        "<|channel|>analysis<|message|>We need the option with the lowest energy.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Answer: B",
        "Answer: B",
        id="gpt-oss-harmony",
    ),
    # ERNIE-4.5-Thinking: the prompt opens <think>, the answer is wrapped.
    pytest.param(
        'The schema needs title and year; Dune was published in 1965.\n</think>\n\n<response>\n{"title": "Dune", '
        '"year": 1965}\n</response>\n',
        '{"title": "Dune", "year": 1965}',
        id="ernie",
    ),
    pytest.param(
        'Need fenced output.\n</think>\n\n<response>\n```json\n{"a": 1}\n```\n</response>\n',
        '```json\n{"a": 1}\n```',
        id="ernie-fenced",
    ),
    # vLLM's ernie45 parser moves the reasoning out but leaves the wrapper.
    pytest.param(
        '\n\n<response>\n```json\n{"a": 1}\n```\n</response>\n', '```json\n{"a": 1}\n```', id="ernie-vllm-parser"
    ),
    # Cut inside the answer: no closing tag.
    pytest.param(
        "Reason.\n</think>\n\n<response>\nThe answer is (B) because the",
        "The answer is (B) because the",
        id="ernie-unclosed",
    ),
    pytest.param("abc\n</think>\ndef", "def", id="ernie-unwrapped"),
    # The tags are answer text anywhere but at the start of a section.
    pytest.param(
        "Reason.\n</think>\nWrap the reply in a <response> element, then close it with </response>.",
        "Wrap the reply in a <response> element, then close it with </response>.",
        id="response-tag-mid-answer",
    ),
    pytest.param(
        "Return it as <response>ok</response>",
        "Return it as <response>ok</response>",
        id="response-tag-mid-text",
    ),
    # Cohere North on vLLM: its markers are not special tokens, so they reach the text.
    pytest.param(
        "The user wants JSON only, no prose. Keys: city, country.<|END_THINKING|>"
        '<|START_TEXT|>{"city": "Paris", "country": "France"}<|END_TEXT|>',
        '{"city": "Paris", "country": "France"}',
        id="north-on",
    ),
    pytest.param(
        '<|START_TEXT|>{"city": "Paris", "country": "France"}<|END_TEXT|>',
        '{"city": "Paris", "country": "France"}',
        id="north-off",
    ),
    # Off mode can still reason inside the text section, then close it.
    pytest.param(
        '<|START_TEXT|>The densest is option c.\n\nThus answer: C. The answer format: "Answer: C".<|END_THINKING|>'
        "<|START_TEXT|>Answer: C<|END_TEXT|>",
        "Answer: C",
        id="north-off-reasoning",
    ),
    pytest.param(
        "<|START_TEXT|><|START_THINKING|>Option C keeps the three-step order.<|END_THINKING|>"
        "<|START_TEXT|>Answer: C<|END_TEXT|>",
        "Answer: C",
        id="north-off-thinking",
    ),
    pytest.param("Plan.<|END_THINKING|><|START_TEXT|>The answer is", "The answer is", id="north-unclosed"),
    # North on llama.cpp without preserved_tokens: every marker is hidden.
    pytest.param(
        f"The user wants JSON only, no prose.{_J}", f"The user wants JSON only, no prose.{_J}", id="north-llamacpp"
    ),
    pytest.param(
        "Its replies sit between <|START_TEXT|> and <|END_TEXT|> markers.",
        "Its replies sit between <|START_TEXT|> and <|END_TEXT|> markers.",
        id="text-markers-mid-text",
    ),
    # Peeling one wrapper moves the section starts the next one is looked for at.
    pytest.param(
        "<response>Option C fits.</think><|START_TEXT|>Answer: C<|END_TEXT|>",
        "Answer: C",
        id="response-then-text-wrapper",
    ),
    pytest.param(
        "<response>Plan.</think>The marker <|START_TEXT|> opens North's answer.",
        "The marker <|START_TEXT|> opens North's answer.",
        id="text-marker-mid-answer-after-peel",
    ),
    # A wrapper nested in another goes too.
    pytest.param(
        "Option C fits.</think><|START_TEXT|><response>Answer: C</response><|END_TEXT|>",
        "Answer: C",
        id="nested-wrappers",
    ),
    # A section also starts where [/THINK] closes the reasoning.
    pytest.param("Plan.[/THINK]<response>Answer: B</response>", "Answer: B", id="response-after-mistral-close"),
    pytest.param("  42  ", "42", id="plain"),
    pytest.param(f"<think>r</think>\n\n{_J}", _J, id="think"),
]


@pytest.mark.parametrize("raw, expected", _MARKUP_CASES)
def test_normalize_reasoning_markup(raw, expected):
    assert remove_thinking_tags(normalize_reasoning_markup(raw)) == expected


@pytest.mark.parametrize("raw, expected", _MARKUP_CASES)
def test_normalize_reasoning_markup_is_idempotent(raw, expected):
    once = normalize_reasoning_markup(raw)
    assert normalize_reasoning_markup(once) == once


@pytest.mark.parametrize(
    "raw",
    [
        "the answer is 42",
        "<think>reasoning</think>answer",
        "leftover reasoning</think>answer",
        "answer<think>truncated reasoning",
        'assistantfinal{"x":1}',
        "Return it as <response>ok</response>",
    ],
)
def test_normalize_reasoning_markup_leaves_other_text_alone(raw):
    assert normalize_reasoning_markup(raw) == raw


_PROMPT_OPENED = "Let me compare the four options. Option A predicts"


@pytest.mark.parametrize(
    "raw, stop_reason, enable_thinking, blanked",
    [
        # The prompt opened the reasoning and the cap cut it: no tag at all.
        (_PROMPT_OPENED, "truncated", True, True),
        # A repetition stop (vLLM's repetition_detection) also cut generation.
        (_PROMPT_OPENED, "doom_loop", True, True),
        # Thinking off or not reported: graded as before.
        (_PROMPT_OPENED, "truncated", False, False),
        (_PROMPT_OPENED, "truncated", None, False),
        # Not cut, or not known to be.
        (_PROMPT_OPENED, "eos", True, False),
        (_PROMPT_OPENED, "unknown", True, False),
        (_PROMPT_OPENED, "failure", True, False),
        (_PROMPT_OPENED, None, True, False),
        # Nothing to drop: generation produced no text.
        ("", "truncated", True, False),
        # The reasoning closed and the answer was cut: graded as is.
        ("Reason.\n</think>\n\n<response>\nThe answer is (B) because the", "truncated", True, False),
        ("<|channel>thought\nOption C fits.<channel|>The answer is", "truncated", True, False),
        ("Plan.<|END_THINKING|><|START_TEXT|>The answer is", "truncated", True, False),
        ("[THINK]plan[/THINK]Answer: B", "truncated", True, False),
        (
            "<|channel|>analysis<|message|>Done.<|end|><|start|>assistant<|channel|>final<|message|>Answer",
            "truncated",
            True,
            False,
        ),
        # gpt-oss cut in its analysis channel.
        (
            "<|channel|>analysis<|message|>We need to compute the integral. Let u = x^2, then du =",
            "truncated",
            True,
            True,
        ),
        # Gemma 4 cut inside its thought.
        ("<|channel>thought\nCompute 17*23: 17*20 = 340", "truncated", True, True),
    ],
)
def test_grading_text(raw, stop_reason, enable_thinking, blanked):
    text, cut = grading_text(SampleCompletion(id="1", completion=raw, stop_reason=stop_reason), enable_thinking)
    # The flag only marks the samples the strict view blanks; the text graded
    # is the rewritten completion either way.
    assert cut is blanked
    assert text == normalize_reasoning_markup(raw)


# What a cut completion is graded on: the text after the markup rewrite and
# remove_thinking_tags. For completions without Gemma 4, North or ERNIE markup
# this is what liquid_evals (d5534cf, parser.remove_thinking_tags, called
# without opener_in_prompt by gpqa_diamond_aa, ifbench_aa, ifstruct and
# math500) grades for the same response.
@pytest.mark.parametrize("enable_thinking", [True, False, None])
@pytest.mark.parametrize("stop_reason", ["truncated", "doom_loop"])
@pytest.mark.parametrize(
    "raw, graded",
    [
        # The prompt opened the reasoning: no tag, graded whole.
        pytest.param(_PROMPT_OPENED, _PROMPT_OPENED, id="prompt-opened"),
        # The completion opened <think> itself: everything from the tag on goes.
        pytest.param("<think>Option B fits. Answer: B. Double-check", "", id="opened-in-text"),
        pytest.param("Sure.\n<think>Option B fits. Answer: B", "Sure.", id="opened-after-text"),
        pytest.param("[THINK]Option B fits. Answer: B", "", id="mistral-opened"),
        # gpt-oss cut in its analysis channel: no final channel, graded whole.
        pytest.param(
            "<|channel|>analysis<|message|>Let u = x^2, then du =",
            "<|channel|>analysis<|message|>Let u = x^2, then du =",
            id="gpt-oss-analysis",
        ),
        # The reasoning closed and the answer was cut: the answer so far.
        pytest.param(
            "Reason.\n</think>\n\nThe answer is (B) because the", "The answer is (B) because the", id="closed"
        ),
        # Markup liquid_evals does not read keeps the scorer's reading.
        pytest.param("<|channel>thought\nCompute 17*23: 17*20 = 340", "", id="gemma4-thought"),
        pytest.param("<|START_THINKING|>Option B fits. Answer: B", "", id="north-thinking"),
        pytest.param("Reason.\n</think>\n\n<response>\nThe answer is (B", "The answer is (B", id="ernie-response"),
    ],
)
def test_cut_completion_is_graded_like_liquid_evals(raw, graded, stop_reason, enable_thinking):
    text, _ = grading_text(SampleCompletion(id="1", completion=raw, stop_reason=stop_reason), enable_thinking)
    assert remove_thinking_tags(text) == graded
