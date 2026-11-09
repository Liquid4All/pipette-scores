"""split_reasoning: one reading of a completion's reasoning, whatever opened it and however it ended."""

import random
import time

import pytest

from pipette_scores.dataset_catalog import prompt_profiles
from pipette_scores.scoring import mmstar, ocrbench, pope, realworldqa, score
from pipette_scores.scoring.completion_text import (
    CLOSED,
    NO_REASONING,
    ORPHAN_CLOSE,
    STATUSES,
    UNCLOSED_CUT,
    UNCLOSED_STOP,
    ReasoningTally,
    grading_text,
    normalize_reasoning_markup,
    normalize_vlm_reasoning_markup,
    remove_thinking_tags,
    split_reasoning,
)
from pipette_scores.types import EvalId, SampleCompletion, VlmSample


def comp(text: str, stop: str | None = "eos") -> SampleCompletion:
    return SampleCompletion(id="1", completion=text, stop_reason=stop)


# --- 1. Nothing that scored before scores differently ------------------------------------------
# Completions of the families scored before, with what the thinking-tag rules of main (a8afd30) graded
# each as (the completion's grading text, then remove_thinking_tags) and whether the strict view,
# with enable_thinking true and the completion cut, blanked it.
EXISTING = [
    pytest.param("<think>\nThe circle is red.\n</think>\n\nC", "C", False, id="qwen-closed"),
    pytest.param(
        "<think>\nA? no. B? no.\n</think>\n\nThe answer is $\\boxed{C}$.",
        "The answer is $\\boxed{C}$.",
        False,
        id="qwen-closed-boxed",
    ),
    pytest.param("The circle is red, so option C.\n</think>\n\nAnswer: C", "Answer: C", False, id="qwen-prompt-opened"),
    pytest.param("<think>\nThe circle is red, so the answer is C but", "", True, id="qwen-unclosed"),
    pytest.param("<think>\n\n</think>\n\nC", "C", False, id="qwen-no-thinking"),
    pytest.param('<think>\nRed.\n</think>\n\n"answer": "C"', '"answer": "C"', False, id="qwen-answer-field"),
    pytest.param("<|channel>thought\nThe circle is red.<channel|>C", "C", False, id="gemma-closed"),
    pytest.param("<|channel>thought\n<channel|>Answer: C", "Answer: C", False, id="gemma-empty"),
    pytest.param("<|channel>thought\nThe circle is red, so C and", "", True, id="gemma-unterminated"),
    pytest.param("The circle is red.<channel|>C", "The circle is red.C", True, id="gemma-stray-close"),
    pytest.param(
        "The circle is red.<|END_THINKING|><|START_TEXT|>Answer: C<|END_TEXT|>", "Answer: C", False, id="cohere-on"
    ),
    pytest.param("<|START_TEXT|>C<|END_TEXT|>", "C", True, id="cohere-off"),
    pytest.param(
        "<|START_THINKING|>Red.<|END_THINKING|><|START_TEXT|>C<|END_TEXT|>", "C", False, id="cohere-thinking-block"
    ),
    pytest.param("Red.</think>\n\n<response>\nC\n</response>", "C", False, id="ernie"),
    pytest.param("We need the colour. It is red.assistantfinalC", "C", False, id="gptoss-bare"),
    pytest.param(
        "<|channel|>analysis<|message|>Red.<|end|><|start|>assistant<|channel|>final<|message|>C<|return|>",
        "C",
        False,
        id="gptoss-tokens",
    ),
    pytest.param(
        "<|channel|>analysis<|message|>We need to find the colour. Let u =",
        "<|channel|>analysis<|message|>We need to find the colour. Let u =",
        True,
        id="gptoss-cut",
    ),
    pytest.param("<think>\nThe circle is red.\n</think>\nAnswer: C", "Answer: C", False, id="minicpm-think"),
    pytest.param("The circle is red.\n\nAnswer: C", "The circle is red.\n\nAnswer: C", True, id="minicpm-plain"),
    pytest.param("<think>\nThe image shows a red circle on the left.\n</think>\n\nC", "C", False, id="internvl-r1"),
    pytest.param("<think>\nThe image shows a red circle and", "", True, id="internvl-r1-cut"),
    pytest.param("C", "C", True, id="lfm-plain"),
    pytest.param("C. red", "C. red", True, id="lfm-letter-text"),
    pytest.param(
        "The circle is red, which is option C.", "The circle is red, which is option C.", True, id="lfm-sentence"
    ),
    pytest.param("[THINK]The circle is red.[/THINK]C", "C", False, id="ministral-closed"),
    pytest.param("The circle is red.[/THINK]C", "C", False, id="ministral-orphan"),
    pytest.param("[THINK]The circle is red, so", "", True, id="ministral-cut"),
    pytest.param("<think>\nGot it, the circle is red.\n</think>\n\n\\boxed{C}", "\\boxed{C}", False, id="mimo"),
    pytest.param("<think>a</think>C<think>b</think>", "C", False, id="two-blocks"),
    pytest.param("<think>a</think>b</think>C", "C", False, id="two-closes"),
    pytest.param("[THINK]a[/THINK]C[THINK]more", "C", False, id="answer-then-open"),
    pytest.param("", "", False, id="empty"),
    pytest.param("  \n  ", "", False, id="whitespace"),
    pytest.param("<THINK>a</THINK>C", "C", False, id="think-uppercase"),
    # remove_thinking_tags' orphan-close rule cuts through the first </think>, else the first [/THINK].
    pytest.param("x[/THINK]y</think>z", "z", False, id="mixed-orphan-closes"),
    # InternVL3.5 under the R1 prompt that wrote no <think>: graded whole, its <answer> included.
    pytest.param(
        "The circle is red, so <answer>C</answer> it is.",
        "The circle is red, so <answer>C</answer> it is.",
        True,
        id="internvl-r1-untagged-answer",
    ),
]


@pytest.mark.parametrize("completion, answer, blanked", EXISTING)
def test_existing_families_grade_as_they_did(completion, answer, blanked):
    assert remove_thinking_tags(normalize_reasoning_markup(completion)) == answer
    assert split_reasoning(comp(completion)).answer == answer
    cut = split_reasoning(comp(completion, "truncated"), True)
    assert cut.answer == answer
    assert cut.blanked is blanked
    assert grading_text(comp(completion, "truncated"), True) == (normalize_reasoning_markup(completion), blanked)


_MCQ = {"A": "blue", "B": "green", "C": "red", "D": "black", "answer": "C", "l2_category": "x"}


def test_existing_families_score_as_they_did_on_mmstar():
    """The letter each family's reply reaches, through the scorer: every reply above that names C."""
    named_c = [
        c for c, a, _ in (p.values for p in EXISTING) if a.strip() and "C" in a and "\\boxed" not in a and len(a) < 40
    ]
    assert len(named_c) >= 12
    samples = [VlmSample(id=str(i), prompt="q", answer=None, raw_doc=_MCQ) for i in range(len(named_c))]
    completions = [SampleCompletion(id=str(i), completion=c, stop_reason="eos") for i, c in enumerate(named_c)]
    _, ctx = mmstar.score(completions, samples)
    assert ctx["accuracy"] == 1.0
    assert ctx["blanked_truncated"] == 0


# --- 2. The matrix: where the opener came from x how the reasoning ended x tag family -----------
# (open tag, close tag); a hidden opener has only its close in the text.
FAMILIES = {
    "think": ("<think>", "</think>"),
    "THINK": ("[THINK]", "[/THINK]"),
    "kimi": ("\u25c1think\u25b7", "\u25c1/think\u25b7"),
    "north": ("<|START_THINKING|>", "<|END_THINKING|>"),
    "thinking": ("<thinking>", "</thinking>"),
    "gemma": ("<|channel>thought\n", "<channel|>"),
}
# Gemma's <channel|> with no opener before it is a stray marker and is dropped, so a hidden Gemma opener is
# not one a reply can close.
HIDDEN_OK = [f for f in FAMILIES if f != "gemma"]
R, A = "The circle is red, so the option is C", "C"

# source -> (opens_reasoning, enable_thinking). visible: the model writes the opener, nothing declared.
# prefill: phi4_rv_think's assistant "<think>". system: a profile's system message has the model reason from
# its first token (nemotron_think's /think). template: the chat template opens it in the generation prompt
# (Step3), declared by enable_thinking. hidden: a control token no text shows (Ministral's [THINK]), nothing declared.
SOURCES = {
    "visible": (False, None),
    "system": (True, None),
    "prefill": (True, None),
    "template": (False, True),
    "hidden": (False, None),
}


def _text(family: str, shape: str) -> str:
    o, c = FAMILIES[family]
    return {
        "opened_closed": f"{o}{R}{c}{A}",
        "opened_unclosed": f"{o}{R}",
        "hidden_closed": f"{R}{c}{A}",
        "hidden_unclosed": R,
        "plain": A,
    }[shape]


# (source, shape, stop) -> (answer, status, blanked). The answer of an unclosed reply is what the
# thinking-tag rules grade: the whole text when no tag shows the reasoning, nothing when its opener does.
MATRIX = {
    # The opener is in the text and nothing says so beforehand.
    ("visible", "opened_closed", "eos"): (A, CLOSED, False),
    ("visible", "opened_closed", "truncated"): (A, CLOSED, False),
    ("visible", "opened_unclosed", "truncated"): ("", UNCLOSED_CUT, True),
    ("visible", "opened_unclosed", "doom_loop"): ("", UNCLOSED_CUT, True),
    ("visible", "opened_unclosed", "eos"): ("", UNCLOSED_STOP, False),
    ("visible", "hidden_closed", "eos"): (A, ORPHAN_CLOSE, False),
    ("visible", "hidden_closed", "truncated"): (A, ORPHAN_CLOSE, False),
    ("visible", "plain", "eos"): (A, NO_REASONING, False),
    ("visible", "plain", "truncated"): (A, NO_REASONING, False),
    # A hidden opener nothing declared: only a close shows it, and a reply cut before the close is not known as reasoning.
    ("hidden", "hidden_closed", "eos"): (A, ORPHAN_CLOSE, False),
    ("hidden", "hidden_closed", "truncated"): (A, ORPHAN_CLOSE, False),
    ("hidden", "hidden_unclosed", "truncated"): (R, NO_REASONING, False),
    ("hidden", "plain", "eos"): (A, NO_REASONING, False),
}
for _source in ("system", "prefill", "template"):
    MATRIX.update(
        {
            # The opener is known; the model may write it again.
            (_source, "opened_closed", "eos"): (A, CLOSED, False),
            (_source, "opened_unclosed", "truncated"): ("", UNCLOSED_CUT, True),
            (_source, "opened_unclosed", "doom_loop"): ("", UNCLOSED_CUT, True),
            (_source, "opened_unclosed", "eos"): ("", UNCLOSED_STOP, False),
            (_source, "hidden_closed", "eos"): (A, CLOSED, False),
            (_source, "hidden_closed", "truncated"): (A, CLOSED, False),
            # Cut inside reasoning the prompt opened: graded whole, counted by the strict view.
            (_source, "hidden_unclosed", "truncated"): (R, UNCLOSED_CUT, True),
            (_source, "hidden_unclosed", "doom_loop"): (R, UNCLOSED_CUT, True),
            # The model stopped without ever closing: graded as written, flagged.
            (_source, "hidden_unclosed", "eos"): (R, UNCLOSED_STOP, False),
            (_source, "plain", "eos"): (A, UNCLOSED_STOP, False),
            (_source, "plain", "truncated"): (A, UNCLOSED_CUT, True),
        }
    )


def _cells():
    for (source, shape, stop), expected in MATRIX.items():
        families = (
            HIDDEN_OK
            if shape.startswith("hidden") and source != "visible" or shape == "hidden_closed"
            else list(FAMILIES)
        )
        if shape == "plain":
            families = ["think"]
        if shape == "hidden_unclosed":
            families = ["think"]
        for family in families:
            yield pytest.param(source, shape, stop, family, expected, id=f"{source}-{shape}-{stop}-{family}")


@pytest.mark.parametrize("source, shape, stop, family, expected", list(_cells()))
def test_the_matrix(source, shape, stop, family, expected):
    answer, status, blanked = expected
    opens, thinking = SOURCES[source]
    text = _text(family, shape)
    got = split_reasoning(comp(text, stop), thinking, opens_reasoning=opens)
    assert (got.answer, got.status, got.blanked) == (answer, status, blanked)
    # A second pass over the answer changes nothing.
    again = split_reasoning(comp(got.answer, stop), thinking, opens_reasoning=opens)
    assert again.answer == got.answer


def test_the_matrix_covers_every_status_from_every_source():
    for source in SOURCES:
        statuses = {s for (src, _, _), (_, s, _) in MATRIX.items() if src == source}
        assert statuses <= set(STATUSES)
    assert {s for (_, _, _), (_, s, _) in MATRIX.items()} == set(STATUSES)


@pytest.mark.parametrize(
    "text, known, status, answer",
    [
        # gpt-oss: the control tokens are gone, the transition shows the analysis channel closed.
        ("We need the colour. It is red.assistantfinalC", False, ORPHAN_CLOSE, "C"),
        ("We need the colour. It is red.assistantfinalC", True, CLOSED, "C"),
        (
            "<|channel|>analysis<|message|>Red.<|end|><|start|>assistant<|channel|>final<|message|>C<|return|>",
            False,
            ORPHAN_CLOSE,
            "C",
        ),
        ("<think>a</think>assistantfinalC", False, CLOSED, "C"),
    ],
)
def test_the_harmony_transition_is_a_close(text, known, status, answer):
    got = split_reasoning(comp(text, "truncated"), True if known else None)
    assert (got.answer, got.status, got.blanked) == (answer, status, False)


def test_harmony_cut_inside_analysis_is_graded_whole_and_blanked_when_declared():
    text = "<|channel|>analysis<|message|>We need to compute. Let u ="
    assert split_reasoning(comp(text, "truncated"), True).blanked is True
    assert split_reasoning(comp(text, "truncated")).blanked is False


# --- 3. Policies for the odd shapes ----------------------------------------------------------
class TestPolicies:
    def test_answer_before_close_then_a_reopened_block_that_gives_no_answer_reads_the_text_before_the_close(self):
        # Ministral-3B-Reasoning on POPE: it answers, closes a block it never opened, and starts thinking again.
        got = split_reasoning(comp("yes[/THINK][THINK]I need to determine whether", "truncated"))
        assert (got.answer, got.status, got.answer_before_close) == ("yes", ORPHAN_CLOSE, True)

    def test_answer_before_close_is_not_read_when_reasoning_precedes_the_close(self):
        # Ministral also reasons before [/THINK]; cut after it opens a second block, it gave no answer.
        got = split_reasoning(comp("The circle looks red, so B.[/THINK][THINK]Wait, check the options", "truncated"))
        assert (got.answer, got.status, got.answer_before_close) == ("", ORPHAN_CLOSE, False)
        got = split_reasoning(comp("The circle is red.\nSo it is C.</think><think>wait</think><think>", "doom_loop"))
        assert (got.answer, got.answer_before_close) == ("", False)

    def test_the_real_ministral_reply_with_a_thinking_block_and_an_answer_wrapper(self):
        reply = "yes[/THINK][THINK]I need to determine whether the image has a red circle.</thinking>\n<answer>\nyes\n</answer>"
        got = split_reasoning(comp(reply))
        assert got.answer == "yes"
        assert got.status == ORPHAN_CLOSE
        # The thinking-tag rules of main grade this empty.
        assert remove_thinking_tags(reply) == ""

    def test_answer_before_close_is_not_read_when_the_opener_was_known(self):
        got = split_reasoning(comp("reasoning</think><think>more", "truncated"), opens_reasoning=True)
        assert got.answer == ""
        assert got.answer_before_close is False

    def test_answer_before_close_is_not_read_when_a_visible_answer_follows(self):
        assert split_reasoning(comp("yes[/THINK]no")).answer == "no"

    def test_a_later_stray_close_is_text_and_what_precedes_it_is_reasoning(self):
        assert split_reasoning(comp("a</think>b</think>c")).answer == "b</think>c"
        assert split_reasoning(comp("<think>a</think>b</think>c")).answer == "c"

    def test_a_second_block_after_the_answer_is_dropped(self):
        assert split_reasoning(comp("<think>a</think>ans<think>more")).answer == "ans"
        assert split_reasoning(comp("<think>a</think>ans<think>more</think>")).answer == "ans"

    def test_an_opener_inside_a_block_is_part_of_it(self):
        assert split_reasoning(comp("<think>a<think>b</think>c")).answer == "c"

    def test_the_two_tag_families_close_each_other(self):
        assert split_reasoning(comp("[THINK]a</think>C")).answer == "C"
        assert split_reasoning(comp("<think>a[/THINK]C")).answer == "C"

    def test_tags_are_case_insensitive(self):
        assert split_reasoning(comp("<THINK>a</Think>C")).answer == "C"
        assert split_reasoning(comp("a[/think]C")).answer == "C"

    def test_unclosed_stop_grades_the_last_complete_answer_wrapper_when_the_opener_is_known_or_visible(self):
        text = "Red, so C. <answer>B</answer> no wait <answer>C</answer>"
        got = split_reasoning(comp(text), opens_reasoning=True)
        assert (got.answer, got.status) == ("C", UNCLOSED_STOP)
        got = split_reasoning(comp("<think>Red. <answer>C</answer>"))
        assert (got.answer, got.status) == ("C", UNCLOSED_STOP)

    def test_unclosed_stop_without_a_wrapper_is_graded_as_the_thinking_tag_rules_grade_it(self):
        assert split_reasoning(comp("Red, so C."), opens_reasoning=True).answer == "Red, so C."
        assert split_reasoning(comp("<think>Red, so C.")).answer == ""

    def test_an_answer_wrapper_is_not_read_without_a_known_or_visible_opener_or_when_the_run_was_cut(self):
        text = "Red, so <answer>C</answer> it is"
        assert split_reasoning(comp(text)).answer == text
        assert split_reasoning(comp(text), True).answer == text  # enable_thinking alone
        assert split_reasoning(comp(text, "truncated"), opens_reasoning=True).answer == text

    def test_the_answer_wrapper_is_peeled_after_reasoning_only(self):
        assert split_reasoning(comp("r</think>\n<answer>\nC\n</answer>")).answer == "C"
        assert split_reasoning(comp("<answer>C</answer>")).answer == "<answer>C</answer>"

    def test_empty_and_whitespace_completions(self):
        for text in ("", "  \n\t "):
            got = split_reasoning(comp(text, "truncated"), True, opens_reasoning=True)
            assert (got.answer, got.blanked, got.status) == ("", False, UNCLOSED_CUT)
            assert split_reasoning(comp(text)).status == NO_REASONING

    def test_unicode_and_whitespace_around_the_tags(self):
        assert split_reasoning(comp("\u00a0r\n</think>\u00a0\nC\u00a0")).answer == "C"
        assert split_reasoning(comp("r\r\n</think>\r\nC\r\n")).answer == "C"
        assert (
            split_reasoning(comp("\u25c1think\u25b7\u8d64\u2615\u25c1/think\u25b7\u7b54\u3048: C")).answer
            == "\u7b54\u3048: C"
        )
        assert split_reasoning(comp("<think>\U0001f600</think>\U0001f600C")).answer == "\U0001f600C"

    def test_the_strict_view_needs_text_and_a_cut_and_no_close(self):
        assert split_reasoning(comp("", "truncated"), True).blanked is False
        assert split_reasoning(comp("<think>x", "eos")).blanked is False
        assert split_reasoning(comp("<think>x", "failure")).blanked is False
        assert split_reasoning(comp("<think>x", None)).blanked is False
        assert split_reasoning(comp("<think>x", "truncated")).blanked is True
        assert split_reasoning(comp("<think>x</think>", "truncated")).blanked is False
        assert split_reasoning(comp("x", "truncated"), False).blanked is False

    def test_the_strict_view_does_not_need_the_run_to_record_enable_thinking(self):
        for thinking in (None, False, True):
            assert split_reasoning(comp("<think>x", "truncated"), thinking).blanked is True


# --- 4. Cost ---------------------------------------------------------------------------------
_RUNAWAYS = {
    "opens": "<think>" * 40_000,
    "closes": "</think>" * 40_000,
    "pairs": "<think>x</think>" * 20_000,
    "mixed": "[THINK]a</think>b[/THINK]" * 12_000,
    "unclosed": "<think>" + "word " * 80_000,
    "gemma": "<|channel>thought\n" * 15_000,
    "answers": "<answer>" * 30_000 + "</answer>" * 30_000,
    "wrappers": "<response>" * 30_000 + "x",
    "text-markers": "<|START_TEXT|>" * 30_000,
    "harmony": "assistant<|channel|>" * 15_000,
    "spaces": " " * 300_000 + "<|end|>",
    "kimi": "\u25c1think\u25b7" * 20_000,
}


@pytest.mark.parametrize("name", _RUNAWAYS)
def test_a_runaway_completion_is_read_in_linear_time(name):
    text = _RUNAWAYS[name]
    start = time.perf_counter()
    for opens in (False, True):
        got = split_reasoning(comp(text, "doom_loop"), opens_reasoning=opens)
        assert got.status in STATUSES
    assert time.perf_counter() - start < 2.0, f"{name} took {time.perf_counter() - start:.1f}s"


# --- VLM markup: what the VLM evals rewrite beyond the text evals' normalize_reasoning_markup -------
_VLM_MARKUP = [
    # Kimi-VL-A3B-Thinking-2506 on llama.cpp (b11744 smoke): the tags are plain text in the vocabulary, so
    # the server prints them. The first two are the model's replies to the smoke's MCQ and OCR questions.
    pytest.param(
        "◁think▷Got it, let's look at the image. The circle is red. So the answer is C.◁/think▷C",
        "C",
        id="kimi-closed",
    ),
    pytest.param(
        "◁think▷So, let's look at the image. The text in the image is \"PIPELINE 4\". Let's check each part."
        "◁/think▷PIPELINE 4",
        "PIPELINE 4",
        id="kimi-closed-ocr",
    ),
    # The opener came before the part the server kept (a smoke reply that starts mid-thought): the
    # orphan close still cuts through.
    pytest.param(
        "5\"? Wait, no, no, wait. Wait, no, the number is 2075. Yes, that's the number.◁/think▷2075",
        "2075",
        id="kimi-opener-missing",
    ),
    # Cut inside the thought: nothing, as Kimi's own extract_summary gives.
    pytest.param("◁think▷Got it, the circle is red, so the answer is C, but wait", "", id="kimi-unclosed"),
    pytest.param("C◁think▷and then more thought", "C", id="kimi-unclosed-after-text"),
    pytest.param("◁think▷Red.◁/think▷Answer: $\\boxed{C}$", "Answer: $\\boxed{C}$", id="kimi-boxed-summary"),
    pytest.param("<thinking>The circle is red.</thinking>C", "C", id="thinking-tags"),
]


@pytest.mark.parametrize("raw, expected", _VLM_MARKUP)
def test_the_vlm_markup_rewrite(raw, expected):
    once = normalize_vlm_reasoning_markup(raw)
    assert remove_thinking_tags(once) == expected
    assert split_reasoning(comp(raw)).answer == expected
    assert normalize_vlm_reasoning_markup(once) == once


@pytest.mark.parametrize(
    "raw, blanked",
    [
        # Kimi-VL cut inside its thought, like Gemma 4's; once its close is rewritten it is a
        # reasoning close, so an answer cut after it is graded as is.
        ("◁think▷Compute 17*23: 17*20 = 340", True),
        ("◁think▷Red.◁/think▷The answer is", False),
    ],
)
def test_a_cut_kimi_reply_is_blanked_only_inside_its_thought(raw, blanked):
    assert split_reasoning(comp(raw, "truncated"), True).blanked is blanked


# --- 5. Never raises, never returns reasoning that a close ended ----------------------------------
_TOKENS = [
    "<think>",
    "</think>",
    "[THINK]",
    "[/THINK]",
    "\u25c1think\u25b7",
    "\u25c1/think\u25b7",
    "<thinking>",
    "</thinking>",
    "<|START_THINKING|>",
    "<|END_THINKING|>",
    "<|channel>thought\n",
    "<channel|>",
    "assistantfinal",
    "<answer>",
    "</answer>",
    "<response>",
    "</response>",
    "<|START_TEXT|>",
    "<|END_TEXT|>",
    "<|end|>",
    "",
    " ",
    "\n",
    "\u00a0",
]


def test_any_completion_is_read_without_raising_and_to_a_status():
    rng = random.Random(7)
    for _ in range(6000):
        text = "".join(rng.choice(_TOKENS + ["word", "yes", "C"]) for _ in range(rng.randint(0, 12)))
        stop = rng.choice(["eos", "truncated", "doom_loop", None, "failure"])
        thinking = rng.choice([None, True, False])
        opens = rng.random() < 0.3
        got = split_reasoning(comp(text, stop), thinking, opens_reasoning=opens)
        assert got.status in STATUSES
        assert isinstance(got.answer, str) and got.answer == got.answer.strip()
        assert got.blanked <= (got.status == UNCLOSED_CUT)


def test_reasoning_that_a_close_ended_is_never_returned():
    """A reply made of reasoning, a close, then an answer never returns the reasoning, whoever opened it."""
    rng = random.Random(11)
    for _ in range(4000):
        family = rng.choice(list(FAMILIES))
        o, c = FAMILIES[family]
        reasoning = " ".join(f"REASON{rng.randint(0, 99)}" for _ in range(rng.randint(1, 6)))
        answer = " ".join(f"ANS{rng.randint(0, 99)}" for _ in range(rng.randint(1, 3)))
        # Gemma's close with no opener is a stray marker, so its opener is always in the text.
        visible_opener = family == "gemma" or rng.random() < 0.5
        text = (o if visible_opener else "") + reasoning + c + answer
        for thinking, opens in ((None, False), (True, False), (None, True)):
            got = split_reasoning(comp(text, rng.choice(["eos", "truncated"])), thinking, opens_reasoning=opens)
            assert "REASON" not in got.answer, (text, got)
            assert got.answer.startswith("ANS"), (text, got)
            assert got.blanked is False


# --- 6. Real replies (smoke runs of llama.cpp b11744) -----------------------------------------
REAL = {
    "kimi-mcq": (
        "\u25c1think\u25b7Got it, let's look at the image. The circle is red. So the answer is C.\u25c1/think\u25b7C",
        "C",
    ),
    "kimi-opener-missing": ('5"? Wait, no, no, wait. Wait, no, the number is 2075.\u25c1/think\u25b72075', "2075"),
    "mimo-mcq": (
        "<think>\nGot it, let's look at the image. There's a red circle on the left. The options are A to D. So the circle is red, "
        "which is option C.\n</think>\n\n\\boxed{C}",
        "\\boxed{C}",
    ),
    "mimo-ocr": ('<think>\nThe text is "PIPETTE 42".\n</think>\n\n\\boxed{PIPETTE 42}', "\\boxed{PIPETTE 42}"),
    "ministral-pope": ('So the answer should be "yes".[/THINK]yes', "yes"),
    "ministral-describe": (
        "It shows the front page.[/THINK]The image is an old newspaper.",
        "The image is an old newspaper.",
    ),
}


@pytest.mark.parametrize("name", REAL)
def test_real_replies(name):
    reply, answer = REAL[name]
    assert split_reasoning(comp(reply)).answer == answer


@pytest.mark.parametrize(
    "name, reply, thinking",
    [
        # Reasoning the server already split off: the content is the answer alone, whatever model wrote it.
        ("qwen3.5-9b", "C", True),
        ("qwen3.6-35b", "C", True),
        ("gemma-4-26b", "C", True),
        ("nemotron-3-omni", "C", True),
        ("nemotron-12b", "C\n", True),
        (
            "minicpm-o-4.5",
            "To determine the color of the circle, observe the image: the circle on the left is red. Among the options, option C is Red.  \n\n\nC",
            True,
        ),
        ("step3", "C", None),
    ],
)
def test_replies_the_server_already_split_are_graded_whole_and_flagged_unclosed_stop_when_thinking_was_declared(
    name, reply, thinking
):
    got = split_reasoning(comp(reply), thinking)
    assert got.answer == reply.strip()
    assert got.status == (UNCLOSED_STOP if thinking else NO_REASONING)
    assert got.blanked is False


def test_the_scorers_grade_the_real_replies():
    samples = [VlmSample(id="1", prompt="q", answer=None, raw_doc=_MCQ)]
    for name in ("kimi-mcq", "mimo-mcq"):
        _, ctx = mmstar.score([comp(REAL[name][0])], samples)
        assert ctx["accuracy"] == 1.0, name
        assert ctx["reasoning_status"][CLOSED] == 1
    pope_sample = VlmSample(id="1", prompt="q", answer=None, raw_doc={"answer": "Yes", "category": "adversarial"})
    scores, ctx = pope.score([comp(REAL["ministral-pope"][0])], [pope_sample])
    assert scores[0].is_correct is True
    assert ctx["reasoning_status"][ORPHAN_CLOSE] == 1
    ocr = VlmSample(
        id="1", prompt="q", answer=None, raw_doc={"answer": ["PIPETTE 42"], "category": "Regular Text Recognition"}
    )
    scores, _ = ocrbench.score([comp(REAL["mimo-ocr"][0])], [ocr])
    assert scores[0].is_correct is True


# --- 7. What the scorers report ----------------------------------------------------------------
def test_the_tally_counts_statuses_and_answers_read_before_a_close():
    tally = ReasoningTally()
    for text, stop in [
        ("a</think>C", "eos"),
        ("<think>a</think>C", "eos"),
        ("<think>a", "truncated"),
        ("C", "eos"),
        ("yes[/THINK][THINK]x", "truncated"),
    ]:
        tally.add(split_reasoning(comp(text, stop)))
    assert tally.context() == {
        "reasoning_status": {CLOSED: 1, ORPHAN_CLOSE: 2, UNCLOSED_CUT: 1, UNCLOSED_STOP: 0, NO_REASONING: 1},
        "reasoning_answer_before_close": 1,
    }


def test_a_scorer_reports_unclosed_stops_and_the_strict_count_without_enable_thinking():
    samples = [VlmSample(id=str(i), prompt="q", answer=None, raw_doc=_MCQ) for i in range(3)]
    completions = [
        SampleCompletion(id="0", completion="<think>The circle is red, so C", stop_reason="truncated"),
        SampleCompletion(id="1", completion="<think>The circle is red, so C", stop_reason="eos"),
        SampleCompletion(id="2", completion="r</think>C", stop_reason="eos"),
    ]
    _, ctx = mmstar.score(completions, samples)
    assert ctx["blanked_truncated"] == 1
    assert ctx["reasoning_status"] == {CLOSED: 0, ORPHAN_CLOSE: 1, UNCLOSED_CUT: 1, UNCLOSED_STOP: 1, NO_REASONING: 0}


def test_every_registered_opening_profile_is_a_profile_and_the_facade_passes_it_on():
    assert prompt_profiles.REASONING_OPENERS <= set(prompt_profiles.PROFILES)
    samples = [VlmSample(id="1", prompt="q", answer=None, raw_doc=_MCQ)]
    cut = [comp("The circle is red, so the answer is C. Let me double", "truncated")]
    for profile, blanked in (
        ("phi4_rv_think", 1),
        ("nemotron_think", 1),
        ("kimi_vl", 1),
        ("mimo_vl", 1),
        # Its R1 system prompt has the model write <think> itself: counted only with enable_thinking, as on main.
        ("internvl_r1", 0),
        ("minicpm_v", 0),
        (None, 0),
    ):
        _, ctx = score(EvalId.MMSTAR, cut, samples, prompt_profile=profile)
        assert ctx["blanked_truncated"] == blanked, profile


def test_the_text_evals_do_not_read_the_parser():
    """Only the VLM scorers read a profile's opener: a text eval scores and counts as on main."""
    from pipette_scores.types import GPQADiamondSample

    sample = GPQADiamondSample(id="1", prompt="Q\nA) a\nB) b\nC) c\nD) d", answer="B")
    cut = [comp("Option B fits. Answer: B. Double-check option", "truncated")]
    _, with_profile = score(EvalId.GPQA_DIAMOND, cut, [sample], prompt_profile="phi4_rv_think")
    _, without = score(EvalId.GPQA_DIAMOND, cut, [sample])
    assert with_profile == without
    assert with_profile["gpqa_diamond_blanked_truncated"] == 0
    assert "gpqa_diamond_reasoning_status" not in with_profile
    # Nor the VLM markup rewrite.
    for raw in ("◁think▷Red.◁/think▷C", "<thinking>Red.</thinking>C"):
        assert normalize_reasoning_markup(raw) == raw


def test_realworldqa_and_pope_report_the_tally_too():
    sample = VlmSample(id="1", prompt="q", answer=None, raw_doc={"A": "x", "B": "y", "answer": "B"})
    _, ctx = realworldqa.score([comp("r</think>B")], [sample])
    assert ctx["reasoning_status"][ORPHAN_CLOSE] == 1
    pope_sample = VlmSample(id="1", prompt="q", answer=None, raw_doc={"answer": "Yes", "category": "adversarial"})
    _, ctx = pope.score([comp("r</think>yes")], [pope_sample])
    assert ctx["reasoning_status"][ORPHAN_CLOSE] == 1
