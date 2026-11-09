"""The deterministic fallbacks after VLMEvalKit's exact matching (vlm_answer)."""

import pytest

import pipette_scores.scoring as scoring
from pipette_scores.scoring.vlm_answer import option, option_hit, yes_no
from pipette_scores.types import SampleCompletion, VlmSample

ABCD = {"A": "Left side", "B": "Right side", "C": "Center", "D": "Behind"}
ABCDEF = {**ABCD, "E": "E", "F": "F"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("B", "B"),
        ("A. Left side", "A"),
        ("The answer is C.", "C"),
    ],
)
def test_an_answer_upstream_reads_is_scored_as_upstream_scores_it(text, expected):
    assert option(text, ABCD) == (expected, False)


@pytest.mark.parametrize(
    ("text", "choices", "expected"),
    [
        # Verbose answers that mention other letters before committing to one.
        (
            "He stands near A, with B and C to his right.\n\nSo, the correct answer is:  \n**A. Left side** ✅",
            ABCD,
            "A",
        ),
        (
            "Only **D** matches, not A or B.\n\nThus, the correct answer is:\n\n> **D. D**\n\n✅ Final Answer: **D**",
            ABCD,
            "D",
        ),
        # The last stated answer wins over an earlier quoted option.
        ("Columns C, F, B, D, A grow.\n\n> **E**\n\n---\n\n✅ Correct Answer: **A. E**", ABCDEF, "A"),
        ("Not B or D; the houses are small.\n\n### Correct Answer: **C. Center** ✅", ABCD, "C"),
        ("A and B are wrong, so the answer is (B), not D.", ABCD, "B"),
        ("Between A and D, I pick\n\\boxed{C}", ABCD, "C"),
        ("A or B?\n\\boxed{\\text{B}}", ABCD, "B"),
        # The last line alone, when nothing is stated.
        ("Options A, B and C look close, as does D.\nC.", ABCD, "C"),
    ],
)
def test_a_committed_answer_is_read_by_the_fallback(text, choices, expected):
    assert option(text, choices) == (expected, True)


@pytest.mark.parametrize(
    "text",
    [
        "The answer could be a dog, which neither A nor B shows.",
        "Answer: All of the above, since A and B both apply.",
        "Final answer: E, as A and B are wrong.",
        "It could be A or B; I am not sure.",
    ],
)
def test_no_letter_is_guessed(text):
    assert option(text, ABCD) == (None, False)


@pytest.mark.parametrize(
    "text",
    [
        '{"answer": "C"}',
        'The middle one.\n\n```json\n{\n  "answer": "C"\n}\n```',
        "answer: 'C'",
        'They are students in the same course.\n\n`answer`: "C"',
        '`answer` "C"',
        'A Pizza Hut serves pizza.\n\n`answer: "C"`',
    ],
)
def test_a_json_answer_field_is_read(text):
    # Qwen3.5's model card asks for `"answer": "C"` on multiple-choice questions.
    assert option(text, ABCD) == ("C", True)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('"B"', "B"),
        ("“A”", "A"),
        ("‘D’.", "D"),
        ('"a"', "A"),
        ('"B', "B"),
        ("The lane on the right, past the barrier.\n\n'C'", "C"),
    ],
)
def test_the_answer_fields_value_alone_is_read(text, expected):
    # Qwen3.5-4B often answers with the field's value and no key: `"B"`.
    assert option(text, ABCD) == (expected, True)


@pytest.mark.parametrize(
    "text",
    [
        '"E"',
        '"B" or "C"',
        "I'm not sure about A or B.",
        '"Bus"',
        '"A man stands left.',
        '"I think so."',
        # Review cases: the format described or refused, alternatives, a quoted article.
        "The required format is `answer`: A, B, C, or D. Unable to decide.",
        "Do not output `answer`: A or B.",
        'Either "A" or\n"B".',
        'The sign on the shop reads\n"a"',
    ],
)
def test_a_quoted_word_or_two_letters_is_not_an_answer(text):
    assert option(text, ABCD) == (None, False)


def test_the_last_commitment_wins_across_statements_and_fields():
    assert option('A and C were considered. Final answer: **A**.\n`answer`: "B"', ABCD) == ("B", True)
    assert option('`answer`: "B"\nA and B fall short.\nFinal Answer: **C**', ABCD) == ("C", True)


def test_a_backticked_yes_no_field_is_read():
    assert yes_no('Both yes and no were considered.\n`answer`: "no"\nThat is my conclusion.') == ("No", True)


def test_a_json_yes_no_answer_field_is_read():
    # Mentions both words, so upstream's rule reads nothing and the field decides.
    assert yes_no('Yes, there is a dog, but no cat.\n{"answer": "no"}') == ("No", True)


@pytest.mark.parametrize(
    "text",
    [
        "I cannot answer yes or no, as the image is missing.",
        "The view is unclear. Therefore, I cannot answer yes or no with certainty.",
        "Without more context I can't answer yes/no.",
    ],
)
def test_a_reply_that_cannot_choose_is_not_read(text):
    # Found by the VLMEvalKit cross-check: the statement rule read the "yes" of
    # "answer yes or no" as the answer.
    assert yes_no(text) == ("Unknown", False)


def test_a_quoted_option_line_is_read():
    # Upstream reads neither: both options' texts occur in "Not the same".
    same = {"A": "The same", "B": "Not the same"}
    assert option('The two shapes differ in size.\n\n"B. Not the same"', same) == ("B", True)
    # Quoted prose that names an option is not an answer.
    assert option('"B. Not the same" is one of the candidates, but no selection is possible.', same) == (
        None,
        False,
    )


def test_upstreams_own_reading_stands():
    # Upstream's case-insensitive "answer is X" reads an article as option A.
    # A quirk of the vendored rule, kept as upstream scores it.
    assert option("The answer is a dog, which neither A nor B shows.", ABCD) == ("A", False)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Answer first, explained after (InternVL3.5's shape).
        (
            "B. Right side\n\nThe man stands near the right edge of the frame, beside a lamp post, while the left side and the center of the stage are empty apart from a drum kit and some amplifiers, and nothing is behind him.",
            "B",
        ),
        (
            "**D**\n\nThe man stands near the right edge of the frame, beside a lamp post, while the left side and the center of the stage are empty apart from a drum kit and some amplifiers, and nothing is behind him.",
            "D",
        ),
        (
            "(C) Center\n\nThe man stands near the right edge of the frame, beside a lamp post, while the left side and the center of the stage are empty apart from a drum kit and some amplifiers, and nothing is behind him.",
            "C",
        ),
        # A line that is exactly an option, after a lead-in.
        (
            "Where he stands:\n\nA. Left side\n\nThe man stands near the right edge of the frame, beside a lamp post, while the left side and the center of the stage are empty apart from a drum kit and some amplifiers, and nothing is behind him.",
            "A",
        ),
    ],
)
def test_an_answer_given_first_is_read(text, expected):
    assert option(text, ABCD) == (expected, True)


@pytest.mark.parametrize(
    "text",
    [
        # A walk through several options commits to none of them.
        "A. Left side: possible, but B is likelier.\nB. Right side: also possible.\nHard to say.",
        # A line naming an option whose text is not that option's.
        "The scene:\n\nA. a dog on the left\n\nBut B and C are unclear.",
    ],
)
def test_a_walk_through_or_a_mismatched_line_is_not_read(text):
    assert option(text, ABCD) == (None, False)


def test_a_yes_or_no_given_first_is_read():
    assert yes_no("Yes, there is a dog. There is no cat.") == ("Yes", True)
    assert yes_no("**No**. The sign says yes, but no dog is visible.") == ("No", True)


def test_option_hit_compares_with_the_ground_truth():
    text = "Not B.\n\nThe correct answer is: **A**"
    assert option_hit(text, ABCD, "A") == (True, True)
    assert option_hit(text, ABCD, "B") == (False, True)
    assert option_hit("No idea, A or B.", ABCD, "A") == (False, False)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Yes", ("Yes", False)),
        ("No, there is none.", ("No", False)),
        ("Is there no dog? There is one, so yes.\n\nFinal Answer: **Yes**", ("Yes", True)),
        ("Yes, a car; no, not a bus.\n\\boxed{No}", ("No", True)),
        ("Is there a car? Yes, but no bus is visible.\nNo.", ("No", True)),
        ("Yes and no.", ("Unknown", False)),
    ],
)
def test_yes_no(text, expected):
    assert yes_no(text) == expected


def test_a_reasoning_model_is_read_after_its_reasoning():
    sample = VlmSample(
        id="1", prompt="q", answer=None, raw_doc={**ABCD, "answer": "B", "category": "c", "l2_category": "x"}
    )
    completion = SampleCompletion(
        id="1",
        completion="<think>A? C? It is not D.</think>\nLooking again, A and C are wrong.\n\nFinal Answer: **B**",
        stop_reason="eos",
    )
    scores, ctx = scoring.score("mmstar", [completion], [sample], enable_thinking=True)
    assert scores[0].is_correct
    assert ctx["fallback_answers"] == 1
