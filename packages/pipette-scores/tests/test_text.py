import pytest

from pipette_scores.scoring.completion_text import remove_thinking_tags


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
