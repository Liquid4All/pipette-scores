"""Completion-text preprocessing shared across scorers."""

import re


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
