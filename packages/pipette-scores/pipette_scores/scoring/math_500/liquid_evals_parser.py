"""liquid_evals' MATH-500 answer extraction and grading, vendored verbatim.

Source: Liquid4All/liquid_evals, liquid_evals/parser.py at commit
d5534cfa17713b9fb2d1fe6626763d411380e9a8 (blob cd04c90839e366eb51d1071166d3727dcdcf0c99).
The plain `tasks/math500.py` task grades with these functions; the scorer here
calls them in the same order (see pipette_scores.scoring.math_500).

What is copied, unchanged:
- parser.py:1-3     the stdlib imports the functions below use;
- parser.py:111-348 extract_math_answer and its _extract_* helpers, and
                    verify_math_answer (the last-150-lines fallback);
- parser.py:351-798 the mathd/sympy graders behind check_math_result, with their
                    imports and the pylatexenc log-level line.

Not copied: template_prefills_thinking, compute_length_metadata, and
remove_thinking_tags (pipette_scores.scoring.completion_text already carries it).
The grader code descends from openai/prm800k's grader (MIT) and Hendrycks'
MATH answer normalization (MIT); see NOTICE and THIRD_PARTY_LICENSES.md.

Do not edit the code below: tests/test_math_500_liquid_evals_parity.py checks it
against outputs of the original. To update it, copy the new upstream lines and
regenerate the parity fixture with scripts/gen_math_500_liquid_evals_parity.py.
"""

import functools
import logging
import re


def extract_math_answer(text):
    """
    Comprehensive answer extraction function that tries multiple patterns in order of priority.

    Priority order:
    1. \\boxed{} pattern
    2. ### or #### patterns
    3. "ANSWER:" pattern
    4. "answer is {}" pattern
    5. Other patterns (final answer, numeric fallback)

    Args:
        text (str): Text to search for answer content

    Returns:
        str: Extracted answer content or empty string if no pattern found
    """
    if not isinstance(text, str):
        return ""

    # 1. First priority: \\boxed{} pattern
    boxed_answer = _extract_boxed_pattern(text)
    if boxed_answer:
        return boxed_answer

    # 2. Second priority: ### or #### patterns
    hash_answer = _extract_hash_pattern(text)
    if hash_answer:
        return hash_answer

    # 3. Third priority: "ANSWER:" pattern
    answer_colon = _extract_answer_colon_pattern(text)
    if answer_colon:
        return answer_colon

    # 4. Fourth priority: "answer is {}" pattern
    answer_is = _extract_answer_is_pattern(text)
    if answer_is:
        return answer_is

    # 5. Fifth priority: Other patterns
    other_answer = _extract_other_patterns(text)
    if other_answer:
        return other_answer

    # 6. Final fallback: numeric extraction from end of text
    numeric_answer = _extract_numeric_fallback(text)
    if numeric_answer:
        return numeric_answer

    return ""


def _extract_boxed_pattern(text):
    """Extract content from \\boxed{} pattern with proper brace handling."""
    matches = []
    i = 0
    while i < len(text):
        # Look for \boxed{ (single backslash)
        boxed_start = text.find("\\boxed{", i)
        if boxed_start == -1:
            break

        content_start = boxed_start + 7  # Length of "\\boxed{"
        if content_start >= len(text):
            break

        # Count braces to find the matching closing brace
        brace_count = 1
        j = content_start
        while j < len(text) and brace_count > 0:
            if text[j] == "{":
                brace_count += 1
            elif text[j] == "}":
                brace_count -= 1
            j += 1

        # If we found the matching closing brace
        if brace_count == 0:
            content = text[content_start : j - 1]
            matches.append(content.strip())

        # Continue searching from the end of this pattern
        i = j

    if matches:
        return matches[-1]  # Return the last match
    return ""


def _extract_hash_pattern(text):
    """Extract content from ### or #### patterns."""
    # Try #### pattern first (more specific)
    quad_hash_pattern = r"####\s*([^#]+?)\s*(?=\n|$)"
    quad_matches = re.findall(quad_hash_pattern, text, re.IGNORECASE)
    if quad_matches:
        return quad_matches[-1].strip()

    # Try ### pattern
    triple_hash_pattern = r"###\s*([^#]+?)\s*(?=\n|$)"
    triple_matches = re.findall(triple_hash_pattern, text, re.IGNORECASE)
    if triple_matches:
        return triple_matches[-1].strip()

    return ""


def _extract_answer_colon_pattern(text):
    """Extract content from 'ANSWER:' pattern."""
    answer_pattern = r"ANSWER\s*:\s*([^\n]+?)(?=\n|$)"
    matches = re.findall(answer_pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return ""


def _extract_answer_is_pattern(text):
    """Extract content from 'answer is {}' patterns."""
    # Pattern for "the answer is [content]" or "answer is [content]"
    answer_is_patterns = [
        r"(?:the\s+)?answer\s+is\s*\[([^\]]+)\]",  # answer is [content]
        r"(?:the\s+)?answer\s+is\s*\{([^}]+)\}",  # answer is {content}
        r"(?:the\s+)?answer\s+is\s*\(([^)]+)\)",  # answer is (content)
        r"(?:the\s+)?answer\s+is\s+([^\n.!?]+)",  # answer is content
    ]

    for pattern in answer_is_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    return ""


def _extract_other_patterns(text):
    """Extract content from other common answer patterns."""
    # Try "The final answer is [answer]" pattern
    final_answer_pattern = r"The final answer is\s*\[?([^\]]+?)\]?\s*(?=\n|$)"
    final_answer_matches = re.findall(final_answer_pattern, text, re.IGNORECASE)
    if final_answer_matches:
        return final_answer_matches[-1].strip()

    # Try "Final answer:" pattern
    final_colon_pattern = r"Final answer\s*:\s*([^\n]+?)(?=\n|$)"
    final_colon_matches = re.findall(final_colon_pattern, text, re.IGNORECASE)
    if final_colon_matches:
        return final_colon_matches[-1].strip()

    # Try simple "Answer:" at start of line
    simple_answer_pattern = r"^Answer\s*:\s*([^\n]+?)(?=\n|$)"
    simple_matches = re.findall(
        simple_answer_pattern, text, re.IGNORECASE | re.MULTILINE
    )
    if simple_matches:
        return simple_matches[-1].strip()

    return ""


def _extract_numeric_fallback(text):
    """Extract numeric answer as fallback from the end of text."""
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line:
            # Look for numbers in the last non-empty line
            numbers = re.findall(r"-?\d+(?:\.\d+)?", line)
            if numbers:
                return numbers[-1]
    return ""


def verify_math_answer(true_label, predicted_label, *fallback_sources):
    """Return True when the predicted answer matches the true answer.

    Falls back to the sympy/mathd based checker when direct string equality
    fails. ``fallback_sources`` may contain additional strings (e.g. the raw
    model response) that the math parser can inspect.
    """

    def _is_non_empty_string(value):
        return isinstance(value, str) and value.strip() != ""

    truth = true_label.strip() if isinstance(true_label, str) else ""

    # Special case: if ground truth is "None" and predicted is empty, it's correct
    if truth.lower() == "None" and not _is_non_empty_string(predicted_label):
        return True

    if not truth:
        return False

    if _is_non_empty_string(predicted_label) and predicted_label.strip() == truth:
        return True

    candidates = []
    if _is_non_empty_string(predicted_label):
        candidates.append(predicted_label.strip())

    for source in fallback_sources:
        if not _is_non_empty_string(source):
            continue
        stripped = source.strip()
        cleaned = re.sub(
            r"^(?:final answer:|the final answer is|answer:|###)\s*",
            "",
            stripped,
            flags=re.IGNORECASE,
        ).strip()
        if cleaned and cleaned != stripped:
            candidates.append(cleaned)
        if "\n" in stripped:
            # Limit the depth of line-based fallback parsing to speed up the process
            MAX_FALLBACK_LINES = 150
            for n_lines, line in enumerate(reversed(stripped.splitlines()), start=1):
                line = line.strip()
                if _is_non_empty_string(line):
                    candidates.append(line)
                    cleaned_line = re.sub(
                        r"^(?:final answer:|the final answer is|answer:|###)\s*",
                        "",
                        line,
                        flags=re.IGNORECASE,
                    ).strip()
                    if cleaned_line and cleaned_line != line:
                        candidates.append(cleaned_line)
                    if n_lines >= MAX_FALLBACK_LINES:
                        break

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if check_math_result(candidate, truth):
            return True

    return False


from typing import Optional

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

# Suppress noisy pylatexenc parse warnings that flood output on malformed LaTeX
logging.getLogger("pylatexenc").setLevel(logging.ERROR)


# Dan Hendrycks' code
def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer


def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string

    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace(r"\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


@functools.lru_cache(maxsize=4096)
def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")

    # pylatexenc can blow up on deeply nested macros;
    # anything beyond 100 LaTeX commands is hallucinated output anyway
    if len(re.findall(r"\\[a-zA-Z]+", expr)) > 100:
        raise ValueError("Too many LaTeX macros for pylatexenc")

    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile(r"([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        # Very long expressions may freeze sympy simplification and trigger OOM.
        MAX_SYMPY_EXPR_LEN = 3000
        if len(expr) > MAX_SYMPY_EXPR_LEN:
            return False
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False


def check_math_result(model_response, solution):
    result_mathd = grade_answer_mathd(model_response, solution)
    result_sympy = grade_answer_sympy(model_response, solution)
    return result_mathd or result_sympy
