#!/usr/bin/env python
"""Regenerate the liquid_evals MATH-500 grader parity fixture.

Runs the ORIGINAL liquid_evals parser on the cases below and records its
outputs in tests/fixtures/math_500_liquid_evals_parity.json, which
tests/test_math_500_liquid_evals_parity.py checks the vendored copy
(pipette_scores.scoring.math_500.liquid_evals_parser) against. Run from the
repo root against a liquid_evals clone; the clone is only read (`git show`):

    uv run --project packages/pipette-scores \\
        python packages/pipette-scores/scripts/gen_math_500_liquid_evals_parity.py \\
        --liquid-evals /path/to/liquid_evals --rev d5534cf

The original parser needs only sympy and pylatexenc, which this package pins
to the versions liquid_evals locks.
"""

import argparse
import importlib.util
import json
import pathlib
import subprocess
import tempfile

PARSER_PATH = "liquid_evals/parser.py"
FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "math_500_liquid_evals_parity.json"

# extract_math_answer inputs: each pattern in its priority order, and the
# brace counting of the \boxed{} scan.
EXTRACT_CASES = [
    "The answer is \\boxed{42}.",
    "\\boxed{\\frac{1}{2}}",
    "\\boxed{\\frac{\\sqrt{3}}{2}}",
    "first \\boxed{3}, then corrected: \\boxed{5}",
    "\\boxed{\\boxed{7}}",
    "\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}",
    "\\boxed{x \\in [-2, 7]}",
    "unbalanced \\boxed{3 so the result is 4",
    "\\boxed{}",
    "empty box \\boxed{} then 12 apples",
    "\\boxed {9} has a space",
    "\\fbox{5} is not a boxed pattern",
    "\\boxed{5\\text{ cm}}",
    "\\boxed{90^\\circ}",
    "$\\boxed{\\$18.90}$",
    "Work.\n#### 18",
    "Work.\n### Final Answer\n5",
    "### 6\nmore text\n### 7",
    "ANSWER: 7",
    "answer: x = 3\nmore",
    "The answer is [3, 4]",
    "the answer is {x+1}",
    "So the answer is (B).",
    "Hence the answer is 12. Done",
    "The final answer is 9",
    "Final answer: 2/3",
    "Answer: -4",
    "We get 3.5 apples\n\n",
    "The total is 1,000 dollars",
    "no numbers here",
    "",
    "<think>\\boxed{1}</think>\\boxed{2}",
]

# (given, ground truth) pairs for check_math_result and its parts: mathd and
# sympy equivalence, units, percents, tuples, intervals, text answers, and the
# guards that keep sympy off long or unbounded input.
CHECK_CASES = [
    ("42", "42"),
    ("\\frac{1}{2}", "\\frac12"),
    ("0.5", "\\frac{1}{2}"),
    ("1/2", "\\frac{1}{2}"),
    ("2/4", "\\frac{1}{2}"),
    ("\\dfrac{3}{4}", "\\frac34"),
    ("\\tfrac{3}{4}", "0.75"),
    ("10\\%", "10"),
    ("10%", "10"),
    ("5 \\text{ cm}", "5"),
    ("5\\text{ cm}\\text{ m}", "5"),
    ("5 cm", "5"),
    ("12 inches", "12"),
    ("3 feet", "3"),
    ("90^\\circ", "90"),
    ("90^{\\circ}", "90"),
    ("\\$5", "5"),
    ("$5.00", "5"),
    ("x=3", "3"),
    ("k = 3", "3"),
    ("y = 2x + 1", "2x+1"),
    ("(1,2)", "(1, 2)"),
    ("[1,2)", "(1,2)"),
    ("(1,2,3)", "(1,2)"),
    ("(-\\infty, 3]", "(-\\infty,3]"),
    ("\\left( 3, \\frac{\\pi}{2} \\right)", "\\left( 3, \\frac{\\pi}{2} \\right)"),
    ("(3, \\pi/2)", "\\left( 3, \\frac{\\pi}{2} \\right)"),
    ("1,000", "1000"),
    ("1,000,000", "10^6"),
    ("3 million", "3000000"),
    ("7 3/4", "\\frac{31}{4}"),
    ("\\sqrt{2}/2", "\\frac{\\sqrt{2}}{2}"),
    ("2\\sqrt3", "2\\sqrt{3}"),
    ("2\\sqrt", "2\\sqrt{3}"),
    ("\\frac", "\\frac{1}{2}"),
    ("\\pi", "\\pi"),
    ("2\\pi", "\\pi*2"),
    ("\\infty", "\\infty"),
    ("x^2+2x+1", "(x+1)^2"),
    ("x^{2}+2x+1", "(x+1)^2"),
    ("2^10", "1024"),
    ("1024", "2^{10}"),
    ("abc", "abd"),
    ("\\text{Monday}", "\\text{Monday}"),
    ("monday", "\\text{Monday}"),
    ("Monday", "\\text{(C)}"),
    ("C", "\\text{(C)}"),
    ("(C)", "\\text{(C)}"),
    ("1 and 2", "1, 2"),
    ("1 or 2", "1,2"),
    ("\\{1,2\\}", "1,2"),
    ("10", "10.0"),
    ("3.0", "3"),
    ("-3", "- 3"),
    (".5", "0.5"),
    ("\\!1\\!000", "1000"),
    ("\\frac{1}{3}", "0.333"),
    ("0.333", "\\frac{1}{3}"),
    ("1e400", "1"),
    ("nan", "nan"),
    ("", "5"),
    ("5", ""),
    ("The answer is 4", "4"),
    ("4", "The answer is 4"),
    ("x = \\frac{-b}{2a}", "-\\frac{b}{2a}"),
    ("+".join(["x"] * 1600), "1600x"),
    (" ".join(["\\alpha"] * 101), "1"),
    ("\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}", "\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}"),
]


def _lines(n: int) -> str:
    return "\n".join(f"step {i} of the work" for i in range(n))


# (raw response, gold) for the tasks/math500.py pipeline: remove_thinking_tags,
# extract_math_answer, verify_math_answer. Covers the fallback over the last
# 150 lines and its counting: blank lines count toward the 150, but the cap is
# only checked on a non-blank line.
VERIFY_CASES = [
    ("So the answer is \\boxed{42}.", "42"),
    ("So the answer is \\boxed{41}.", "42"),
    ("\\boxed{\\frac{\\sqrt{3}}{2}}", "\\frac{\\sqrt{3}}{2}"),
    ("\\boxed{\\dfrac{\\sqrt3}{2}}", "\\frac{\\sqrt{3}}{2}"),
    ("\\boxed{\\boxed{7}}", "7"),
    ("\\boxed{5\\text{ cm}}", "5"),
    ("\\boxed{5 \\text{cm}}", "5"),
    ("\\boxed{90^\\circ}", "90^\\circ"),
    ("\\boxed{(3, \\frac{\\pi}{2})}", "\\left( 3, \\frac{\\pi}{2} \\right)"),
    ("\\boxed{\\text{(C)}}", "\\text{(C)}"),
    ("\\boxed{C}", "\\text{(C)}"),
    ("The answer is 7", "7"),
    ("Therefore, the final answer is 12", "12"),
    ("Final answer: 5", "5"),
    ("We compute.\nFinal answer: 5", "5"),
    ("We compute 3 + 4.\nIt is 7 apples", "7"),
    ("no answer here at all", "7"),
    ("", "7"),
    ("Hmm.\nMonday", "\\text{Monday}"),
    ("\\boxed{3}\nThe answer is 4", "4"),
    ("\\boxed{3}\n4", "4"),
    ("4\n" + _lines(148) + "\n\\boxed{3}", "4"),
    ("4\n" + _lines(149) + "\n\\boxed{3}", "4"),
    ("4\n\n" + _lines(148) + "\n\\boxed{3}", "4"),
    ("4\n\n\n" + _lines(148) + "\n\\boxed{3}", "4"),
    ("<think>maybe \\boxed{7}</think>\nSo \\boxed{42}", "42"),
    ("<think>maybe \\boxed{42}</think>\nSo \\boxed{7}", "42"),
    ("reasoning \\boxed{42}</think>\nThe answer is \\boxed{42}", "42"),
    ("reasoning so far \\boxed{42} wait</think>", "42"),
    ("Partial work, so \\boxed{42}? Check the sign of", "42"),
    ("Start.<think>long reasoning \\boxed{42} that was cut", "42"),
    ("[THINK]x\\boxed{1}[/THINK]\\boxed{2}", "2"),
    ("analysisWe need 2+2.assistantfinal\\boxed{4}", "4"),
    ("\\boxed{0.5}", "\\frac{1}{2}"),
    ("\\boxed{2/4}", "\\frac{1}{2}"),
    ("\\boxed{x^2 + 2x + 1}", "(x+1)^2"),
    ("\\boxed{1,\\!000}", "1000"),
    ("\\boxed{\\$18.90}", "\\$18.90"),
    ("\\boxed{18.9}", "\\$18.90"),
]


def _load_original(repo: pathlib.Path, rev: str, tmp: pathlib.Path):
    source = subprocess.run(
        ["git", "-C", str(repo), "show", f"{rev}:{PARSER_PATH}"], check=True, capture_output=True, text=True
    ).stdout
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"{rev}^{{commit}}"], check=True, capture_output=True, text=True
    ).stdout.strip()
    blob = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"{rev}:{PARSER_PATH}"], check=True, capture_output=True, text=True
    ).stdout.strip()
    path = tmp / "liquid_evals_parser_original.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("liquid_evals_parser_original", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, commit, blob


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--liquid-evals", type=pathlib.Path, required=True, help="liquid_evals clone (read only)")
    parser.add_argument("--rev", default="d5534cf", help="liquid_evals revision to read parser.py at")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        original, commit, blob = _load_original(args.liquid_evals, args.rev, pathlib.Path(tmp))

        def verify(raw: str, gold: str) -> dict:
            response = original.remove_thinking_tags(raw)
            predicted = original.extract_math_answer(response)
            correct = original.verify_math_answer(gold, predicted, response)
            return {"raw": raw, "gold": gold, "predicted": predicted, "correct": bool(correct)}

        fixture = {
            "source": {"repo": "Liquid4All/liquid_evals", "commit": commit, "path": PARSER_PATH, "blob": blob},
            "extract": [{"text": t, "expected": original.extract_math_answer(t)} for t in EXTRACT_CASES],
            "check": [
                {
                    "given": given,
                    "truth": truth,
                    "mathd_normalize_given": original.mathd_normalize_answer(given),
                    "normalize_given": original._normalize(given),
                    "mathd": original.grade_answer_mathd(given, truth),
                    "sympy": original.grade_answer_sympy(given, truth),
                    "check": original.check_math_result(given, truth),
                }
                for given, truth in CHECK_CASES
            ],
            "verify": [verify(raw, gold) for raw, gold in VERIFY_CASES],
        }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")
    print(f"Wrote {FIXTURE} from {PARSER_PATH} @ {commit}")


if __name__ == "__main__":
    main()
