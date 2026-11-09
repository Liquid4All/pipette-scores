"""A profile's VLMEvalKit wrapper extraction (scoring/profile_answers.py)."""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from pipette_scores import cli
from pipette_scores.api.app import app
from pipette_scores.scoring import profile_answers
from pipette_scores.scoring.completion_text import split_reasoning
from pipette_scores.types import ChatMessage, EvalId, EvalSample, SampleCompletion, VlmSample


@pytest.mark.parametrize(
    ("benchmark", "reply", "answer"),
    [
        # A reply that ends inside its reasoning still gives its stated answer.
        (EvalId.REALWORLDQA, "<think>\nThe sign says 35, so B.\nAnswer: B", "B"),
        # After a closed reasoning block only the reply counts.
        (EvalId.MMSTAR, "<think>Answer: A, no wait</think>\n\nAnswer: C", "C"),
        # The last of several, any case, and upstream's other spellings.
        (EvalId.MMBENCH_EN_DEV, "Answer: A ... so the Answer is d", "d"),
        (EvalId.MMMU_VAL, "答案是B", "B"),
        (EvalId.MMSTAR, "<answer>C</answer>", "C"),
        # Yes/no keeps everything after the first Answer:.
        (EvalId.POPE, "<think>Is there a dog? No.</think>\nAnswer: No", "No"),
        (EvalId.MME, "<think>\nlooking\nAnswer: Yes", "Yes"),
        # Nothing to cut: the reply as upstream hands it on.
        (EvalId.REALWORLDQA, "<think>\nhmm</think>\nIt is B", "It is B"),
        (EvalId.MMSTAR, "<think>\nno answer stated", "<think>\nno answer stated"),
    ],
)
def test_minicpm_v_cot_reads_the_answer_as_upstream_does(benchmark, reply, answer):
    assert profile_answers.minicpm_v_cot(benchmark, reply) == answer


def _completions(text: str) -> list[SampleCompletion]:
    return [SampleCompletion(id="1", completion=text, stop_reason="eos")]


@pytest.mark.parametrize(
    ("profile", "benchmark"),
    [
        # Another profile's wrapper keeps the reply whole, as does no profile.
        ("lfm2_5_vl_3b", "realworldqa"),
        (None, "realworldqa"),
        # OCRBench is whole on every profile, as upstream leaves it.
        ("minicpm_v_cot", "ocrbench"),
        # A text eval is not a VLM benchmark.
        ("minicpm_v_cot", "ifbench"),
    ],
)
def test_other_replies_are_left_whole(profile, benchmark):
    completions = _completions("<think>\nAnswer: B")
    assert profile_answers.apply(profile, benchmark, completions) == completions


def _rwqa(answer: str) -> VlmSample:
    return VlmSample(
        id="1",
        prompt="Q?",
        answer=answer,
        num_images=1,
        raw_doc={"A": "25", "B": "35", "C": "45", "answer": answer},
    )


@pytest.mark.parametrize(("profile", "correct"), [("minicpm_v_cot", True), (None, False)])
def test_score_grades_what_the_profiles_wrapper_hands_on(profile, correct):
    """A MiniCPM-V reply that states its answer and ends inside its reasoning is
    right under the profile, as under VLMEvalKit; without it the reasoning
    rules drop the reply. The row keeps the reply as sent."""
    reply = "<think>\nThe sign says 35, so B.\nAnswer: B"
    body = {
        "eval_id": "realworldqa",
        "dataset_name": "test",
        "completions": [{"id": "1", "completion": reply, "stop_reason": "eos"}],
    }
    if profile:
        body["prompt_profile"] = profile
    prompts = [EvalSample(id="1", messages=[ChatMessage(role="user", content="Q?")])]
    with (
        patch("pipette_scores.api.app._load_dataset_cached", return_value=[_rwqa("B")]),
        patch("pipette_scores.api.app._load_prompts_cached", return_value=prompts),
    ):
        resp = TestClient(app).post("/score", json=body)
    assert resp.status_code == 200, resp.text
    row = resp.json()["scored_samples"][0]
    assert row["is_correct"] is correct
    assert row["completion"] == reply


@pytest.mark.parametrize(("profile", "accuracy"), [("minicpm_v_cot", "100.00%"), (None, "0.00%")])
def test_the_cli_grades_what_the_profiles_wrapper_hands_on(profile, accuracy, tmp_path, monkeypatch, capsys):
    """The CLI grades as /score does."""
    completions = tmp_path / "completions.jsonl"
    completions.write_text(json.dumps({"id": "1", "completion": "<think>\nSo B.\nAnswer: B", "stop_reason": "eos"}))
    argv = ["pipette-scores", "score", "--eval", "realworldqa", "--dataset", "test", "-f", str(completions)]
    monkeypatch.setattr("sys.argv", [*argv, "--no-samples", *(["--prompt-profile", profile] if profile else [])])
    with patch("pipette_scores.dataset_catalog.load_eval_samples", return_value=[_rwqa("B")]):
        cli.main()
    assert accuracy in capsys.readouterr().out


# Replies from the b11744 smoke runs (Kimi-VL-A3B-Thinking-2506, Q4_K_M) to its MCQ and OCR questions, and the
# 2075 reply, which the server delivered starting mid-thought.
KIMI_MCQ = "\u25c1think\u25b7Got it, let's look at the image. The circle is red. So the answer is C.\u25c1/think\u25b7C"
KIMI_OCR = (
    "\u25c1think\u25b7So, let's look at the image. The text in the image is \"PIPELINE 4\". Let's check each part. "
    "The letters are P, I, R, E, P, L, E, then 4. So combining them: PIPELINE 4.\u25c1/think\u25b7PIPELINE 4"
)
KIMI_NO_OPENER = '5"? Wait, no, no, wait. Wait, no, the number is 2075.\u25c1/think\u25b72075'


@pytest.mark.parametrize(
    ("benchmark", "reply", "answer"),
    [
        # The summary after the thought, on every benchmark.
        (EvalId.REALWORLDQA, KIMI_MCQ, "C"),
        (EvalId.OCRBENCH, KIMI_OCR, "PIPELINE 4"),
        (EvalId.MME, "\u25c1think\u25b7Is it? Yes.\u25c1/think\u25b7  Yes\n", "Yes"),
        (EvalId.POPE, KIMI_NO_OPENER, "2075"),
        # A close after the answer began is a stray one: the text before it is reasoning, as liquid_evals reads it.
        (EvalId.MMBENCH_EN_DEV, "\u25c1think\u25b7a\u25c1/think\u25b7B\u25c1/think\u25b7C", "C"),
        # An unclosed thought is no answer, as upstream's extract_summary has it, however far the reply got.
        (EvalId.MMSTAR, "\u25c1think\u25b7So the answer is C, but let me check", ""),
        # Unless the model stopped with a complete <answer>...</answer> in it: split_reasoning grades that.
        (EvalId.MMSTAR, "\u25c1think\u25b7So <answer>C</answer>, but let me check", "C"),
        (EvalId.REALWORLDQA, "C\u25c1think\u25b7wait", "C"),  # the text before the opener stays, as for every family
        # Unless the thought closes: the summary is only what follows the close, as extract_summary reads it.
        (EvalId.MMSTAR, "A\u25c1think\u25b7Maybe A. No, it is B.\u25c1/think\u25b7Answer: $\\boxed{B}$", "B"),
        (EvalId.REALWORLDQA, "A\u25c1think\u25b7Maybe A. No, B.\u25c1/think\u25b7B", "B"),
        # No tags: the reply as it is.
        (EvalId.REALWORLDQA, "It is B", "It is B"),
        # The boxed content of the last \\boxed{}, on MMMU and MMStar only.
        (EvalId.MMSTAR, "\u25c1think\u25b7Red.\u25c1/think\u25b7Answer: $\\boxed{C}$", "C"),
        (EvalId.MMMU_VAL, "\u25c1think\u25b7x\u25c1/think\u25b7First \\boxed{A}, no: Answer: $\\boxed{B}$", "B"),
        (EvalId.MMSTAR, "\u25c1think\u25b7x\u25c1/think\u25b7$\\boxed{\\frac{1}{2}}$", "\\frac{1}{2}"),
        (EvalId.MMSTAR, "\u25c1think\u25b7x\u25c1/think\u25b7$\\boxed{\\text{C}}$", "\\text{C}"),
        (EvalId.MMSTAR, "\u25c1think\u25b7x\u25c1/think\u25b7I think \\boxed{C", "I think \\boxed{C"),
        (EvalId.REALWORLDQA, "\u25c1think\u25b7x\u25c1/think\u25b7Answer: $\\boxed{C}$", "Answer: $\\boxed{C}$"),
        # A boxed answer inside the thought is not the answer.
        (EvalId.MMSTAR, "\u25c1think\u25b7\\boxed{A}\u25c1/think\u25b7C", "C"),
    ],
)
def test_kimi_vl_reads_the_answer_as_its_wrapper_does(benchmark, reply, answer):
    # The extraction keeps the thought in the reply; the reasoning rules then cut it, as they do for every family.
    extracted = profile_answers.kimi_vl(benchmark, reply)
    graded = split_reasoning(SampleCompletion(id="1", completion=extracted, stop_reason="eos"), opens_reasoning=True)
    assert graded.answer == answer


def test_apply_leaves_ocrbench_whole_on_every_profile():
    completions = _completions("\u25c1think\u25b7PIPETTE 42, wait")
    for profile in ("kimi_vl", "minicpm_v_cot", "phi4_rv_think"):
        assert profile_answers.apply(profile, "ocrbench", completions) == completions


def test_a_profile_without_an_extraction_leaves_the_new_profiles_siblings_whole():
    completions = _completions("\u25c1think\u25b7x\u25c1/think\u25b7C")
    for profile in ("glm_v", "nemotron_think", "mimo_vl", "phi4_rv_think"):
        assert profile_answers.apply(profile, "mmstar", completions) == completions


def _mmstar(answer: str) -> VlmSample:
    return VlmSample(
        id="1",
        prompt="Q?",
        answer=answer,
        raw_doc={"A": "blue", "B": "green", "C": "red", "D": "black", "answer": answer, "l2_category": "x"},
    )


def _post_score(profile, reply, *, stop_reason="eos", benchmark="mmstar", sample=None):
    body = {
        "eval_id": benchmark,
        "dataset_name": "test",
        "completions": [{"id": "1", "completion": reply, "stop_reason": stop_reason}],
    }
    if profile:
        body["prompt_profile"] = profile
    prompts = [EvalSample(id="1", messages=[ChatMessage(role="user", content="Q?")])]
    with (
        patch("pipette_scores.api.app._load_dataset_cached", return_value=[sample or _mmstar("C")]),
        patch("pipette_scores.api.app._load_prompts_cached", return_value=prompts),
    ):
        resp = TestClient(app).post("/score", json=body)
    assert resp.status_code == 200, resp.text
    _post_score.context = resp.json()["context"]
    return resp.json()["scored_samples"][0]


@pytest.mark.parametrize(
    ("profile", "reply", "correct"),
    [
        # Kimi's real reply is graded on what follows its thought, with or without its profile.
        ("kimi_vl", KIMI_MCQ, True),
        (None, KIMI_MCQ, True),
        # A reply with a final answer after the thought, whatever the thought says on the way.
        ("kimi_vl", "\u25c1think\u25b7Maybe A? No, B? No.\u25c1/think\u25b7Answer: $\\boxed{C}$", True),
        (None, "\u25c1think\u25b7Maybe A? No, B? No.\u25c1/think\u25b7Answer: $\\boxed{C}$", True),
        # The thought is cut: the letter it mentions is not an answer.
        ("kimi_vl", "\u25c1think\u25b7The answer is C, but wait", False),
        (None, "\u25c1think\u25b7The answer is C, but wait", False),
        # The wrong answer after a right thought is wrong.
        ("kimi_vl", "\u25c1think\u25b7The answer is C.\u25c1/think\u25b7Answer: $\\boxed{B}$", False),
    ],
)
def test_score_grades_kimi_replies_on_the_summary_after_their_thought(profile, reply, correct):
    row = _post_score(profile, reply)
    assert row["is_correct"] is correct
    assert row["completion"] == reply


def test_score_grades_kimi_ocrbench_on_the_summary():
    sample = VlmSample(
        id="1",
        prompt="Q?",
        answer=json.dumps(["PIPELINE 4"]),
        raw_doc={"question": "What is written?", "answer": ["PIPELINE 4"], "category": "Regular Text Recognition"},
    )
    assert _post_score("kimi_vl", KIMI_OCR, benchmark="ocrbench", sample=sample)["is_correct"] is True
    # A cut thought that quotes the text is not the answer.
    cut = _post_score("kimi_vl", "\u25c1think\u25b7The text is PIPELINE 4, but", benchmark="ocrbench", sample=sample)
    assert cut["is_correct"] is False


@pytest.mark.parametrize(
    ("reply", "stop_reason", "correct", "blanked"),
    [
        # What follows the prefill: the rest of the thought, the close, the solution.
        ("The circle is red, so C.</think>\nC", "eos", True, 0),
        ("The circle is red, so C.</think>\n\nAnswer: C", "eos", True, 0),
        ("Not A, not B.</think>\nA", "eos", False, 0),
        # The solution cut short after the close is graded as it stands.
        ("Red.</think>\nAnswer: C. The circle", "truncated", True, 0),
        # The model answered and stopped without closing the thought its prefill opened: graded as written.
        ("C", "eos", True, 0),
        # Cut inside the thought: graded whole, as liquid_evals grades a prompt-opened thought, and
        # counted by the strict view, which grades it as empty.
        ("The circle is red, so the answer is C. Let me double", "truncated", True, 1),
        ("The circle is red, so the answer is B. Let me double", "doom_loop", False, 1),
    ],
)
def test_score_grades_phi4_replies_that_follow_the_think_prefill(reply, stop_reason, correct, blanked):
    row = _post_score("phi4_rv_think", reply, stop_reason=stop_reason)
    assert row["is_correct"] is correct
    assert row["completion"] == reply
    assert _post_score.context["blanked_truncated"] == blanked


def test_without_the_profile_a_cut_thought_with_no_tag_is_not_known_to_be_reasoning():
    """The profile is what says a tagless reply was reasoning."""
    _post_score(None, "The circle is red, so the answer is C. Let me double", stop_reason="truncated")
    assert _post_score.context["blanked_truncated"] == 0


@pytest.mark.parametrize("profile", ["kimi_vl", "phi4_rv_think"])
def test_the_cli_applies_the_new_extractions(profile, tmp_path, monkeypatch, capsys):
    reply = {"kimi_vl": KIMI_MCQ, "phi4_rv_think": "Red.</think>\nC"}[profile]
    completions = tmp_path / "completions.jsonl"
    completions.write_text(json.dumps({"id": "1", "completion": reply, "stop_reason": "eos"}))
    argv = ["pipette-scores", "score", "--eval", "mmstar", "--dataset", "test", "-f", str(completions)]
    monkeypatch.setattr("sys.argv", [*argv, "--no-samples", "--prompt-profile", profile])
    with patch("pipette_scores.dataset_catalog.load_eval_samples", return_value=[_mmstar("C")]):
        cli.main()
    assert "100.00%" in capsys.readouterr().out
