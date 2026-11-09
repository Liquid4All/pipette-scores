"""Unit tests for the VLM scorers (extraction + metrics vendored from VLMEvalKit).

All samples are synthetic (no parquet / LFS data); expected values follow
VLMEvalKit @ 8eca8e5 semantics.
"""

import pytest

import pipette_scores.scoring as scoring
from pipette_scores.scoring import (
    mmbench_en_dev,
    mme,
    mmmu_val,
    mmstar,
    ocrbench,
    pope,
    realworldqa,
)
from pipette_scores.scoring.vlm_upstream.matching_util import can_infer
from pipette_scores.scoring.vlm_upstream.yorn import YOrN_Extraction
from pipette_scores.types import SampleCompletion, VlmSample


def _vlm(id: str, raw_doc: dict, prompt: str = "q") -> VlmSample:
    return VlmSample(id=id, prompt=prompt, answer=None, raw_doc=raw_doc)


def _comp(id: str, text: str) -> SampleCompletion:
    return SampleCompletion(id=id, completion=text)


# ---------------------------------------------------------------------------
# can_infer (VLMEvalKit matching_util)
# ---------------------------------------------------------------------------

CHOICES = {"A": "teddy bear", "B": "rabbit", "C": "cat", "D": "dog"}


class TestCanInfer:
    def test_bare_letter(self):
        assert can_infer("B", CHOICES) == "B"

    def test_answer_phrase(self):
        assert can_infer("The correct choice is B", CHOICES) == "B"

    def test_parenthesized_letter(self):
        assert can_infer("The answer is (C).", CHOICES) == "C"

    def test_letter_with_period(self):
        assert can_infer("D. dog", CHOICES) == "D"

    def test_option_text_match(self):
        assert can_infer("It looks like a rabbit to me", CHOICES) == "B"

    def test_position_constraint_rejects_early_letter(self):
        # The single choice-letter token must be among the last 4 tokens.
        ans = "B is one option but after long reflection over many tokens I remain unsure"
        assert can_infer(ans, CHOICES) is False

    def test_unparseable_is_false(self):
        assert can_infer("I cannot tell what this image shows", CHOICES) is False

    def test_two_letters_no_match(self):
        assert can_infer("Either B or C", CHOICES) is False

    def test_refusal_maps_to_z(self):
        assert can_infer("Sorry, I can't help with images of people yet.", CHOICES) == "Z"

    def test_long_text_answer_rejected(self):
        long = "word " * 50 + "rabbit"
        assert can_infer(long, CHOICES) is False

    def test_verbose_answer_is_letter(self):
        ans = "The correct answer is **B**. Here's why: A and C are both wrong, and D is not shown at all."
        assert can_infer(ans, CHOICES) == "B"

    def test_verbose_answer_letter_outside_choices(self):
        assert can_infer("The answer is D, I think, or maybe A or C", {"A": "x", "B": "y", "C": "z"}) is False


# ---------------------------------------------------------------------------
# Y/N extraction (VLMEvalKit yorn)
# ---------------------------------------------------------------------------


class TestYOrNExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("The answer is yes.", "Yes"),
            ("Yes", "Yes"),
            ("yes, there is a dog.", "Yes"),
            ("No, there is not.", "No"),
            ("The answer is no", "No"),
            ("yes and no", "Unknown"),
            ("maybe", "Unknown"),
            ("", "Unknown"),
        ],
    )
    def test_extraction(self, text, expected):
        assert YOrN_Extraction(text) == expected


# ---------------------------------------------------------------------------
# MME — pairing + rating
# ---------------------------------------------------------------------------


def _mme_sample(id, category, image, answer):
    return _vlm(id, {"category": category, "image_path": image, "answer": answer})


class TestMME:
    def test_pairing_and_rating(self):
        # existence (perception): img1 both correct, img2 one of two correct
        # -> acc = 75, acc_plus = 50, category = 125. No cognition samples.
        samples = [
            _mme_sample("1", "existence", "img1.jpg", "Yes"),
            _mme_sample("2", "existence", "img1.jpg", "No"),
            _mme_sample("3", "existence", "img2.jpg", "Yes"),
            _mme_sample("4", "existence", "img2.jpg", "No"),
        ]
        completions = [
            _comp("1", "The answer is yes."),
            _comp("2", "No."),
            _comp("3", "Yes, it is."),
            _comp("4", "Yes."),  # wrong
        ]
        scores, ctx = mme.score(completions, samples)
        assert [s.is_correct for s in scores] == [True, True, True, False]
        assert ctx["mme_perception_score"] == 125.0
        assert ctx["mme_cognition_score"] == 0.0
        assert ctx["leaderboard_score"] == 125.0
        assert ctx["accuracy"] == pytest.approx(125.0 / 2800, abs=1e-6)

    def test_acc_plus_requires_both_of_pair(self):
        # One of the pair wrong -> acc 50, acc_plus 0 -> category 50.
        samples = [
            _mme_sample("1", "code_reasoning", "c1.png", "Yes"),
            _mme_sample("2", "code_reasoning", "c1.png", "No"),
        ]
        completions = [_comp("1", "yes"), _comp("2", "yes")]
        _, ctx = mme.score(completions, samples)
        assert ctx["mme_cognition_score"] == 50.0
        assert ctx["mme_perception_score"] == 0.0

    def test_pairs_by_question_id_fallback(self):
        # lmms-eval export: question_id is the image relative path.
        samples = [
            _vlm("1", {"category": "OCR", "question_id": "ocr/0001.jpg", "answer": "Yes"}),
            _vlm("2", {"category": "OCR", "question_id": "ocr/0001.jpg", "answer": "No"}),
        ]
        completions = [_comp("1", "yes"), _comp("2", "no")]
        _, ctx = mme.score(completions, samples)
        # both correct: acc 100 + acc_plus 100
        assert ctx["mme_perception_score"] == 200.0


# ---------------------------------------------------------------------------
# POPE — F1 headline
# ---------------------------------------------------------------------------


class TestPOPE:
    def test_f1_on_imbalanced_set(self):
        gts = ["Yes", "Yes", "Yes", "No"]
        preds = ["yes", "The answer is yes.", "no", "Yes, there is."]
        samples = [_vlm(str(i), {"answer": gt, "category": "adversarial"}) for i, gt in enumerate(gts)]
        completions = [_comp(str(i), p) for i, p in enumerate(preds)]
        scores, ctx = pope.score(completions, samples)
        # tp=2, fp=1, fn=1 -> precision=recall=2/3, f1=2/3; accuracy=0.5
        assert ctx["precision"] == pytest.approx(2 / 3)
        assert ctx["recall"] == pytest.approx(2 / 3)
        assert ctx["f1"] == pytest.approx(2 / 3)
        assert ctx["yes_no_accuracy"] == pytest.approx(0.5)
        # Headline metric is F1, not accuracy.
        assert ctx["accuracy"] == pytest.approx(2 / 3)
        assert ctx["leaderboard_score"] == pytest.approx(200 / 3, abs=1e-3)
        assert [s.is_correct for s in scores] == [True, True, False, False]

    def test_unknown_counts_as_no_prediction(self):
        samples = [_vlm("1", {"answer": "Yes", "category": "random"})]
        completions = [_comp("1", "I am not sure")]
        scores, ctx = pope.score(completions, samples)
        assert scores[0].is_correct is False
        assert ctx["f1"] == 0.0

    def test_comma_joined_category_exploded(self):
        samples = [_vlm("1", {"answer": "Yes", "category": "adversarial,popular"})]
        completions = [_comp("1", "yes")]
        _, ctx = pope.score(completions, samples)
        assert set(ctx["per_category"]) == {"adversarial", "popular"}


# ---------------------------------------------------------------------------
# MMMU val — MMMU_preproc + can_infer
# ---------------------------------------------------------------------------


class TestMMMUVal:
    def test_mcq_question(self):
        raw = {"A": "1", "B": "2", "C": "3", "D": "4", "answer": "B", "category": "Math"}
        scores, ctx = mmmu_val.score([_comp("1", "The answer is B.")], [_vlm("1", raw)])
        assert scores[0].is_correct is True
        assert ctx["accuracy"] == 1.0
        assert ctx["num_open_questions_reformulated"] == 0

    def test_open_question_reformulated_to_two_choice(self):
        raw = {"answer": "42", "category": "Physics"}  # no option columns -> open
        scores, ctx = mmmu_val.score([_comp("1", "The answer is 42")], [_vlm("1", raw)])
        assert scores[0].is_correct is True
        assert ctx["num_open_questions_reformulated"] == 1

    def test_open_question_wrong_answer(self):
        raw = {"answer": "42", "category": "Physics"}
        scores, _ = mmmu_val.score([_comp("1", "The result is 41")], [_vlm("1", raw)])
        assert scores[0].is_correct is False

    def test_unparseable_counts_wrong_no_random_fallback(self):
        raw = {"A": "1", "B": "2", "C": "3", "D": "4", "answer": "B", "category": "Math"}
        completions = [_comp("1", "I have no idea, sorry about that friend")]
        for _ in range(3):  # deterministic across calls
            scores, _ = mmmu_val.score(completions, [_vlm("1", raw)])
            assert scores[0].is_correct is False


# ---------------------------------------------------------------------------
# MMStar — micro accuracy
# ---------------------------------------------------------------------------


class TestMMStar:
    def test_micro_not_macro(self):
        # cat X: 2 samples 1 hit; cat Y: 1 sample 1 hit.
        # micro = 2/3; macro would be (0.5 + 1.0)/2 = 0.75.
        samples = [
            _vlm("1", {"A": "red", "B": "blue", "answer": "A", "l2_category": "X"}),
            _vlm("2", {"A": "red", "B": "blue", "answer": "A", "l2_category": "X"}),
            _vlm("3", {"A": "one", "B": "two", "answer": "B", "l2_category": "Y"}),
        ]
        completions = [_comp("1", "A"), _comp("2", "B"), _comp("3", "The answer is B")]
        _, ctx = mmstar.score(completions, samples)
        assert ctx["accuracy"] == pytest.approx(2 / 3)
        assert ctx["per_l2_category_accuracy"] == {"X": 0.5, "Y": 1.0}

    def test_prefix_style_answers(self):
        samples = [_vlm("1", {"A": "cat", "B": "dog", "C": "fox", "D": "owl", "answer": "C"})]
        _, ctx = mmstar.score([_comp("1", "Option (C) is correct")], samples)
        assert ctx["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# RealWorldQA — can_infer over A-D
# ---------------------------------------------------------------------------


class TestRealWorldQA:
    def test_letter_answer(self):
        samples = [_vlm("1", {"A": "left", "B": "right", "C": "straight", "answer": "B"})]
        scores, ctx = realworldqa.score([_comp("1", "B.")], samples)
        assert scores[0].is_correct is True
        assert ctx["leaderboard_score"] == 100.0

    def test_option_text_answer(self):
        samples = [_vlm("1", {"A": "left", "B": "right", "C": "forward", "answer": "C"})]
        scores, _ = realworldqa.score([_comp("1", "The car should go forward")], samples)
        assert scores[0].is_correct is True

    def test_option_text_substring_collision_is_ambiguous(self):
        # Faithful VLMEvalKit behavior: 'cat' is a substring of 'catfish', so
        # a text answer naming 'catfish' matches two options -> no inference.
        samples = [_vlm("1", {"A": "cat", "B": "catfish", "C": "heron", "answer": "B"})]
        scores, _ = realworldqa.score([_comp("1", "It is clearly one catfish")], samples)
        assert scores[0].is_correct is False

    def test_unparseable_is_wrong(self):
        samples = [_vlm("1", {"A": "left", "B": "right", "answer": "A"})]
        scores, _ = realworldqa.score([_comp("1", "hmm, tough to say for sure today")], samples)
        assert scores[0].is_correct is False


# ---------------------------------------------------------------------------
# MMBench — circular grouping
# ---------------------------------------------------------------------------


class TestMMBench:
    def test_circular_all_rotations_required(self):
        base = {"A": "cat", "B": "dog", "answer": "A", "category": "CP"}
        samples = [
            _vlm("1", {**base, "index": 7}),
            _vlm("2", {**base, "answer": "B", "index": 1_000_007}),  # rotation
        ]
        # First rotation right, second wrong -> base question is a miss.
        scores, ctx = mmbench_en_dev.score([_comp("1", "A"), _comp("2", "A")], samples)
        assert [s.is_correct for s in scores] == [True, False]
        assert ctx["num_questions"] == 1
        assert ctx["num_hits"] == 0
        assert ctx["accuracy"] == 0.0
        assert ctx["per_rotation_accuracy"] == 0.5

    def test_circular_hit(self):
        base = {"A": "cat", "B": "dog", "category": "CP"}
        samples = [
            _vlm("1", {**base, "answer": "A", "index": 7}),
            _vlm("2", {**base, "answer": "B", "index": 1_000_007}),
        ]
        _, ctx = mmbench_en_dev.score([_comp("1", "A"), _comp("2", "The answer is B")], samples)
        assert ctx["accuracy"] == 1.0
        assert ctx["num_hits"] == 1


# ---------------------------------------------------------------------------
# OCRBench — category-keyed matching, HME case sensitivity
# ---------------------------------------------------------------------------

HME = "Handwritten Mathematical Expression Recognition"


class TestOCRBench:
    def test_hme_case_sensitive(self):
        samples = [_vlm("1", {"answer": ["X^{2}+Y"], "category": HME})]
        scores, _ = ocrbench.score([_comp("1", "x^{2}+y")], samples)
        assert scores[0].is_correct is False  # case mismatch

        scores, _ = ocrbench.score([_comp("1", "The expression is X^{2}+Y")], samples)
        assert scores[0].is_correct is True

    def test_hme_whitespace_stripped(self):
        samples = [_vlm("1", {"answer": ["a + b"], "category": HME})]
        scores, _ = ocrbench.score([_comp("1", "a+b")], samples)
        assert scores[0].is_correct is True

    def test_other_categories_case_insensitive(self):
        samples = [_vlm("1", {"answer": ["Hello"], "category": "Regular Text Recognition"})]
        scores, ctx = ocrbench.score([_comp("1", "the sign says hello")], samples)
        assert scores[0].is_correct is True
        assert ctx["leaderboard_score"] == 1.0
        assert ctx["accuracy"] == pytest.approx(1 / 1000)
        assert ctx["final_score_norm"] == pytest.approx(0.1)

    def test_stringified_list_answers(self):
        samples = [_vlm("1", {"answer": "['abc', 'def']", "category": "Doc-oriented VQA"})]
        scores, _ = ocrbench.score([_comp("1", "it reads DEF")], samples)
        assert scores[0].is_correct is True

    def test_question_type_field_accepted(self):
        samples = [_vlm("1", {"answer": ["7"], "question_type": "Digit String Recognition"})]
        scores, ctx = ocrbench.score([_comp("1", "7")], samples)
        assert scores[0].is_correct is True
        assert ctx["final_score"] == 1.0


# ---------------------------------------------------------------------------
# Partial submissions — headline metrics use the DATASET denominator
# ---------------------------------------------------------------------------


class TestPartialSubmission:
    """A submission covering only part of the dataset must not inflate the
    headline: missing samples count as wrong, and ``samples_missing`` reports
    the gap. Per-sample scores still cover ONLY the submitted completions
    (the response contract with mgmt)."""

    MCQ_RAW = {"A": "red", "B": "blue", "answer": "A"}

    def _mcq_samples(self):
        return [_vlm(str(i), dict(self.MCQ_RAW)) for i in range(4)]

    @pytest.mark.parametrize("scorer", [mmstar, realworldqa, mmmu_val])
    def test_mcq_dataset_denominator(self, scorer):
        samples = self._mcq_samples()
        # 2 of 4 samples submitted, 1 correct: 0.25 (dataset), not 0.5.
        completions = [_comp("0", "A"), _comp("1", "B")]
        scores, ctx = scorer.score(completions, samples)
        assert [s.id for s in scores] == ["0", "1"]  # submitted-only response
        assert [s.is_correct for s in scores] == [True, False]
        assert ctx["accuracy"] == pytest.approx(0.25)
        assert ctx["leaderboard_score"] == pytest.approx(25.0)
        assert ctx["samples_missing"] == 2

    @pytest.mark.parametrize("scorer", [mmstar, realworldqa, mmmu_val])
    def test_mcq_full_coverage_no_missing(self, scorer):
        samples = self._mcq_samples()
        completions = [_comp(str(i), "A") for i in range(4)]
        _, ctx = scorer.score(completions, samples)
        assert ctx["accuracy"] == 1.0
        assert ctx["samples_missing"] == 0

    def test_mmbench_groups_built_from_dataset(self):
        base = {"A": "cat", "B": "dog", "category": "CP"}
        samples = [
            _vlm("1", {**base, "answer": "A", "index": 7}),
            _vlm("2", {**base, "answer": "B", "index": 1_000_007}),
            _vlm("3", {**base, "answer": "A", "index": 8}),
            _vlm("4", {**base, "answer": "B", "index": 1_000_008}),
        ]
        # Only base question 7 fully submitted (both rotations correct); base
        # question 8 not submitted at all -> it is a miss, accuracy 0.5 not 1.0.
        completions = [_comp("1", "A"), _comp("2", "B")]
        scores, ctx = mmbench_en_dev.score(completions, samples)
        assert [s.id for s in scores] == ["1", "2"]
        assert ctx["num_questions"] == 2  # from the dataset, not the submission
        assert ctx["num_hits"] == 1
        assert ctx["accuracy"] == 0.5
        assert ctx["samples_missing"] == 2

    def test_mmbench_missing_rotation_is_a_miss(self):
        base = {"A": "cat", "B": "dog", "category": "CP"}
        samples = [
            _vlm("1", {**base, "answer": "A", "index": 7}),
            _vlm("2", {**base, "answer": "B", "index": 1_000_007}),
        ]
        # Only one of the two rotations submitted (and correct): group misses.
        _, ctx = mmbench_en_dev.score([_comp("1", "A")], samples)
        assert ctx["num_questions"] == 1
        assert ctx["num_hits"] == 0
        assert ctx["accuracy"] == 0.0
        assert ctx["samples_missing"] == 1

    def test_pope_missing_counts_as_the_wrong_answer(self):
        gts = ["Yes", "Yes", "No", "No"]
        samples = [_vlm(str(i), {"answer": gt, "category": "adversarial"}) for i, gt in enumerate(gts)]
        # Submit only ids 0 and 2, both correct. Missing id 1 (gt Yes) is a
        # false negative; missing id 3 (gt No) a false positive.
        # tp=1 fp=1 fn=1 -> precision 0.5, recall 0.5, f1 0.5.
        completions = [_comp("0", "yes"), _comp("2", "no")]
        scores, ctx = pope.score(completions, samples)
        assert [s.id for s in scores] == ["0", "2"]
        assert [s.is_correct for s in scores] == [True, True]
        assert ctx["precision"] == pytest.approx(0.5)
        assert ctx["recall"] == pytest.approx(0.5)
        assert ctx["f1"] == pytest.approx(0.5)
        assert ctx["accuracy"] == pytest.approx(0.5)  # headline = F1
        assert ctx["yes_no_accuracy"] == pytest.approx(0.5)
        assert ctx["samples_missing"] == 2
        assert ctx["per_sample_accuracy"] == 1.0  # submitted-only diagnostic

    def test_pope_leaving_out_the_negatives_cannot_raise_f1(self):
        gts = ["Yes", "Yes", "No", "No"]
        samples = [_vlm(str(i), {"answer": gt, "category": "adversarial"}) for i, gt in enumerate(gts)]
        _, partial = pope.score([_comp("0", "yes"), _comp("1", "yes")], samples)
        _, full = pope.score([_comp(str(i), gt.lower()) for i, gt in enumerate(gts)], samples)
        assert full["f1"] == pytest.approx(1.0)
        # tp=2 fp=2 fn=0 -> precision 0.5, recall 1, f1 2/3.
        assert partial["f1"] == pytest.approx(2 / 3)

    def test_mme_missing_counts_zero_in_acc_and_acc_plus(self):
        samples = [
            _mme_sample("1", "existence", "img1.jpg", "Yes"),
            _mme_sample("2", "existence", "img1.jpg", "No"),
            _mme_sample("3", "existence", "img2.jpg", "Yes"),
            _mme_sample("4", "existence", "img2.jpg", "No"),
        ]
        # Submit only img1's pair, both correct. Over the dataset: acc = 2/4
        # -> 50, acc_plus = img1 hit only -> 50; category = 100 (NOT 200).
        completions = [_comp("1", "yes"), _comp("2", "no")]
        scores, ctx = mme.score(completions, samples)
        assert [s.id for s in scores] == ["1", "2"]
        assert ctx["mme_perception_score"] == 100.0
        assert ctx["leaderboard_score"] == 100.0
        assert ctx["accuracy"] == pytest.approx(100.0 / 2800, abs=1e-6)
        assert ctx["samples_missing"] == 2
        assert ctx["per_sample_accuracy"] == 1.0  # submitted-only diagnostic

    def test_mme_full_coverage_no_missing(self):
        samples = [
            _mme_sample("1", "existence", "img1.jpg", "Yes"),
            _mme_sample("2", "existence", "img1.jpg", "No"),
        ]
        _, ctx = mme.score([_comp("1", "yes"), _comp("2", "no")], samples)
        assert ctx["mme_perception_score"] == 200.0
        assert ctx["samples_missing"] == 0

    def test_ocrbench_fixed_denominator_and_missing_count(self):
        raw = {"answer": ["Hello"], "category": "Regular Text Recognition"}
        samples = [_vlm(str(i), dict(raw)) for i in range(4)]
        completions = [_comp("0", "hello"), _comp("1", "nope")]
        scores, ctx = ocrbench.score(completions, samples)
        assert [s.id for s in scores] == ["0", "1"]
        assert ctx["final_score"] == 1.0
        assert ctx["accuracy"] == pytest.approx(1 / 1000)  # /1000 fixed
        assert ctx["samples_missing"] == 2
        assert ctx["per_sample_accuracy"] == 0.5  # submitted-only diagnostic

    def test_pope_full_coverage_no_missing(self):
        samples = [_vlm("1", {"answer": "Yes", "category": "random"})]
        _, ctx = pope.score([_comp("1", "yes")], samples)
        assert ctx["samples_missing"] == 0


# ---------------------------------------------------------------------------
# Dispatch wiring
# ---------------------------------------------------------------------------


class TestDispatch:
    @pytest.mark.parametrize(
        "eval_id", ["mme", "mmstar", "ocrbench", "mmmu_val", "realworldqa", "mmbench_en_dev", "pope"]
    )
    def test_score_dispatch(self, eval_id):
        raw = {
            "A": "x",
            "B": "y",
            "answer": "Yes" if eval_id in ("mme", "pope") else "A",
            "category": "existence" if eval_id == "mme" else "Regular Text Recognition",
            "image_path": "i.jpg",
            "index": 1,
        }
        samples = [_vlm("1", raw)]
        completions = [_comp("1", "yes A")]
        scores, ctx = scoring.score(eval_id, completions, samples)
        assert len(scores) == 1
        assert "accuracy" in ctx
        assert "leaderboard_score" in ctx

    @staticmethod
    def _sample(eval_id: str) -> tuple[VlmSample, str]:
        yes_no = eval_id in ("mme", "pope")
        raw = {
            "A": "x",
            "B": "y",
            "answer": "Yes" if yes_no else ("GV" if eval_id == "ocrbench" else "A"),
            "category": "existence" if eval_id == "mme" else "Regular Text Recognition",
            "image_path": "i.jpg",
            "index": 1,
        }
        return _vlm("1", raw), "Yes" if yes_no else ("GV" if eval_id == "ocrbench" else "A")

    @pytest.mark.parametrize(
        "eval_id", ["mme", "mmstar", "ocrbench", "mmmu_val", "realworldqa", "mmbench_en_dev", "pope"]
    )
    def test_reasoning_is_stripped_before_extraction(self, eval_id):
        sample, answer = self._sample(eval_id)
        completion = _comp("1", f"<think>Not sure: No, or B.</think>{answer}")
        scores, ctx = scoring.score(eval_id, [completion], [sample], enable_thinking=True)
        assert scores[0].is_correct
        assert ctx["blanked_truncated"] == 0

    @pytest.mark.parametrize(
        "eval_id", ["mme", "mmstar", "ocrbench", "mmmu_val", "realworldqa", "mmbench_en_dev", "pope"]
    )
    def test_reasoning_cut_at_the_cap_is_graded_empty(self, eval_id):
        sample, answer = self._sample(eval_id)
        completion = SampleCompletion(id="1", completion=f"<think>It is {answer}", stop_reason="truncated")
        scores, ctx = scoring.score(eval_id, [completion], [sample], enable_thinking=True)
        assert not scores[0].is_correct
        assert ctx["blanked_truncated"] == 1


# ---------------------------------------------------------------------------
# Dataset catalog wiring (_vlm_common, synthetic parquet — no LFS data)
# ---------------------------------------------------------------------------


class TestVlmDatasetCatalog:
    def _write_dataset(self, tmp_path, eval_id="pope"):
        import json

        import pyarrow as pa
        import pyarrow.parquet as pq

        d = tmp_path / "datasets" / eval_id / "test_ds"
        d.mkdir(parents=True)
        rows = [
            {
                "sample_id": "s1",
                "prompt": "Is there a dog in the image?",
                "answer": json.dumps("Yes"),
                "raw_doc": json.dumps({"answer": "Yes", "category": "adversarial"}),
                "num_images": 1,
                "image_0": b"\xff\xd8fakejpeg",
            }
        ]
        pq.write_table(pa.Table.from_pylist(rows), d / "test.parquet")
        return tmp_path

    def test_load_eval_samples(self, tmp_path):
        from pipette_scores.dataset_catalog import load_eval_samples
        from pipette_scores.types import EvalId

        root = self._write_dataset(tmp_path)
        samples = load_eval_samples(EvalId.POPE, "test_ds", root_dir=root)
        assert len(samples) == 1
        assert isinstance(samples[0], VlmSample)
        assert samples[0].raw_doc["answer"] == "Yes"

    def test_load_prompt_samples_multimodal_content(self, tmp_path):
        from pipette_scores.dataset_catalog import load_prompt_samples
        from pipette_scores.types import EvalId

        root = self._write_dataset(tmp_path)
        prompts = load_prompt_samples(EvalId.POPE, "test_ds", root_dir=root)
        assert len(prompts) == 1
        content = prompts[0].messages[0].content
        assert isinstance(content, list)
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert content[-1] == {"type": "text", "text": "Is there a dog in the image?"}


# ---------------------------------------------------------------------------
# Replies as the b11744 smoke runs got them from the new VLM families (no profile
# extraction: the reasoning rules and the fallbacks alone)
# ---------------------------------------------------------------------------

_MCQ_DOC = {"A": "blue", "B": "green", "C": "red", "D": "black", "answer": "C", "l2_category": "x"}


class TestSmokeReplies:
    def test_mimo_think_then_boxed_letter(self):
        reply = (
            "<think>\nGot it, let's look at the image. There's a red circle on the left. The options are A to D. "
            "So the circle is red, which is option C.\n</think>\n\n\\boxed{C}"
        )
        _, ctx = mmstar.score([_comp("1", reply)], [_vlm("1", _MCQ_DOC)])
        assert ctx["accuracy"] == 1.0

    def test_mimo_boxed_ocr_answer(self):
        reply = '<think>\nThe text is "PIPETTE 42".\n</think>\n\n\\boxed{PIPETTE 42}'
        samples = [_vlm("1", {"answer": ["PIPETTE 42"], "category": "Regular Text Recognition"})]
        scores, _ = ocrbench.score([_comp("1", reply)], samples)
        assert scores[0].is_correct is True

    def test_kimi_closed_thought_then_the_letter(self):
        reply = "◁think▷Got it, let's look at the image. The circle is red. So the answer is C.◁/think▷C"
        _, ctx = mmstar.score([_comp("1", reply)], [_vlm("1", _MCQ_DOC)])
        assert ctx["accuracy"] == 1.0

    def test_kimi_grades_the_letter_after_the_thought_not_the_ones_inside_it(self):
        reply = "◁think▷Maybe A, maybe B.◁/think▷C"
        _, ctx = mmstar.score([_comp("1", reply)], [_vlm("1", _MCQ_DOC)])
        assert ctx["accuracy"] == 1.0
        reply = "◁think▷Maybe A, maybe C.◁/think▷B"
        _, ctx = mmstar.score([_comp("1", reply)], [_vlm("1", _MCQ_DOC)])
        assert ctx["accuracy"] == 0.0

    def test_ministral_orphan_close_then_yes(self):
        reply = 'The image has a red circle, so the answer should be "yes".[/THINK]yes'
        scores, _ = pope.score([_comp("1", reply)], [_vlm("1", {"answer": "Yes", "category": "adversarial"})])
        assert scores[0].is_correct is True
