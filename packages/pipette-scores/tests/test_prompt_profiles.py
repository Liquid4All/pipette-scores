"""Each profile's text, spelled out, against VLMEvalKit 8eca8e5's wrappers."""

import functools
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipette_scores import dataset_catalog
from pipette_scores.dataset_catalog import _vlm_common, prompt_profiles
from pipette_scores.dataset_catalog.prompt_profiles import BENCHMARKS, PROFILES
from pipette_scores.types import EvalId

NAN = float("nan")
CHOICE = {"question": "What is shown?", "hint": "Look closely.", "A": "a cat", "B": "a dog", "C": NAN, "D": None}
CHOICE_DEFAULT = (
    "Hint: Look closely.\nQuestion: What is shown?\nOptions:\nA. a cat\nB. a dog\n"
    "Please select the correct answer from the options above. \n"
)
OPEN = {"question": "How many legs?", "hint": NAN, "A": NAN}  # an MMMU question without options
OPEN_DEFAULT = "Question: How many legs?\n"
YES_NO = {"question": "Is there a cat? Please answer yes or no.", "category": "existence"}
OCR = {"question": "what is written in the image?", "category": "Regular Text Recognition"}

LETTER = "Answer with the option's letter from the given choices directly."
SHORT = "Answer the question using a single word or phrase."
LFM_DIRECT = "Please answer directly with only the final answer, do not give any explanation."
MINICPM_CHOICE_COT = (
    "Carefully read the following multichoice question, solve it step by step and finally pick the option "
    'associated with the correct answer in the format of "Answer: selected option\n\n'
)
MINICPM_SHORT_COT = (
    "Read the following question carefully, solve it step by step, and then output the final answer in the "
    'format of "Answer: single number or single word or phrase".\n\n'
)


def text(profile: str, benchmark: EvalId, doc: dict, default: str = "DEFAULT") -> str:
    return PROFILES[profile](benchmark, doc, default)


class TestInternVL:
    def test_multiple_choice_puts_hint_question_options_then_the_letter_instruction(self):
        assert text("internvl", EvalId.MMSTAR, CHOICE) == (
            f"Look closely.\nWhat is shown?\nA. a cat\nB. a dog\n{LETTER}"
        )

    def test_without_options_it_asks_for_a_direct_answer(self):
        assert text("internvl", EvalId.MMMU_VAL, OPEN) == "How many legs?\nAnswer the question directly."

    def test_chinese_text_gets_the_chinese_instruction(self):
        doc = {"question": "图中是什么？", "A": "猫", "B": "狗"}
        assert text("internvl", EvalId.MMBENCH_EN_DEV, doc) == "图中是什么？\nA. 猫\nB. 狗\n请直接回答选项字母。"

    def test_yes_no_and_ocr(self):
        assert text("internvl", EvalId.MME, YES_NO) == f"Is there a cat? Please answer yes or no. {SHORT}"
        assert text("internvl", EvalId.POPE, YES_NO) == YES_NO["question"]
        assert text("internvl", EvalId.OCRBENCH, OCR) == f"what is written in the image?\n{SHORT}"


class TestGraniteVision:
    def test_multiple_choice_matches_internvl(self):
        for benchmark in (EvalId.MMSTAR, EvalId.REALWORLDQA, EvalId.MMBENCH_EN_DEV, EvalId.MMMU_VAL):
            assert text("granite_vision", benchmark, CHOICE) == text("internvl", benchmark, CHOICE)

    def test_ocr_suffix_and_dataset_prompt_elsewhere(self):
        assert text("granite_vision", EvalId.OCRBENCH, OCR) == (
            "what is written in the image?\nReply with only one word or a short phrase or a full address."
        )
        assert text("granite_vision", EvalId.MME, YES_NO, "D") == "D"
        assert text("granite_vision", EvalId.POPE, YES_NO, "D") == "D"


class TestMiniCPMV:
    def test_direct_multiple_choice(self):
        assert text("minicpm_v", EvalId.REALWORLDQA, CHOICE) == (
            f"Hint: Look closely.\nQuestion: What is shown?\nOptions:\nA. a cat\nB. a dog\n\n{LETTER}"
        )

    def test_direct_without_options(self):
        assert text("minicpm_v", EvalId.MMMU_VAL, OPEN) == (
            "Carefully read the following question. Answer the question directly.\nQuestion: How many legs?\n"
        )

    def test_direct_yes_no_and_ocr(self):
        assert text("minicpm_v", EvalId.MME, YES_NO) == YES_NO["question"]
        assert text("minicpm_v", EvalId.POPE, YES_NO) == YES_NO["question"]
        assert text("minicpm_v", EvalId.OCRBENCH, OCR) == f"{SHORT}\nwhat is written in the image?"

    def test_step_by_step_multiple_choice(self):
        assert text("minicpm_v_cot", EvalId.MMSTAR, CHOICE) == (
            MINICPM_CHOICE_COT + "Hint: Look closely.\nWhat is shown?\nA. a cat\nB. a dog\n"
        )

    def test_step_by_step_without_options_yes_no_and_ocr(self):
        assert text("minicpm_v_cot", EvalId.MMMU_VAL, OPEN) == MINICPM_SHORT_COT + "How many legs?\n"
        assert text("minicpm_v_cot", EvalId.MME, YES_NO) == MINICPM_SHORT_COT + YES_NO["question"]
        assert text("minicpm_v_cot", EvalId.OCRBENCH, OCR) == (
            "Carefully observe the image and answer the OCR-related questions below. \n\nwhat is written in the image?"
        )


class TestLFM:
    def test_lfm_vl_appends_the_direct_answer_instruction_everywhere(self):
        for benchmark in BENCHMARKS:
            assert text("lfm_vl", benchmark, CHOICE, CHOICE_DEFAULT) == f"{CHOICE_DEFAULT}\n{LFM_DIRECT}"

    def test_lfm2_5_vl_3b(self):
        assert text("lfm2_5_vl_3b", EvalId.MMBENCH_EN_DEV, CHOICE, CHOICE_DEFAULT) == f"{CHOICE_DEFAULT}\n{LFM_DIRECT}"
        assert text("lfm2_5_vl_3b", EvalId.MMMU_VAL, OPEN, OPEN_DEFAULT) == OPEN_DEFAULT
        assert text("lfm2_5_vl_3b", EvalId.POPE, YES_NO, "D") == "D"
        assert text("lfm2_5_vl_3b", EvalId.OCRBENCH, OCR, OCR["question"]) == (
            "what is written in the image?\nPlease answer concisely with short words or phrases when possible."
        )

    def test_lfm2_5_vl_3b_mme_prompt_follows_the_category(self):
        assert text("lfm2_5_vl_3b", EvalId.MME, YES_NO) == YES_NO["question"]
        artwork = {**YES_NO, "category": "artwork"}
        assert text("lfm2_5_vl_3b", EvalId.MME, artwork) == (
            "Is there a cat? Please answer yes or no.\nAnswer with a single word: Yes or No."
        )
        celebrity = {**YES_NO, "category": "celebrity"}
        assert text("lfm2_5_vl_3b", EvalId.MME, celebrity) == (
            "This is a face recognition question about the person shown. Decide whether the name in the question "
            "matches the image. Is there a cat? Please answer yes or no.\nAnswer with a single word: Yes or No."
        )

    def test_mme_categories_are_upstreams_fourteen(self):
        assert set(prompt_profiles.LFM_MME_BY_CATEGORY) == {
            "ocr", "artwork", "celebrity", "color", "count", "existence", "landmark", "position", "posters",
            "scene", "code_reasoning", "commonsense_reasoning", "numerical_calculation", "text_translation",
        }  # fmt: skip


class TestQwen3_5:
    QWEN_CHOICE = 'Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`.'

    def test_multiple_choice_adds_the_model_cards_answer_field_instruction(self):
        for benchmark in (EvalId.MMSTAR, EvalId.REALWORLDQA, EvalId.MMBENCH_EN_DEV, EvalId.MMMU_VAL):
            assert text("qwen3_5", benchmark, CHOICE, CHOICE_DEFAULT) == (
                "Hint: Look closely.\nQuestion: What is shown?\nOptions:\nA. a cat\nB. a dog\n"
                f"Please select the correct answer from the options above.\n{self.QWEN_CHOICE}"
            )

    def test_open_questions_yes_no_and_ocr_keep_the_dataset_prompt(self):
        assert text("qwen3_5", EvalId.MMMU_VAL, OPEN, OPEN_DEFAULT) == OPEN_DEFAULT
        assert text("qwen3_5", EvalId.MME, YES_NO, "D") == "D"
        assert text("qwen3_5", EvalId.POPE, YES_NO, "D") == "D"
        assert text("qwen3_5", EvalId.OCRBENCH, OCR, "D") == "D"


GLM_YES_NO = "Please answer yes or no."
KIMI_COT = (
    "Answer the preceding question. The last line of your response should follow this format: "
    "'Answer: $\\boxed{LETTER}$' (without quotes), where LETTER is one of the options. "
    "Think step by step logically, considering all relevant information before answering."
)
SELECT = "Please select the correct answer from the options above. \n"


class TestGlmV:
    def test_pope_gets_vlmevalkits_yes_no_line_on_its_own_line(self):
        assert text("glm_v", EvalId.POPE, YES_NO, "Is there a cat in the image?") == (
            f"Is there a cat in the image?\n{GLM_YES_NO}"
        )

    def test_every_other_benchmark_keeps_the_dataset_prompt(self):
        for benchmark in BENCHMARKS - {EvalId.POPE}:
            assert text("glm_v", benchmark, CHOICE, CHOICE_DEFAULT) == CHOICE_DEFAULT


class TestKimiVL:
    def test_mmstar_replaces_the_closing_line_with_the_boxed_instruction(self):
        assert text("kimi_vl", EvalId.MMSTAR, CHOICE, CHOICE_DEFAULT) == (
            "Hint: Look closely.\nQuestion: What is shown?\nOptions:\nA. a cat\nB. a dog\n" + KIMI_COT
        )

    def test_mmmu_question_without_options_gets_the_instruction_too(self):
        # Upstream's replace is a no-op there and its append is not conditional.
        assert text("kimi_vl", EvalId.MMMU_VAL, OPEN, OPEN_DEFAULT) == OPEN_DEFAULT + KIMI_COT

    def test_mmmu_carries_the_instruction_on_every_piece_of_text_around_its_images(self):
        # VLMEvalKit's split_MMMU turns this into text, image, text, image, text, and appends the
        # instruction to each text item; the loader puts the images back at the placeholders.
        default = "Question: <image 1> and <image 2> differ.\nOptions:\nA. a\nB. b\n" + SELECT
        assert text("kimi_vl", EvalId.MMMU_VAL, CHOICE, default) == (
            f"Question: {KIMI_COT}<image 1> and {KIMI_COT}<image 2> differ.\nOptions:\nA. a\nB. b\n{KIMI_COT}"
        )

    def test_mmmu_pieces_around_adjacent_placeholders_are_empty_text_items_with_the_instruction(self):
        default = "<image 1><image 2>Which?\n" + SELECT
        assert text("kimi_vl", EvalId.MMMU_VAL, CHOICE, default) == (
            f"{KIMI_COT}<image 1>{KIMI_COT}<image 2>Which?\n{KIMI_COT}"
        )

    def test_mmmu_with_a_malformed_placeholder_gets_the_instruction_once(self):
        default = "See <image x> please.\n" + SELECT
        assert text("kimi_vl", EvalId.MMMU_VAL, CHOICE, default) == "See <image x> please.\n" + KIMI_COT

    def test_the_other_benchmarks_keep_the_dataset_prompts(self):
        for benchmark in (EvalId.REALWORLDQA, EvalId.MMBENCH_EN_DEV):
            assert text("kimi_vl", benchmark, CHOICE, CHOICE_DEFAULT) == CHOICE_DEFAULT
        for benchmark in (EvalId.MME, EvalId.POPE, EvalId.OCRBENCH):
            assert text("kimi_vl", benchmark, YES_NO, "D") == "D"


class TestNemotronThink:
    def test_the_dataset_prompt_starts_with_a_newline(self):
        for benchmark in BENCHMARKS:
            assert text("nemotron_think", benchmark, CHOICE, CHOICE_DEFAULT) == "\n" + CHOICE_DEFAULT

    def test_the_system_message_is_the_think_flag(self):
        assert prompt_profiles.SYSTEM_PROMPTS["nemotron_think"] == "/think"


class TestPhi4RvThink:
    def test_the_dataset_prompts_stand_and_the_conversation_ends_with_a_think_prefill(self):
        for benchmark in BENCHMARKS:
            assert text("phi4_rv_think", benchmark, CHOICE, CHOICE_DEFAULT) == CHOICE_DEFAULT
        assert prompt_profiles.PREFILLS == {"phi4_rv_think": "<think>"}
        assert "phi4_rv_think" not in prompt_profiles.SYSTEM_PROMPTS


class TestMimoVl:
    def test_the_dataset_prompts_stand_under_a_helpful_assistant_system_message(self):
        for benchmark in BENCHMARKS:
            assert text("mimo_vl", benchmark, CHOICE, CHOICE_DEFAULT) == CHOICE_DEFAULT
        assert prompt_profiles.SYSTEM_PROMPTS["mimo_vl"] == "You are a helpful assistant."


# The profiles that existed before glm_v, kimi_vl, nemotron_think, phi4_rv_think and mimo_vl.
_EXISTING = (
    "internvl",
    "internvl_r1",
    "granite_vision",
    "minicpm_v",
    "minicpm_v_cot",
    "lfm_vl",
    "lfm2_5_vl_3b",
    "qwen3_5",
)


def test_existing_profiles_render_the_same_prompts_as_before_the_new_ones_were_added():
    """Every existing profile on every benchmark, for a multiple-choice row and an open one,
    digested; the digest was taken on origin/main (a8afd30) before the new profiles."""
    docs = {"choice": {**CHOICE, "category": "existence"}, "open": {**OPEN, "category": "ocr"}}
    rendered = {
        f"{name}|{benchmark.value}|{kind}": PROFILES[name](benchmark, doc, "DEFAULT\n")
        for name in _EXISTING
        for benchmark in sorted(BENCHMARKS, key=lambda b: b.value)
        for kind, doc in docs.items()
    }
    assert len(rendered) == 112
    blob = json.dumps(rendered, sort_keys=True, ensure_ascii=False)
    assert (
        hashlib.sha256(blob.encode()).hexdigest() == "bbe18c2f3e8df1edc5f0147d606822e3b8da96a6e7f2ba8d04359ea57d051b47"
    )


def test_the_profile_names_are_the_existing_ones_and_the_new_ones():
    assert set(PROFILES) == {*_EXISTING, "glm_v", "kimi_vl", "nemotron_think", "phi4_rv_think", "mimo_vl"}


@pytest.mark.parametrize("name", PROFILES)
@pytest.mark.parametrize("benchmark", sorted(BENCHMARKS))
def test_every_profile_covers_every_benchmark(name, benchmark):
    doc = {**CHOICE, "category": "existence"} if benchmark != EvalId.OCRBENCH else OCR
    assert isinstance(text(name, benchmark, doc), str)


class TestLoading:
    def _dataset(self, tmp_path):
        row = {
            "sample_id": "s1",
            "doc_id": "d1",
            "prompt": CHOICE_DEFAULT,
            "answer": json.dumps("A"),
            "raw_doc": json.dumps({k: (None if v != v else v) for k, v in CHOICE.items()}),
            "num_images": 0,
        }
        d = tmp_path / "ds"
        d.mkdir()
        pq.write_table(pa.Table.from_pylist([row]), d / "test.parquet")
        return d

    def test_a_profile_replaces_the_prompt_text(self, tmp_path):
        d = self._dataset(tmp_path)
        default = _vlm_common.load_prompt_samples(d)
        profiled = _vlm_common.load_prompt_samples(d, prompt=functools.partial(PROFILES["internvl"], EvalId.MMSTAR))
        assert default[0].messages[0].content[-1]["text"] == CHOICE_DEFAULT
        assert profiled[0].messages[0].content[-1]["text"] == text("internvl", EvalId.MMSTAR, CHOICE)

    def test_internvl_r1_adds_the_r1_system_prompt_before_internvls_prompt(self, tmp_path):
        d = self._dataset(tmp_path)
        samples = _vlm_common.load_prompt_samples(
            d,
            prompt=functools.partial(PROFILES["internvl_r1"], EvalId.MMSTAR),
            system=prompt_profiles.SYSTEM_PROMPTS["internvl_r1"],
        )
        system, user = samples[0].messages
        assert system.role == "system"
        assert system.content.startswith("You are an AI assistant that rigorously follows this response protocol:")
        assert "Enclose this entire thinking process within <think> and </think> tags." in system.content
        assert user.content[-1]["text"] == text("internvl", EvalId.MMSTAR, CHOICE)

    def test_only_these_profiles_have_a_system_prompt(self):
        assert set(prompt_profiles.SYSTEM_PROMPTS) == {"internvl_r1", "nemotron_think", "mimo_vl"}

    def test_only_phi4_rv_think_has_a_prefill(self):
        assert set(prompt_profiles.PREFILLS) == {"phi4_rv_think"}

    def test_existing_profiles_send_the_same_messages_as_before_the_prefill_existed(self, tmp_path):
        d = self._dataset(tmp_path)
        for name in _EXISTING:
            samples = _vlm_common.load_prompt_samples(
                d,
                prompt=functools.partial(PROFILES[name], EvalId.MMSTAR),
                system=prompt_profiles.SYSTEM_PROMPTS.get(name),
                prefill=prompt_profiles.PREFILLS.get(name),
            )
            roles = [m.role for m in samples[0].messages]
            assert roles == (["system", "user"] if name == "internvl_r1" else ["user"])

    def test_a_system_message_comes_first_and_a_prefill_last(self, tmp_path):
        d = self._dataset(tmp_path)
        samples = _vlm_common.load_prompt_samples(d, system="S", prefill="<think>")
        assert [(m.role, m.content if isinstance(m.content, str) else "blocks") for m in samples[0].messages] == [
            ("system", "S"),
            ("user", "blocks"),
            ("assistant", "<think>"),
        ]

    def _served(self, tmp_path, profile: str, benchmark: EvalId = EvalId.MMSTAR):
        """The samples the catalog serves for a profile: what /samples sends."""
        row = {
            "sample_id": "s1",
            "doc_id": "d1",
            "prompt": CHOICE_DEFAULT,
            "answer": json.dumps("A"),
            "raw_doc": json.dumps({k: (None if v != v else v) for k, v in CHOICE.items()}),
            "num_images": 1,
            "image_0": b"\x89PNG\r\n\x1a\n0",
        }
        dataset = tmp_path / "datasets" / benchmark.value / "x"
        dataset.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([row]), dataset / "test.parquet")
        return dataset_catalog.load_prompt_samples.__wrapped__(
            benchmark, "x", root_dir=tmp_path, prompt_profile=profile
        )

    def test_nemotron_think_serves_a_think_system_message_then_the_image_then_a_newline_led_question(self, tmp_path):
        sample = self._served(tmp_path, "nemotron_think", EvalId.MMSTAR)[0]
        system, user = sample.messages
        assert (system.role, system.content) == ("system", "/think")
        assert user.role == "user"
        assert [b["type"] for b in user.content] == ["image_url", "text"]
        assert user.content[-1]["text"] == "\n" + CHOICE_DEFAULT

    def test_phi4_rv_think_serves_the_user_turn_then_an_assistant_think_prefill_and_no_system_message(self, tmp_path):
        sample = self._served(tmp_path, "phi4_rv_think")[0]
        assert [m.role for m in sample.messages] == ["user", "assistant"]
        assert [b["type"] for b in sample.messages[0].content] == ["image_url", "text"]
        assert sample.messages[0].content[-1]["text"] == CHOICE_DEFAULT
        assert sample.messages[1].content == "<think>"
        assert sample.messages[1].model_dump() == {"role": "assistant", "content": "<think>"}

    def test_mimo_vl_serves_a_helpful_assistant_system_message(self, tmp_path):
        sample = self._served(tmp_path, "mimo_vl")[0]
        assert [(m.role, m.content if isinstance(m.content, str) else "") for m in sample.messages] == [
            ("system", "You are a helpful assistant."),
            ("user", ""),
        ]

    def test_kimi_vl_serves_mmmu_images_between_pieces_of_text_that_each_carry_the_instruction(self, tmp_path):
        row = {
            "sample_id": "s1",
            "doc_id": "d1",
            "prompt": "Question: <image 1> vs <image 2>?\nOptions:\nA. a\nB. b\n" + SELECT,
            "answer": json.dumps("A"),
            "raw_doc": json.dumps({"question": "q", "A": "a", "B": "b"}),
            "num_images": 2,
            "image_0": b"\xff\xd8\xff0",
            "image_1": b"\xff\xd8\xff1",
        }
        dataset = tmp_path / "datasets" / "mmmu_val" / "x"
        dataset.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([row]), dataset / "test.parquet")
        sample = dataset_catalog.load_prompt_samples.__wrapped__(
            EvalId.MMMU_VAL, "x", root_dir=tmp_path, prompt_profile="kimi_vl"
        )[0]
        (user,) = sample.messages
        assert [b["type"] for b in user.content] == ["text", "image_url", "text", "image_url", "text"]
        assert [b["text"] for b in user.content if b["type"] == "text"] == [
            f"Question: {KIMI_COT}",
            f" vs {KIMI_COT}",
            f"?\nOptions:\nA. a\nB. b\n{KIMI_COT}",
        ]

    def test_unknown_profile_and_text_evals_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="unknown prompt profile"):
            dataset_catalog.load_prompt_samples(EvalId.MMSTAR, "x", root_dir=tmp_path, prompt_profile="nope")
        with pytest.raises(ValueError, match="no prompt profiles"):
            dataset_catalog.load_prompt_samples(EvalId.MATH_500, "x", root_dir=tmp_path, prompt_profile="internvl")
