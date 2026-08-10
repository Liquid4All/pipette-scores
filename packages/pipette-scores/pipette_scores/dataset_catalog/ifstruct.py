import json
import pathlib

from ifstruct.dataset import IfStructExample, load_examples

from pipette_scores.repeats import expand, read_repeats
from pipette_scores.types import ChatMessage, EvalSample, IFStructSample
from pipette_scores.hashing import short_hash


def _sample_id(ex: IfStructExample) -> str:
    # `seed` is excluded — it's the generator's PRNG input, not task identity.
    return short_hash(
        "|".join(
            map(
                str,
                [
                    ex.prompt,
                    json.dumps(ex.json_schema, sort_keys=True, separators=(",", ":")),
                    ex.top_level_key,
                    ex.top_level_count,
                    ex.require_wrapper_key,
                    ex.require_code_block,
                    ex.require_no_commentary,
                    ex.output_format,
                    ex.entity_type,
                ],
            )
        )
    )


def load_eval_samples(dataset_dir: pathlib.Path) -> list[IFStructSample]:
    samples = [
        IFStructSample(
            id=_sample_id(ex),
            seed=ex.seed,
            prompt=ex.prompt,
            json_schema=ex.json_schema,
            top_level_count=ex.top_level_count,
            top_level_key=ex.top_level_key,
            require_wrapper_key=ex.require_wrapper_key,
            require_code_block=ex.require_code_block,
            require_no_commentary=ex.require_no_commentary,
            output_format=ex.output_format,
            entity_type=ex.entity_type,
        )
        for ex in load_examples(dataset_dir / "test.jsonl")
    ]
    return expand(samples, read_repeats(dataset_dir))


def load_prompt_samples(dataset_dir: pathlib.Path) -> list[EvalSample]:
    samples = [
        EvalSample(
            id=_sample_id(ex),
            messages=[ChatMessage(role="user", content=ex.prompt)],
        )
        for ex in load_examples(dataset_dir / "test.jsonl")
    ]
    return expand(samples, read_repeats(dataset_dir))
