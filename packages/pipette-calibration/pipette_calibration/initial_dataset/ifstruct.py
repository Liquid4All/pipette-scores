from pathlib import Path

from ifstruct.dataset import load_examples

# vendor/ifstruct/data/test.jsonl, relative to the workspace root.
# __file__ → ifstruct.py → initial_dataset → pipette_calibration → pipette-calibration → packages → <repo>
_VENDOR_JSONL = Path(__file__).resolve().parents[4] / "vendor" / "ifstruct" / "data" / "test.jsonl"


def load_upstream():
    rows = [
        {
            "seed": ex.seed,
            "entity_type": ex.entity_type,
            "prompt": ex.prompt,
            "json_schema": ex.json_schema,
            "top_level_count": ex.top_level_count,
            "top_level_key": ex.top_level_key,
            "require_wrapper_key": ex.require_wrapper_key,
            "require_code_block": ex.require_code_block,
            "require_no_commentary": ex.require_no_commentary,
            "output_format": ex.output_format,
        }
        for ex in load_examples(_VENDOR_JSONL)
    ]
    return {"test": rows}
