import hashlib
import json
import pathlib

from pipette_scores.repeats import expand, read_repeats
from pipette_scores.types import ChatMessage, EvalSample, IFBenchSample
from pipette_scores.hashing import short_hash
from pipette_scores.parquet_io import read_parquet

_DATA_FILE = "train.parquet"


def content_sha256(rows: list[dict]) -> str:
    """Fingerprint of the dataset's logical content.

    IFBench rows carry nested kwargs, so each row is hashed as sorted-key JSON
    (not a flat tuple like the MCQ evals). Computed over the parquet round-trip
    so it matches what the loader sees. The builder writes this to metadata.json
    and the loader re-verifies it, so a materialized blob is provably canonical.
    """
    canonical = "\n".join(sorted(json.dumps(r, sort_keys=True, default=str) for r in rows))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_rows(dataset_dir: pathlib.Path) -> list[dict]:
    parquet = dataset_dir / _DATA_FILE
    if not parquet.exists():
        raise FileNotFoundError(
            f"IFBench dataset not materialized: {parquet} is missing. Run "
            "scripts/build_ifbench_dataset.py, or point PIPETTE_SCORES_DATA_DIR "
            "at a prepared copy."
        )
    rows = read_parquet(parquet)

    # Skip when no fingerprint is recorded (e.g. a metadata-less test fixture),
    # mirroring read_repeats' graceful handling of a missing metadata.json.
    meta_path = dataset_dir / "metadata.json"
    expected = json.loads(meta_path.read_text()).get("content_sha256") if meta_path.exists() else None
    if expected:
        actual = content_sha256(rows)
        if actual != expected:
            raise ValueError(
                f"IFBench integrity check failed for {dataset_dir}: content_sha256 "
                f"{actual} != {expected} from metadata.json — the materialized dataset "
                "doesn't match the committed fingerprint."
            )
    return rows


def load_eval_samples(dataset_dir: pathlib.Path) -> list[IFBenchSample]:
    samples = [
        IFBenchSample(
            id=short_hash(row["prompt"]),
            key=str(row["key"]),
            instruction_id_list=tuple(row["instruction_id_list"]),
            prompt=row["prompt"],
            kwargs=tuple(row["kwargs"]),
            n_constraints=row.get("n_constraints") or len(row["instruction_id_list"]),
        )
        for row in _read_rows(dataset_dir)
    ]
    return expand(samples, read_repeats(dataset_dir))


def load_prompt_samples(dataset_dir: pathlib.Path) -> list[EvalSample]:
    samples = [
        EvalSample(
            id=short_hash(row["prompt"]),
            messages=[ChatMessage(role="user", content=row["prompt"])],
        )
        for row in _read_rows(dataset_dir)
    ]
    return expand(samples, read_repeats(dataset_dir))
