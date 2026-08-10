import hashlib
import json
import pathlib

from pipette_scores.repeats import expand, read_repeats
from pipette_scores.types import ChatMessage, EvalSample, GPQADiamondSample
from pipette_scores.hashing import short_hash
from pipette_scores.parquet_io import read_parquet

_DATA_FILE = "test.parquet"


def content_sha256(rows: list[tuple[str, str]]) -> str:
    """Fingerprint of the dataset's logical content.

    Hashes the canonical (prompt, answer) rows, not the parquet bytes — parquet
    encoding isn't stable across writer versions, but the logical content is.
    The builder writes this to metadata.json (alongside the dataset; reveals no
    questions) and the loader re-verifies it, so a materialized blob fetched from
    outside git is provably the canonical dataset.
    """
    canonical = "\n".join(sorted(f"{prompt}\x1f{answer}" for prompt, answer in rows))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_rows(dataset_dir: pathlib.Path) -> list[dict]:
    # GPQA is gated and gitignored, so test.parquet is materialized out of band
    # (builder script or PIPETTE_SCORES_DATA_DIR mount), not shipped in the repo.
    parquet = dataset_dir / _DATA_FILE
    if not parquet.exists():
        raise FileNotFoundError(
            f"GPQA Diamond dataset not materialized: {parquet} is missing. GPQA is "
            "gated and not committed. Accept the terms at "
            "https://huggingface.co/datasets/Idavidrein/gpqa and run "
            "scripts/build_gpqa_diamond_dataset.py, or point PIPETTE_SCORES_DATA_DIR "
            "at a prepared copy."
        )
    rows = read_parquet(parquet)

    expected = json.loads((dataset_dir / "metadata.json").read_text()).get("content_sha256")
    if expected:
        actual = content_sha256([(r["prompt"], str(r["answer"])) for r in rows])
        if actual != expected:
            raise ValueError(
                f"GPQA Diamond integrity check failed for {dataset_dir}: content_sha256 "
                f"{actual} != {expected} from metadata.json — the materialized dataset "
                "doesn't match the committed fingerprint."
            )
    return rows


def load_eval_samples(dataset_dir: pathlib.Path) -> list[GPQADiamondSample]:
    samples = [
        GPQADiamondSample(
            id=short_hash(row["prompt"]),
            prompt=row["prompt"],
            answer=str(row["answer"]),
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
