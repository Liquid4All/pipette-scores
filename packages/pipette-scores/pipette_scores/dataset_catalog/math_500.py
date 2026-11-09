import hashlib
import json
import pathlib

from pipette_scores.repeats import expand, read_repeats
from pipette_scores.types import ChatMessage, EvalSample, Math500Grader, Math500Sample
from pipette_scores.hashing import short_hash
from pipette_scores.parquet_io import read_parquet

_DATA_FILE = "test.parquet"


def content_sha256(rows: list[tuple[str, str]]) -> str:
    """Fingerprint of the dataset's logical content.

    Hashes the canonical (prompt, answer) rows, not the parquet bytes — parquet
    encoding isn't stable across writer versions, but the logical content is.
    The builder writes this to metadata.json and the loader re-verifies it, so a
    materialized blob is provably the canonical dataset.
    """
    canonical = "\n".join(sorted(f"{prompt}\x1f{answer}" for prompt, answer in rows))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_rows(dataset_dir: pathlib.Path) -> list[dict]:
    parquet = dataset_dir / _DATA_FILE
    if not parquet.exists():
        raise FileNotFoundError(
            f"MATH-500 dataset not materialized: {parquet} is missing. Run "
            "scripts/build_math_500_dataset.py, or point PIPETTE_SCORES_DATA_DIR "
            "at a prepared copy."
        )
    rows = read_parquet(parquet)

    expected = json.loads((dataset_dir / "metadata.json").read_text()).get("content_sha256")
    if expected:
        actual = content_sha256([(r["prompt"], str(r["answer"])) for r in rows])
        if actual != expected:
            raise ValueError(
                f"MATH-500 integrity check failed for {dataset_dir}: content_sha256 "
                f"{actual} != {expected} from metadata.json — the materialized dataset "
                "doesn't match the committed fingerprint."
            )
    return rows


def read_grader(dataset_dir: pathlib.Path) -> Math500Grader:
    """Read ``metadata.grader``: the grader this dataset version is scored with.

    A version without the key (2026.06.1) is scored with ``math_generic``, so
    its results stay reproducible. An unknown value fails the load rather than
    falling back to a grader the dataset was not built for.
    """
    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        return Math500Grader.MATH_GENERIC
    grader = json.loads(meta_path.read_text()).get("grader", Math500Grader.MATH_GENERIC.value)
    try:
        return Math500Grader(grader)
    except ValueError:
        known = ", ".join(g.value for g in Math500Grader)
        raise ValueError(f"metadata.grader must be one of {known}, got {grader!r} in {meta_path}") from None


def load_eval_samples(dataset_dir: pathlib.Path) -> list[Math500Sample]:
    grader = read_grader(dataset_dir)
    samples = [
        Math500Sample(
            id=short_hash(row["prompt"]),
            prompt=row["prompt"],
            answer=str(row["answer"]),
            grader=grader,
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
