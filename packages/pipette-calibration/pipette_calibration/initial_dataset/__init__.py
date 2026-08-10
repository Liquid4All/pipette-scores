from pipette_scores.dataset_catalog import save_raw_split
from pipette_scores.types import EvalId

from pipette_calibration.initial_dataset import ifstruct


def create_initial_dataset(eval_id: EvalId, name: str = "default") -> int:
    """Download upstream data and save raw rows as a named dataset. Returns row count."""
    loaders = {
        EvalId.IFSTRUCT: ifstruct.load_upstream,
    }
    try:
        load_upstream = loaders[EvalId(eval_id)]
    except KeyError as exc:
        supported = ", ".join(e.value for e in loaders)
        raise NotImplementedError(f"create-initial-dataset supports: {supported}") from exc
    raw_splits = load_upstream()

    total = 0
    for split, rows in raw_splits.items():
        save_raw_split(eval_id, name, split, rows)
        total += len(rows)
    return total
