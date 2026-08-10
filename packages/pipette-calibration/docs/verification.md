# Verification

## 1. Check one eval across the registry

```bash
pipette-calibration slurm-completions status --eval <eval> [--dataset NAME] [--results-path DIR]
```

Runs `verify_eval()` against the frozen `MODELS` registry
(`pipette_calibration/slurm/models.py`) and renders a rich table. Defaults:
`~/lambdafs/calibration/results`, dataset `default`. Exits non-zero if any
model has issues.

## 2. Check only models present on disk

```bash
uv run pipette-calibration verify-completions --eval <eval> --results-path <dir>
```

Skips the registry; reports only what's already on disk. Exits non-zero on
failure.

## 3. Table cells

| Cell                 | Style  | Meaning                              |
|----------------------|--------|--------------------------------------|
| `ok`                 | green  | Complete, no issues                  |
| `MISSING`            | red    | No results directory on disk         |
| `N/M (K missing)`    | yellow | K sample IDs absent from completions |
| `N dups`             | yellow | Duplicate IDs across shards          |
| `all N empty`        | red    | Every completion is blank            |
| `E/N empty (P%)`     | yellow | Some completions blank               |

Cells can show multiple issues stacked.

## 4. Package API

```python
from pipette_calibration.verify_completions import ModelVerification, verify_eval

results = verify_eval("math_500", "default", Path("_calibration_results"))
```

### 4.1 `ModelVerification` fields

| Field        | Type | Description                                       |
|--------------|------|---------------------------------------------------|
| `model`      | str  | Disk name (`org--repo`)                           |
| `eval_id`    | str  | Eval identifier                                   |
| `total`      | int  | Expected samples from dataset                     |
| `have`       | int  | Matched completion IDs                            |
| `missing`    | int  | Expected IDs not in completions                   |
| `extra`      | int  | Completion IDs not in dataset                     |
| `duplicates` | int  | Duplicate IDs across shards                       |
| `empty`      | int  | Completions with blank text                       |
| `ok`         | bool | `missing == 0 and extra == 0 and duplicates == 0` |
| `status`     | str  | `ok`/`missing`/`incomplete`/`duplicates`/`zero-score`/`partial-empty` |

### 4.2 Functions

- `verify_model(eval_id, dataset_dir, expected_ids)` — one model, one eval
- `verify_eval(eval_id, dataset, results_path)` — all models on disk for one eval
- `verify_completions(eval_id, dataset, results_path)` — bool wrapper for CLI
