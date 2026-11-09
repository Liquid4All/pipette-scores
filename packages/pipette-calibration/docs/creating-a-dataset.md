# Creating a Calibration Dataset

End-to-end workflow for producing a new versioned calibration dataset.

## 1. Initialize datasets

Write the source datasets into the dataset catalog at
`<repo>/datasets/<eval>/<dataset>/`. Use the build scripts in
`packages/pipette-scores/scripts/` for ifbench, gpqa_diamond, and math_500.
These scripts write a version directory, for example `2026.06.1`.

For IFStruct, use `create-initial-dataset`. It writes the dataset `default`:

```bash
uv run pipette-calibration create-initial-dataset --eval ifstruct
```

## 2. Generate completions

Run each model on each eval. Locally:

```bash
uv run --group gpu pipette-calibration generate-completions \
    --eval math_500 --model LiquidAI/LFM2-1.2B \
    --dataset 2026.06.1 --output-dir _calibration_results
```

On a Slurm cluster, use `pipette-calibration slurm-completions submit`. See
[slurm.md](slurm.md).

Set `--dataset` to the source dataset, which is the full set of samples. This
document calls it `<source>`. The default value is `default`. Only
`create-initial-dataset` writes a dataset with that name.

## 3. Verify completions

Run this command once for each eval:

```bash
pipette-calibration slurm-completions status --eval <eval> --dataset <source>
```

Every model in the frozen registry (`pipette_calibration/slurm/models.py`)
must be clean across all evals before proceeding. See
[verification.md](verification.md).

## 4. Create representative subset

Run this command once for each eval:

```bash
uv run pipette-calibration create-representative-dataset \
    --eval <eval> \
    --source-dataset <source> \
    --results-path _calibration_results \
    --dataset 2026.03.2
```

This command scores all completions, builds pass vectors, and selects the
subset. Scoring and selection details: [calibration process](#5-calibration-process).

### 4.1 Key flags

| Flag                  | Default   | Description                            |
|-----------------------|-----------|----------------------------------------|
| `--dataset`           | required  | Output dataset name                    |
| `--source-dataset`    | `default` | Source dataset to select samples from  |
| `--results-path`      | required  | Directory containing model completions |
| `--num-samples`       | 200       | Target subset size                     |

`--dataset` is the name of the output (e.g. `2026.03.2`).
`--source-dataset` is the full dataset that completions were generated against.

### 4.2 Naming convention

Datasets are named by date: `2026.03.2` (year.month.revision). The revision
increments if a dataset is regenerated within the same month.

## 5. Calibration process

Steps 4's `create-representative-dataset` runs internally:

### 5.1 Scoring

Each model's completions are scored against ground truth using per-eval
scorers from [pipette-scores](https://github.com/Liquid4All/pipette-scores).
Scoring runs in parallel subprocesses (at most one for each CPU) and produces
an S x M boolean pass/fail matrix.

| Eval      | Method                                           |
|-----------|--------------------------------------------------|
| ifbench   | Instruction-following constraint checks           |
| ifstruct  | YAML/JSON schema validation against target spec  |
| gpqa_diamond | Multiple-choice answer extraction             |
| math_500  | LaTeX answer extraction and symbolic comparison   |

### 5.2 Model exclusions

Models are automatically excluded if:

- Scoring raises an error for the model.
- A scoring subprocess is killed (for example, by OOM). The pool then stops,
  and all models that are not yet scored are excluded.
- Completions are missing sample IDs (incomplete run).
- Model scores 0% (no discriminative signal).

Exclusions are recorded in `metadata.json` alongside the output dataset.

### 5.3 Selection

The pass vectors feed into the selection algorithm — see
[representative-selection.md](representative-selection.md) for algorithm
details.

## 6. Output

Each eval produces files in `<repo>/datasets/<eval>/<dataset>/`:

- Split file (`train.parquet`, `test.parquet`, or `test.jsonl`) — the selected samples.
- `metadata.json` — models used, models skipped (with reasons).

## 7. What to check

Review the log output for each eval:

- **Max deviation** — the log shows the deviation for each model and the maximum. Check it.
- **Skipped models** — check `metadata.json` for the full list and reasons.
- **Sample count** — should match `--num-samples` (200).

## 8. Committing the output

The output datasets are in `<repo>/datasets/`. After you generate them:

1. Review the new dataset files and metadata.
2. Commit only the ifstruct output. The repository does not redistribute the
   gpqa_diamond, math_500, and ifbench data, and `.gitignore` excludes them.
3. Tag the commit if this is a release dataset.

## 9. Re-running with different models

If new models are added after a dataset was created:

1. Generate completions for the new models (see [slurm.md §5](slurm.md#5-adding-a-model)).
2. Re-run `create-representative-dataset` with the same dataset name.
   It overwrites the previous selection using all available models.
3. The new `metadata.json` reflects the updated model set.
