# Creating a Calibration Dataset

End-to-end workflow for producing a new versioned calibration dataset.

## 1. Initialize datasets

Download upstream data into the `pipette-scores` dataset catalog. The
`pipette-calibration create-initial-dataset` command currently supports
IFStruct; the other current evals are materialized by the build scripts in
`packages/pipette-scores/scripts/`.

```bash
uv run pipette-calibration create-initial-dataset --eval ifstruct
```

This writes dataset files to `<pipette-scores>/datasets/<eval>/default/`.

## 2. Generate completions

Run each model on each eval. Locally:

```bash
uv run --group gpu pipette-calibration generate-completions \
    --eval math_500 --model LiquidAI/LFM2-1.2B \
    --dataset default --output-dir _calibration_results
```

On a Slurm cluster, use `submit.sh` instead — see [slurm.md](slurm.md).

Completions are generated against the `--dataset` (default: `default`). This
is the source dataset — the full set of samples.

## 3. Verify completions

```bash
for eval in ifbench ifstruct gpqa_diamond math_500; do
  pipette-calibration slurm-completions status --eval $eval
done
```

Every model in the frozen registry (`pipette_calibration/slurm/models.py`)
must be clean across all evals before proceeding. See
[verification.md](verification.md).

## 4. Create representative subset

```bash
for eval in ifbench ifstruct gpqa_diamond math_500; do
  uv run pipette-calibration create-representative-dataset \
      --eval $eval \
      --results-path _calibration_results \
      --dataset 2026.03.2
done
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
`--source-dataset` is the full dataset that completions were generated against
(usually `default`).

### 4.2 Naming convention

Datasets are named by date: `2026.03.2` (year.month.revision). The revision
increments if a dataset is regenerated within the same month.

## 5. Calibration process

Steps 4's `create-representative-dataset` runs internally:

### 5.1 Scoring

Each model's completions are scored against ground truth using per-eval
scorers from [pipette-scores](https://github.com/Liquid4All/pipette-scores).
Scoring runs in parallel (one subprocess per model), producing an S x M
boolean pass/fail matrix.

| Eval      | Method                                           |
|-----------|--------------------------------------------------|
| ifbench   | Instruction-following constraint checks           |
| ifstruct  | YAML/JSON schema validation against target spec  |
| gpqa_diamond | Multiple-choice answer extraction             |
| math_500  | LaTeX answer extraction and symbolic comparison   |

### 5.2 Model exclusions

Models are automatically excluded if:

- Scoring subprocess crashes (OOM, dependency error).
- Completions are missing sample IDs (incomplete run).
- Model scores 0% (no discriminative signal).

Exclusions are recorded in `metadata.json` alongside the output dataset.

### 5.3 Selection

The pass vectors feed into the selection algorithm — see
[representative-selection.md](representative-selection.md) for algorithm
details.

## 6. Output

Each eval produces files in `<pipette-scores>/datasets/<eval>/<dataset>/`:

- Primary split file (`train.parquet`, `test.parquet`, or `test.jsonl`) — the selected samples.
- Other splits copied unchanged from the source dataset.
- `metadata.json` — models used, models skipped (with reasons).

## 7. What to check

Review the log output for each eval:

- **Max deviation** — should be under 1%. Typical: 0.2-0.5%.
- **Skipped models** — check `metadata.json` for the full list and reasons.
- **Sample count** — should match `--num-samples` (200).

## 8. Updating pipette-scores

The output datasets land in the `pipette-scores` package tree. After generating:

1. Review the new parquet files and metadata.
2. Commit and push in `pipette-scores`.
3. Tag if this is a release dataset.

## 9. Re-running with different models

If new models are added after a dataset was created:

1. Generate completions for the new models (see [slurm.md §5](slurm.md#5-adding-a-model)).
2. Re-run `create-representative-dataset` with the same dataset name.
   It overwrites the previous selection using all available models.
3. The new `metadata.json` reflects the updated model set.
