# Architecture

## 1. Pipeline overview

```
                          pipette-calibration
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │ 1. Build     │    │ 2. Generate      │    │ 3. Verify        │   │
│  │    datasets  │───>│    completions   │───>│    completions   │   │
│  │              │    │                  │    │                  │   │
│  │ upstream ->  │    │ vLLM on GPU      │    │ check IDs, dups, │   │
│  │ datasets/    │    │ (Slurm cluster)  │    │ empty responses  │   │
│  └──────────────┘    └──────────────────┘    └────────┬─────────┘   │
│                                                       │             │
│                      ┌──────────────────┐             │             │
│                      │ 4. Score         │<────────────┘             │
│                      │    completions   │                           │
│                      │                  │  pipette-scores           │
│                      │ per-model pass/  │  (scoring library)        │
│                      │ fail vectors     │                           │
│                      └────────┬─────────┘                           │
│                               │                                     │
│                      ┌────────v─────────┐                           │
│                      │ 5. Select        │                           │
│                      │    representative│                           │
│                      │    subset        │                           │
│                      │                  │                           │
│                      │ greedy + local   │                           │
│                      │ search           │                           │
│                      └────────┬─────────┘                           │
│                               │                                     │
└───────────────────────────────┼─────────────────────────────────────┘
                                │
                                v
                  <repo>/datasets/
                  <eval>/<dataset>/<split file>
```

## 2. Steps

### 2.1 Build datasets

The build scripts in `packages/pipette-scores/scripts/` write the source
datasets into the dataset catalog. `create-initial-dataset` builds IFStruct
from the vendored IFStruct data.

### 2.2 Generate completions

`generate-completions` loads a model via vLLM, applies the eval's prompt
template, and writes one JSONL line per sample. On a Slurm cluster, this is
parallelised across array tasks (shards), each processing a slice of the
dataset. See [slurm.md](slurm.md).

### 2.3 Verify completions

`verify-completions` and `slurm-completions status` compare completion IDs
against the dataset and check for duplicates. The two commands use different
pass rules. See [verification.md](verification.md).

### 2.4 Score completions

Runs inside `create-representative-dataset`. For each model, completions are
scored against ground truth to produce a boolean pass/fail per sample. Scoring
runs in parallel subprocesses. See [creating-a-dataset.md §5](creating-a-dataset.md#5-calibration-process).

### 2.5 Select representative subset

The scored pass vectors (S samples x M models) are fed into the selection
algorithm. The goal: pick N samples (default 200) where each model's subset
accuracy is close to its full-set accuracy. Output is a new dataset in the
dataset catalog. See [representative-selection.md](representative-selection.md).

## 3. Data flow

```
upstream data
        │
        v
datasets/<eval>/<source>/<split file>           # full dataset
        │
        v (vLLM inference, per model)
<results>/<eval>/<org--model>/<source>/
        completions.shard*.jsonl                # raw completions (completions.jsonl if unsharded)
        │
        v (scoring + selection)
datasets/<eval>/<dataset>/<split file>          # representative subset
                          metadata.json         # models used, skipped
```

`datasets/` is at the repository root, or at `$PIPETTE_SCORES_DATA_DIR` if
that variable is set. The split file is `test.jsonl` for ifstruct,
`train.parquet` for ifbench, and `test.parquet` for the other evals.

## 4. Package boundaries

Both packages are members of one uv workspace.

| Concern               | Package                 |
|-----------------------|-------------------------|
| Dataset catalog       | pipette-scores          |
| Scoring logic         | pipette-scores          |
| Prompt templates      | pipette-scores          |
| Completion generation | pipette-calibration     |
| Verification          | pipette-calibration     |
| Subset selection      | pipette-calibration     |
| Slurm scripts         | pipette-calibration     |
