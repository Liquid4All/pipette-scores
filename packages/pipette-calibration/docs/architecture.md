# Architecture

## 1. Pipeline overview

```
                          pipette-calibration
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │ 1. Download   │    │ 2. Generate       │    │ 3. Verify        │  │
│  │    datasets   │───>│    completions    │───>│    completions   │  │
│  │              │    │                  │    │                  │  │
│  │ HuggingFace  │    │ vLLM on GPU      │    │ check IDs, dups, │  │
│  │ -> parquet   │    │ (Slurm cluster)  │    │ empty responses  │  │
│  └──────────────┘    └──────────────────┘    └────────┬─────────┘  │
│                                                        │            │
│                      ┌──────────────────┐              │            │
│                      │ 4. Score          │<─────────────┘            │
│                      │    completions    │                           │
│                      │                  │  pipette-scores        │
│                      │ per-model pass/  │  (scoring library)        │
│                      │ fail vectors     │                           │
│                      └────────┬─────────┘                           │
│                               │                                     │
│                      ┌────────v─────────┐                           │
│                      │ 5. Select         │                           │
│                      │    representative │                           │
│                      │    subset         │                           │
│                      │                  │                           │
│                      │ greedy + local   │                           │
│                      │ search / SA      │                           │
│                      └────────┬─────────┘                           │
│                               │                                     │
└───────────────────────────────┼─────────────────────────────────────┘
                                │
                                v
                  pipette-scores/datasets/
                  <eval>/<dataset>/test.parquet
```

## 2. Steps

### 2.1 Download datasets

`create-initial-dataset` fetches upstream data from HuggingFace and writes
parquet files into the `pipette-scores` dataset catalog. Run once per eval.

### 2.2 Generate completions

`generate-completions` loads a model via vLLM, applies the eval's prompt
template, and writes one JSONL line per sample. On a Slurm cluster, this is
parallelised across array tasks (shards), each processing a slice of the
dataset. See [slurm.md](slurm.md).

### 2.3 Verify completions

`verify-completions` and `slurm-completions status` compare completion IDs
against the dataset, check for duplicates, and flag empty responses. This
gates whether completions are ready for scoring. See [verification.md](verification.md).

### 2.4 Score completions

Runs inside `create-representative-dataset`. For each model, completions are
scored against ground truth to produce a boolean pass/fail per sample. Scoring
runs in parallel subprocesses. See [creating-a-dataset.md §5](creating-a-dataset.md#5-calibration-process).

### 2.5 Select representative subset

The scored pass vectors (S samples x M models) are fed into the selection
algorithm. The goal: pick N samples (default 200) where each model's subset
accuracy matches its full-set accuracy within ~0.5%. Output is a new parquet
dataset in `pipette-scores`. See [representative-selection.md](representative-selection.md).

## 3. Data flow

```
HuggingFace datasets
        │
        v
datasets/<eval>/<source>/test.parquet          # full dataset (source="default")
        │
        v (vLLM inference, per model)
_calibration_results/<eval>/<model>/<source>/
        completions.shard*.jsonl                # raw completions
        │
        v (scoring + selection)
datasets/<eval>/<dataset>/test.parquet          # representative subset
                          metadata.json         # models used, skipped
```

## 4. Repo boundaries

| Concern               | Repo                    |
|-----------------------|-------------------------|
| Dataset catalog       | pipette-scores       |
| Scoring logic         | pipette-scores       |
| Prompt templates      | pipette-scores       |
| Completion generation | pipette-calibration  |
| Verification          | pipette-calibration  |
| Subset selection      | pipette-calibration  |
| Slurm scripts         | pipette-calibration  |
