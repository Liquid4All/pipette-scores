# Representative Dataset Selection

Select a subset of N samples (typically 200) from a full evaluation dataset such
that every model's accuracy on the subset closely matches its full-set accuracy.

## 1. Inputs

- Evaluation results for M models, each scored on the full dataset of S samples.
- Group labels per sample (e.g. IFStruct entity type or GPQA answer).
- Target size N (default 200).

## 2. Data structure

Each sample is represented by a pass vector — a tuple of M booleans indicating
whether each model answered correctly. The selection objective operates entirely
on these vectors without needing the raw completions.

## 3. Objective

Minimise the max deviation across all models:

    max_m |subset_acc_m - full_acc_m|

where `subset_acc_m = correct_m / N` and `full_acc_m` is computed over the full
dataset.

## 4. Algorithm

Cell-quota greedy + local search:

Used by: ifbench, ifstruct, gpqa_diamond, math_500.

**Cell construction.** Samples are stratified by group label and within-group
difficulty bin, where difficulty is estimated from each sample's pass count
across the model panel.

**Quota allocation.** Each cell receives a target count proportional to its
full-dataset size, capped by available samples and adjusted to sum to N.

**Greedy forward selection.** Within each cell, repeatedly pick the candidate
that minimises max deviation from full-set model accuracies.

**Local search.** For each selected sample, find the unselected sample in the
same cell that gives the lowest max deviation. Swap them if the deviation
decreases. Stop when a round makes no swap, or after 20 rounds.

## 5. Orchestration

`create_representative_dataset()`:

1. Discover model result dirs under `results_path/eval_id/*/source_dataset/`.
2. Score all models in parallel (`ProcessPoolExecutor`, at most one subprocess for each CPU).
   Models with incomplete results or 0% accuracy are excluded.
3. Build pass vectors (S x M boolean matrix).
4. Run selection.
5. Save the selected rows as a new dataset in the source dataset's file format.
   Write `metadata.json` in the same directory.

## 6. Source

`pipette_calibration/calibration/representative.py`
