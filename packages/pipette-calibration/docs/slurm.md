# Slurm

## 1. Model registry

Frozen in code at `pipette_calibration/slurm/models.py`. Each entry is a
`ModelSpec(hf_id, mem, base_shards)` — host RAM (`sbatch --mem`) and base
shard count. Adding a model is a PR to this file.

```bash
pipette-calibration slurm-completions models         # list the registry
```

## 2. Submitting jobs

```bash
pipette-calibration slurm-completions submit --eval ifstruct                               # all models, eval ifstruct
pipette-calibration slurm-completions submit --eval math_500 --dataset 2026.06.1 --model Qwen/Qwen3-0.6B # one model
pipette-calibration slurm-completions submit --eval ifstruct --model A --model B           # subset (repeat --model)
pipette-calibration slurm-completions submit --eval ifstruct --dry-run                     # preview sbatch commands
pipette-calibration slurm-completions submit --eval ifstruct --exclude-nodes node42,node43 # sbatch --exclude
```

`submit` pre-downloads each model on the head node. Then it runs one `sbatch`
per (eval, model), with `--array` if the job has more than one shard. Jobs run
in HF offline mode, so concurrent starts do not call the HF API.

`--dataset` sets the source dataset, called `<source>` below (default:
`default`). Set it to a dataset name that exists for the eval, for example
`2026.06.1`.

## 3. Worker

`pipette_calibration/slurm/run_eval.sh` is the sbatch body — 1 GPU, 4 CPUs,
24h limit. It loads the cluster's CUDA toolkit module and exports the HF
offline env vars. Then it execs `pipette-calibration generate-completions`
with the shard flags.

In this document, `<shared-root>` is the shared filesystem directory under
`$HOME` that `run_eval.sh`, `submit.py`, and `slurm/cli.py` use. The scripts
put the HF cache, the Triton and TorchInductor caches, results, and logs there.

Results land in `<shared-root>/calibration/results/<eval>/<org--model>/<dataset>/`.
Logs go to `<shared-root>/logs/<jobname>-<jobid>_<arrayid>.{out,err}`.

## 4. Sharding

`base_shards` controls parallelism. Larger models need more shards to stay
within the 24h time limit. Each shard writes `completions.shard<N>.jsonl`;
scoring and verification read all shards transparently.

A dataset with a name other than `default` always runs as one job, with no
array. This rule also applies to source datasets such as `2026.06.1`.

## 5. Adding a model

1. Append a `ModelSpec(...)` row to `MODELS` in
   `pipette_calibration/slurm/models.py`.
2. `pipette-calibration slurm-completions submit --eval <eval> --dataset <source> --model org/repo`
3. After jobs complete: `pipette-calibration slurm-completions status --eval <eval> --dataset <source>`

## 6. Resource guidelines

`MODELS` in `pipette_calibration/slurm/models.py` holds the `mem` and
`base_shards` values for each model. Use the entries for models of a similar
size as a starting point for a new model.

## 7. Results layout

```
<shared-root>/calibration/results/
  <eval>/
    <org--model>/
      <dataset>/
        completions.shard0.jsonl
        completions.shard1.jsonl
        ...
```

## 8. Troubleshooting

### 8.1 `trust_remote_code` errors

Models with custom architectures (Mamba hybrids, custom configs) are
auto-detected via `auto_map` in the HF model config or tokenizer config. If a
new model fails with a `trust_remote_code=True` error, verify that one of these
configs contains `auto_map`.

### 8.2 Triton cache corruption

```
FileNotFoundError: .../triton/.../_chunk_scan_fwd_kernel.ptx
```

`run_eval.sh` sets `TRITON_CACHE_DIR` to a Triton cache under `<shared-root>`.
All jobs share this cache. To fix the error, clear that directory and resubmit
the failed jobs.

### 8.3 Stale HF modules cache

```
ImportError: cannot import name 'RopeParameters' from 'transformers.modeling_rope_utils'
```

A previously downloaded model left incompatible Python files in the shared
cache. Fix: delete `modules/transformers_modules/` in the HF cache under
`<shared-root>`.

### 8.4 Incomplete shards

`pipette-calibration slurm-completions status --eval <eval>` shows the
missing sample count for each model.
Check Slurm logs: `<shared-root>/logs/<jobname>-<jobid>_<arrayid>.err`. Common
causes: OOM (bump `mem` in the registry), timeout (bump `base_shards`), node
failures. Resubmit:
`pipette-calibration slurm-completions submit --eval <eval> --dataset <source> --model org/repo`.

### 8.5 `BrokenProcessPool` during dataset creation

A scoring subprocess was killed (typically OOM). The pool stops, and all
models that are not yet scored are excluded. Re-run the command. It scores
all models again.

### 8.6 Model scores 0% and is excluded

Models with zero accuracy are excluded from representative dataset selection.
Check if all completions are empty (context window too small) or if the
model's chat template needs specific formatting.
