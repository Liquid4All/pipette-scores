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
pipette-calibration slurm-completions submit --eval math_500 --model Qwen/Qwen3-0.6B       # one model
pipette-calibration slurm-completions submit --eval ifstruct --model A --model B           # subset (repeat --model)
pipette-calibration slurm-completions submit --eval ifstruct --dry-run                     # preview sbatch commands
pipette-calibration slurm-completions submit --eval ifstruct --exclude-nodes node42,node43 # sbatch --exclude
```

`submit` pre-downloads each model on the head node, then fires one `sbatch
--array` per (eval, model). Array tasks run in HF offline mode so the API
does not get hit by N concurrent starts.

## 3. Worker

`pipette_calibration/slurm/run_eval.sh` is the sbatch body — 1 GPU, 4 CPUs,
24h limit. It loads `cuda13.0/toolkit/13.0.2`, exports the HF offline env
vars, then execs `pipette-calibration generate-completions` with the shard
flags.

Results land in `~/lambdafs/calibration/results/<eval>/<org--model>/<dataset>/`.
Logs go to `~/lambdafs/logs/<jobname>-<jobid>_<arrayid>.{out,err}`.

## 4. Sharding

`base_shards` controls parallelism. Larger models need more shards to stay
within the 24h time limit. Each shard writes `completions.shard<N>.jsonl`;
scoring and verification read all shards transparently.

Non-default datasets (representative subsets, ~200 samples) collapse to a
single shard automatically — no fan-out.

## 5. Adding a model

1. Append a `ModelSpec(...)` row to `MODELS` in
   `pipette_calibration/slurm/models.py`.
2. `pipette-calibration slurm-completions submit --eval <eval> --model org/repo`
3. After jobs complete: `pipette-calibration slurm-completions status --eval <eval>`

## 6. Resource guidelines

| Model size  | mem  | base_shards |
|-------------|------|-------------|
| < 500M      | 16G  | 3           |
| 500M - 1.5B | 16G  | 4           |
| 1.5B - 3B   | 16G  | 6           |
| 3B - 5B     | 24G  | 6-8         |
| 5B - 9B     | 32G  | 8-16        |
| > 9B        | 32G  | 16          |

## 7. Results layout

```
~/lambdafs/calibration/results/
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
auto-detected via `auto_map` in the HF config. If a new model fails with a
`trust_remote_code=True` error, verify its `config.json` contains `auto_map`.

### 8.2 Triton cache corruption

```
FileNotFoundError: .triton/cache/.../_chunk_scan_fwd_kernel.ptx
```

Clear the cache: `rm -rf ~/.triton/cache/`. To prevent on shared filesystems,
set a per-job cache dir:

```bash
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID:-$$}"
```

### 8.3 Stale HF modules cache

```
ImportError: cannot import name 'RopeParameters' from 'transformers.modeling_rope_utils'
```

A previously downloaded model left incompatible Python files in the shared
cache. Fix: `rm -rf ~/lambdafs/.cache/huggingface/modules/transformers_modules/`

### 8.4 Incomplete shards

`pipette-calibration slurm-completions status --eval <eval>` shows missing shards.
Check Slurm logs: `~/lambdafs/logs/<jobname>-<jobid>_<arrayid>.err`. Common
causes: OOM (bump `mem` in the registry), timeout (bump `base_shards`), node
failures. Resubmit:
`pipette-calibration slurm-completions submit --eval <eval> --model org/repo`.

### 8.5 `BrokenProcessPool` during dataset creation

A scoring subprocess was killed (typically OOM). The killed model is excluded;
re-run the command — already-completed models are fast on retry.

### 8.6 Model scores 0% and is excluded

Models with zero accuracy are excluded from representative dataset selection.
Check if all completions are empty (context window too small) or if the
model's chat template needs specific formatting.
