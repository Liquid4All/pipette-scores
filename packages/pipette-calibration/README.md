# pipette-calibration

Generate model completions on evaluation datasets and select representative
subsets for calibration.

## Evals

ifbench, ifstruct, gpqa_diamond, math_500.

## CLI

```
pipette-calibration create-initial-dataset        --eval <eval>
pipette-calibration generate-completions          --eval <eval> --model <org/repo> --output-dir <dir>
pipette-calibration verify-completions            --eval <eval> --results-path <dir>
pipette-calibration create-representative-dataset --eval <eval> --results-path <dir> --dataset <name>

pipette-calibration slurm-completions models                  # list frozen model registry
pipette-calibration slurm-completions submit --eval <eval>    # sbatch one eval across the registry
pipette-calibration slurm-completions status --eval <eval>    # verify a finished run
```

## Docs

- [Architecture](docs/architecture.md) — pipeline steps, data flow, repo boundaries
- [Creating a dataset](docs/creating-a-dataset.md) — end-to-end workflow, scoring, calibration process
- [Slurm](docs/slurm.md) — running completions on a cluster, troubleshooting
- [Verification](docs/verification.md) — checking completions for issues
- [Representative selection](docs/representative-selection.md) — subset selection algorithm
