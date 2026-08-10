"""`pipette-calibration slurm <subcommand>` wiring."""

import argparse
import pathlib

from pipette_scores.types import EvalId

from pipette_calibration.slurm.models import MODELS
from pipette_calibration.slurm.status import status as _status
from pipette_calibration.slurm.submit import submit as _submit


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    slurm_p = subparsers.add_parser("slurm-completions", help="Slurm calibration-run infra")
    slurm_subs = slurm_p.add_subparsers(dest="slurm_command", required=True)

    models_p = slurm_subs.add_parser("models", help="List the frozen model registry")
    models_p.set_defaults(func=_cmd_models)

    submit_p = slurm_subs.add_parser("submit", help="Submit sbatch jobs for one eval")
    submit_p.add_argument("--eval", choices=list(EvalId), required=True)
    submit_p.add_argument(
        "--model", action="append", default=None, help="Restrict to this hf_id (repeatable; default = all)"
    )
    submit_p.add_argument("--dataset", default="default")
    submit_p.add_argument("--exclude-nodes", default=None, help="Passed to sbatch --exclude")
    submit_p.add_argument("--dry-run", action="store_true")
    submit_p.set_defaults(func=_cmd_submit)

    status_p = slurm_subs.add_parser("status", help="Verify completions for one eval")
    status_p.add_argument("--eval", choices=list(EvalId), required=True)
    status_p.add_argument("--dataset", default="default")
    status_p.add_argument(
        "--results-path",
        type=pathlib.Path,
        default=pathlib.Path.home() / "lambdafs" / "calibration" / "results",
    )
    status_p.set_defaults(func=_cmd_status)


def _cmd_models(args):
    for m in MODELS:
        print(f"  {m.hf_id:<45s}  mem={m.mem:<4s}  shards={m.base_shards}")


def _cmd_submit(args):
    _submit(
        eval_id=args.eval,
        models=args.model,
        dataset=args.dataset,
        exclude_nodes=args.exclude_nodes,
        dry_run=args.dry_run,
    )


def _cmd_status(args):
    ok = _status(eval_id=args.eval, dataset=args.dataset, results_path=args.results_path)
    if not ok:
        raise SystemExit(1)
