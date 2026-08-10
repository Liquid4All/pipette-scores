import argparse
import logging
import pathlib

from pipette_scores.types import EvalId

from pipette_calibration.slurm import cli as slurm_cli

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Edge eval CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- create-initial-dataset --
    create_p = subparsers.add_parser("create-initial-dataset", help="Download upstream data into a named dataset")
    create_p.add_argument("--eval", choices=list(EvalId), required=True)
    create_p.add_argument("--dataset", default="default", help="Target dataset name")
    create_p.set_defaults(func=_create_initial_dataset)

    # -- generate-completions --
    gen_p = subparsers.add_parser(
        "generate-completions", help="Run a model on a dataset and write per-sample completions"
    )
    gen_p.add_argument("--eval", choices=list(EvalId), required=True)
    gen_p.add_argument("--dataset", default="default", help="Dataset name")
    gen_p.add_argument("--model", required=True, help="Model as org/name (e.g. LiquidAI/LFM2-1.2B)")
    gen_p.add_argument("--output-dir", type=pathlib.Path, required=True, help="Output directory")
    gen_p.add_argument("--hf-cache", type=pathlib.Path, default=None)
    gen_p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    gen_p.add_argument("--shard", type=int, default=None)
    gen_p.add_argument("--num-shards", type=int, default=None)
    gen_p.set_defaults(func=_generate_completions)

    # -- create-representative-dataset --
    rep_p = subparsers.add_parser(
        "create-representative-dataset",
        help="Select a representative subset matching full-set accuracy per model",
    )
    rep_p.add_argument("--eval", choices=list(EvalId), required=True)
    rep_p.add_argument("--dataset", required=True, help="New representative dataset name")
    rep_p.add_argument(
        "--source-dataset",
        default="default",
        help="Source dataset to select samples from",
    )
    rep_p.add_argument(
        "--results-path",
        type=pathlib.Path,
        required=True,
        help="Root results directory (contains <eval>/<model>/<dataset>/)",
    )
    rep_p.add_argument("--num-samples", type=int, default=200)
    rep_p.set_defaults(func=_create_representative_dataset)

    # -- verify-completions --
    verify_p = subparsers.add_parser(
        "verify-completions",
        help="Check completions against dataset for missing/extra IDs",
    )
    verify_p.add_argument("--eval", choices=list(EvalId), required=True)
    verify_p.add_argument("--dataset", default="default", help="Dataset name")
    verify_p.add_argument(
        "--results-path",
        type=pathlib.Path,
        required=True,
        help="Root results directory (contains <eval>/<model>/<dataset>/)",
    )
    verify_p.set_defaults(func=_verify_completions)

    # -- slurm <subcommand> --
    slurm_cli.add_subparser(subparsers)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.func(args)


def _create_initial_dataset(args):
    from pipette_calibration.initial_dataset import create_initial_dataset

    total = create_initial_dataset(args.eval, name=args.dataset)
    logger.info("Created dataset '%s' for %s (%d rows)", args.dataset, args.eval, total)


def _generate_completions(args):
    from pipette_calibration.inference import generate_completions

    generate_completions(
        eval_id=args.eval,
        dataset=args.dataset,
        model=args.model,
        output_dir=args.output_dir,
        hf_cache=args.hf_cache,
        gpu_memory_utilization=args.gpu_memory_utilization,
        shard=args.shard,
        num_shards=args.num_shards,
    )


def _create_representative_dataset(args):
    from pipette_calibration.calibration.representative import create_representative_dataset

    count = create_representative_dataset(
        eval_id=args.eval,
        source_dataset=args.source_dataset,
        dataset=args.dataset,
        results_path=args.results_path,
        num_samples=args.num_samples,
    )
    logger.info(
        "Created representative dataset '%s' for %s (%d samples)",
        args.dataset,
        args.eval,
        count,
    )


def _verify_completions(args):
    from pipette_calibration.verify_completions import verify_completions

    ok = verify_completions(
        eval_id=args.eval,
        dataset=args.dataset,
        results_path=args.results_path,
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
