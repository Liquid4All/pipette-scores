"""CLI for scoring LLM completions against edge evaluation benchmarks."""

import argparse
import json
import logging
import os
import pathlib
import sys
import tempfile

from tabulate import tabulate

from collections import Counter

from pipette_scores import packages, scoring, dataset_catalog
from pipette_scores.dataset_catalog import prompt_profiles
from pipette_scores.scoring import profile_answers
from pipette_scores.types import ASR_EVAL_IDS, EvalId, SampleCompletion

logger = logging.getLogger(__name__)


def cmd_list(args: argparse.Namespace) -> None:
    rows = []
    for eval_id in scoring.list_eval_ids():
        datasets = dataset_catalog.list_dataset_names(eval_id, root_dir=args.data_dir)
        rows.append([eval_id.value, ", ".join(datasets) if datasets else "(none)"])
    print(tabulate(rows, headers=["EVAL", "DATASETS"], tablefmt="simple"))


def cmd_prompts(args: argparse.Namespace) -> None:
    eval_id = EvalId(args.eval)
    # ASR: the profile's requests without their audio (the server streams those).
    samples = dataset_catalog.load_prompt_samples(
        eval_id, args.dataset, root_dir=args.data_dir, prompt_profile=args.prompt_profile
    )

    if args.limit:
        samples = samples[: args.limit]

    print(f"samples: {len(samples)}", file=sys.stderr)

    out = open(args.output, "w") if args.output else sys.stdout
    try:
        for sample in samples:
            out.write(sample.model_dump_json() + "\n")
    finally:
        if out is not sys.stdout:
            out.close()


def _load_completions_jsonl(path: pathlib.Path) -> list[SampleCompletion]:
    """Load completions from a single JSONL file."""
    completions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            completions.append(SampleCompletion.model_validate(obj))
    return completions


def _load_payload_json(path: pathlib.Path) -> list[SampleCompletion]:
    """Load completions from a payload.json file."""
    with open(path) as f:
        data = json.load(f)
    return [SampleCompletion.model_validate(c) for c in data["completions"]]


def _load_completions(path: pathlib.Path) -> list[SampleCompletion]:
    """Load completions from a JSONL file, payload.json, or a directory of shards."""
    if path.is_file():
        if path.suffix == ".json":
            return _load_payload_json(path)
        return _load_completions_jsonl(path)
    if path.is_dir():
        payload_path = path / "payload.json"
        if payload_path.exists():
            return _load_payload_json(payload_path)
        completions = []
        for shard in sorted(path.glob("*.jsonl")):
            loaded = _load_completions_jsonl(shard)
            logger.info("Loaded %d completions from %s", len(loaded), shard.name)
            completions.extend(loaded)
        return completions
    raise FileNotFoundError(path)


def cmd_score(args: argparse.Namespace) -> None:
    completions = _load_completions(pathlib.Path(args.file))
    eval_id = EvalId(args.eval)

    # Load ground truth samples
    samples = dataset_catalog.load_eval_samples(eval_id, args.dataset, root_dir=args.data_dir)

    # The checks /score makes before scoring: one completion per id, each a dataset sample.
    known = {s.id for s in samples}
    if duplicates := sorted(i for i, n in Counter(c.id for c in completions).items() if n > 1):
        raise SystemExit(f"duplicate completion ids: {', '.join(duplicates[:10])}")
    if unknown := sorted({c.id for c in completions} - known):
        raise SystemExit(f"completions reference ids not in {eval_id}/{args.dataset}: {', '.join(unknown[:10])}")
    if eval_id in ASR_EVAL_IDS and args.prompt_profile is None:
        raise SystemExit(f"{eval_id} needs --prompt-profile (the family whose answers these are)")
    if eval_id not in ASR_EVAL_IDS and args.prompt_profile is not None:
        if args.prompt_profile not in prompt_profiles.PROFILES:
            raise SystemExit(
                f"Unknown prompt profile: {args.prompt_profile} (known: {', '.join(prompt_profiles.PROFILES)})"
            )
        if eval_id not in prompt_profiles.BENCHMARKS:
            raise SystemExit(f"{eval_id} has no prompt profiles")
    if eval_id in ASR_EVAL_IDS:
        # A result is an answer to every sample, or no result.
        unanswered = sorted(known - {c.id for c in completions})
        failed = sorted(c.id for c in completions if c.stop_reason == "failure")
        if unanswered or failed:
            gaps = [
                f"{label}: {', '.join(ids[:10])}"
                for label, ids in (("unanswered", unanswered), ("failed", failed))
                if ids
            ]
            raise SystemExit(f"an ASR eval is scored only with an answer to every sample; {'; '.join(gaps)}")

    logger.info("Scoring %d completions for %s/%s", len(completions), eval_id, args.dataset)
    # As /score: grade the answer the profile's wrapper hands on.
    scores, context = scoring.score(
        eval_id,
        profile_answers.apply(args.prompt_profile, eval_id, completions),
        samples,
        enable_thinking=args.enable_thinking,
        prompt_profile=args.prompt_profile,
    )

    total = len(scores)
    correct = sum(1 for v in scores if v.is_correct)
    accuracy = correct / total if total > 0 else 0.0

    # Summary table
    print()
    if eval_id in ASR_EVAL_IDS:
        # The corpus error rate; is_correct ("no error in its unit") is no ASR metric.
        result = "n/a" if context["result"] is None else f"{context['result']:.4f}%"
        print(
            tabulate(
                [[eval_id.value, args.dataset, context["samples_total"], context["units_scored"], result]],
                headers=["EVAL", "DATASET", "SAMPLES", "UNITS", "WER"],
                tablefmt="simple",
            )
        )
    else:
        print(
            tabulate(
                [[eval_id.value, args.dataset, total, correct, f"{accuracy:.2%}"]],
                headers=["EVAL", "DATASET", "TOTAL", "CORRECT", "ACCURACY"],
                tablefmt="simple",
            )
        )

    # Eval-specific context table
    if context:
        print()
        rows = [[k, f"{v:.4f}" if isinstance(v, float) else v] for k, v in context.items()]
        print(tabulate(rows, headers=["METRIC", "VALUE"], tablefmt="simple"))

    # Per-sample results table
    if not args.no_samples:
        print()
        sample_rows = [[v.id, "PASS" if v.is_correct else "FAIL"] for v in scores]
        print(tabulate(sample_rows, headers=["SAMPLE", "RESULT"], tablefmt="simple"))

    # Write full JSON to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "eval_id": eval_id.value,
                    "dataset": args.dataset,
                    "total": total,
                    "correct": correct,
                    "accuracy": accuracy,
                    "context": context,
                    "scores": [v.model_dump() for v in scores],
                },
                f,
                indent=2,
            )
            f.write("\n")
        print(f"\nFull results written to {args.output}", file=sys.stderr)


def cmd_packages(args: argparse.Namespace) -> None:
    # The index is keyed by the release the scores server runs, which looks it up under its own version.
    if not args.version or args.version == "dev":
        raise SystemExit("packages needs --version or PIPETTE_SCORES_VERSION: the scores release, not dev")
    store = packages.open_store(args.packages)
    remote = packages.open_store(args.datasets) if args.datasets.startswith("s3://") else None
    if remote is None and not pathlib.Path(args.datasets, "datasets").is_dir():
        raise SystemExit(f"{args.datasets} has no datasets/ directory")
    failed = []
    for eval_id in args.eval or [e.value for e in EvalId if packages.profiles(e)]:
        # Each dataset version, with the keys to download for it (none: a local one is read in place).
        if remote is None:
            versions = {name: [] for name in dataset_catalog.list_dataset_names(eval_id, root_dir=args.datasets)}
        else:
            versions = {}
            for key in remote.list(f"{eval_id}/"):
                versions.setdefault(key.split("/")[1], []).append(key)
        for name, keys in sorted(versions.items()):
            if store.read(packages.index_key(args.version, eval_id, name)) is not None:
                logger.info("eval=%s dataset=%s indexed for %s already, skipped", eval_id, name, args.version)
                continue
            try:
                with tempfile.TemporaryDirectory(prefix="datasets-") as tmp:
                    for key in keys:
                        remote.download(key, pathlib.Path(tmp, "datasets", key))
                    packages.build(eval_id, name, args.datasets if remote is None else tmp, store, args.version)
            except Exception:
                logger.exception("eval=%s dataset=%s failed", eval_id, name)
                failed.append(f"{eval_id}/{name}")
    if failed:
        raise SystemExit(f"{len(failed)} datasets failed: {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pipette-scores",
        description="Score LLM completions against edge evaluation benchmarks.",
    )
    parser.add_argument("--data-dir", default=None, help="Override dataset root directory")

    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List available evals and their datasets")

    # prompts
    p_prompts = sub.add_parser("prompts", help="Export prompt samples to JSONL")
    p_prompts.add_argument("--eval", required=True, help="Eval ID (e.g. ifbench, ifstruct)")
    p_prompts.add_argument("--dataset", required=True, help="Dataset name (e.g. default)")
    p_prompts.add_argument("--limit", type=int, default=None, help="Max number of samples to export")
    p_prompts.add_argument(
        "--prompt-profile",
        default=None,
        help="VLM: a model family's prompts (optional). ASR: required; the requests without their audio",
    )
    p_prompts.add_argument("-o", "--output", default=None, help="Output file (default: stdout)")

    # score
    p_score = sub.add_parser(
        "score",
        help="Score completions from a file or directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
supported input formats:
  JSONL file       one {"id": "...", "completion": "...", "stop_reason"?: "..."} per line
  payload.json     {"completions": [...]}
  directory        looks for payload.json first, then *.jsonl shards

examples:
  %(prog)s --eval ifbench --dataset default -f completions.jsonl
  %(prog)s --eval ifstruct --dataset edge_2026.03.2 -f results/payload.json
  %(prog)s --eval ifbench --dataset default -f results/shards_dir/
  %(prog)s --eval asr_fleurs_nl_nl --dataset 2026.10.1 --prompt-profile voxtral -f completions.jsonl
""",
    )
    p_score.add_argument("--eval", required=True, help="Eval ID (e.g. ifbench, ifstruct)")
    p_score.add_argument("--dataset", required=True, help="Dataset name (e.g. default, edge_2026.03.2)")
    p_score.add_argument("-f", "--file", required=True, help="JSONL file, payload.json, or directory")
    p_score.add_argument("-o", "--output", default=None, help="Write full JSON results to file")
    p_score.add_argument("--no-samples", action="store_true", help="Hide per-sample results table")
    p_score.add_argument(
        "--prompt-profile",
        default=None,
        help="The run's prompt profile, as /score takes it (required for ASR evals: it cleans each answer)",
    )
    p_score.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="The job's enable_thinking model flag, as /score takes it (default: unset)",
    )

    # packages
    p_packages = sub.add_parser(
        "packages",
        help="Pack every VLM and ASR prompt set into the eval packages mgmt serves (docs/packages.md)",
    )
    p_packages.add_argument(
        "--datasets", required=True, help="s3://<bucket>/<prefix> of <eval>/<version>/, or a data dir with datasets/"
    )
    p_packages.add_argument("--packages", required=True, help="s3://<bucket>/<prefix> or a directory to store into")
    p_packages.add_argument(
        "--eval", nargs="+", choices=[e.value for e in EvalId if packages.profiles(e)], help="Only these evals"
    )
    p_packages.add_argument(
        "--version",
        default=os.environ.get("PIPETTE_SCORES_VERSION"),
        help="The scores release the index is written for (default: PIPETTE_SCORES_VERSION)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s", stream=sys.stderr)

    {"list": cmd_list, "prompts": cmd_prompts, "score": cmd_score, "packages": cmd_packages}[args.command](args)


if __name__ == "__main__":
    main()
