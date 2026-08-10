"""CLI for scoring LLM completions against edge evaluation benchmarks."""

import argparse
import json
import logging
import pathlib
import sys

from tabulate import tabulate

from pipette_scores import scoring, dataset_catalog
from pipette_scores.types import EvalId, SampleCompletion

logger = logging.getLogger(__name__)


def cmd_list(args: argparse.Namespace) -> None:
    rows = []
    for eval_id in scoring.list_eval_ids():
        datasets = dataset_catalog.list_dataset_names(eval_id, root_dir=args.data_dir)
        rows.append([eval_id.value, ", ".join(datasets) if datasets else "(none)"])
    print(tabulate(rows, headers=["EVAL", "DATASETS"], tablefmt="simple"))


def cmd_prompts(args: argparse.Namespace) -> None:
    eval_id = EvalId(args.eval)
    samples = dataset_catalog.load_prompt_samples(eval_id, args.dataset, root_dir=args.data_dir)

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
            completions.append(SampleCompletion(id=obj["id"], completion=obj["completion"]))
    return completions


def _load_payload_json(path: pathlib.Path) -> list[SampleCompletion]:
    """Load completions from a payload.json file."""
    with open(path) as f:
        data = json.load(f)
    return [SampleCompletion(id=c["id"], completion=c["completion"]) for c in data["completions"]]


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

    logger.info("Scoring %d completions for %s/%s", len(completions), eval_id, args.dataset)
    scores, context = scoring.score(eval_id, completions, samples)

    total = len(scores)
    correct = sum(1 for v in scores if v.is_correct)
    accuracy = correct / total if total > 0 else 0.0

    # Summary table
    print()
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
    p_prompts.add_argument("-o", "--output", default=None, help="Output file (default: stdout)")

    # score
    p_score = sub.add_parser(
        "score",
        help="Score completions from a file or directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
supported input formats:
  JSONL file       one {"id": "...", "completion": "..."} per line
  payload.json     {"completions": [...]}
  directory        looks for payload.json first, then *.jsonl shards

examples:
  %(prog)s --eval ifbench --dataset default -f completions.jsonl
  %(prog)s --eval ifstruct --dataset edge_2026.03.2 -f results/payload.json
  %(prog)s --eval ifbench --dataset default -f results/shards_dir/
""",
    )
    p_score.add_argument("--eval", required=True, help="Eval ID (e.g. ifbench, ifstruct)")
    p_score.add_argument("--dataset", required=True, help="Dataset name (e.g. default, edge_2026.03.2)")
    p_score.add_argument("-f", "--file", required=True, help="JSONL file, payload.json, or directory")
    p_score.add_argument("-o", "--output", default=None, help="Write full JSON results to file")
    p_score.add_argument("--no-samples", action="store_true", help="Hide per-sample results table")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s", stream=sys.stderr)

    {"list": cmd_list, "prompts": cmd_prompts, "score": cmd_score}[args.command](args)


if __name__ == "__main__":
    main()
