#!/usr/bin/env python3
"""CLI client for the pipette-scores server.

Usage:
    # Dump prompt samples as JSON (pipe to file or jq)
    uv run scripts/eval_client.py prompts --eval ifstruct --dataset default
    uv run scripts/eval_client.py prompts --eval ifstruct --dataset default --limit 5

    # Score completions from a JSON file
    #   file format: [{"id": "abc123", "completion": "The answer is B"}, ...]
    uv run scripts/eval_client.py score --eval ifstruct --dataset default --file completions.json

    # Custom server URL
    uv run scripts/eval_client.py --url http://evals:9000 prompts --eval ifstruct --dataset default
"""

import argparse
import json
import sys

import httpx


def _dump(data):
    json.dump(data, sys.stdout, indent=2)
    print()


def cmd_prompts(client: httpx.Client, args: argparse.Namespace):
    resp = client.get(f"/evals/{args.eval}/datasets/{args.dataset}/samples")
    resp.raise_for_status()
    samples = resp.json()["samples"]
    if args.limit:
        samples = samples[: args.limit]
    _dump(samples)


def cmd_score(client: httpx.Client, args: argparse.Namespace):
    with open(args.file) as f:
        completions = json.load(f)

    resp = client.post(
        "/score",
        json={
            "eval_id": args.eval,
            "dataset_name": args.dataset,
            "completions": completions,
        },
    )
    resp.raise_for_status()
    _dump(resp.json())


def main():
    parser = argparse.ArgumentParser(description="CLI client for the pipette-scores server")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="Evals server URL (default: http://localhost:8000)"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_prompts = sub.add_parser("prompts", help="Dump prompt samples as JSON")
    p_prompts.add_argument("--eval", required=True, help="Eval ID")
    p_prompts.add_argument("--dataset", required=True, help="Dataset name")
    p_prompts.add_argument("--limit", type=int, default=None, help="Max number of samples to output")

    p_score = sub.add_parser("score", help="Score completions from a JSON file")
    p_score.add_argument("--eval", required=True, help="Eval ID")
    p_score.add_argument("--dataset", required=True, help="Dataset name")
    p_score.add_argument("--file", required=True, help="Path to JSON file with completions")

    args = parser.parse_args()

    client = httpx.Client(base_url=args.url, timeout=60.0)

    {"prompts": cmd_prompts, "score": cmd_score}[args.command](client, args)


if __name__ == "__main__":
    main()
