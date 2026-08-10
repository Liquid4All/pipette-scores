"""Render calibration-run status for one eval against the frozen MODELS registry."""

from pathlib import Path

from pipette_scores.types import EvalId
from rich.console import Console
from rich.table import Table

from pipette_calibration.slurm.models import MODELS
from pipette_calibration.verify_completions import ModelVerification, verify_eval


def _disk_name(hf_id: str) -> str:
    # Mirrors pipette_calibration.inference.generate: model.replace("/", "--").
    return hf_id.replace("/", "--")


def _cell(v: ModelVerification) -> tuple[str, str]:
    if v.have == 0 and v.missing > 0:
        return "red", "MISSING"
    parts: list[str] = []
    style = "green"
    if v.missing > 0:
        parts.append(f"{v.have}/{v.total} ({v.missing} missing)")
        style = "yellow"
    if v.duplicates > 0:
        parts.append(f"{v.duplicates} dups")
        style = "yellow"
    if v.empty == v.total and v.total > 0:
        parts.append(f"all {v.total} empty")
        style = "red"
    elif v.empty > 0:
        pct = v.empty / v.total * 100 if v.total else 0
        parts.append(f"{v.empty}/{v.total} empty ({pct:.0f}%)")
        style = "yellow"
    return (style, "\n".join(parts) if parts else "ok")


def status(*, eval_id: EvalId, dataset: str, results_path: Path) -> bool:
    """Print a status table for one eval. Returns True when every model is clean."""
    console = Console()
    if not results_path.is_dir():
        console.print(f"[red]results directory not found: {results_path}[/red]")
        return False

    by_disk = {v.model: v for v in verify_eval(eval_id, dataset, results_path)}

    table = Table(title=f"{eval_id} / {dataset}", show_lines=False)
    table.add_column("Model", style="bold", no_wrap=True)
    table.add_column("Status", justify="left")

    missing: list[str] = []  # no dir, or dir with zero completions
    incomplete: list[str] = []  # has some data but missing/dup/empty rows
    for spec in MODELS:
        v = by_disk.get(_disk_name(spec.hf_id))
        if v is None:
            table.add_row(spec.hf_id, "[red]MISSING[/red]")
            missing.append(spec.hf_id)
            continue
        style, text = _cell(v)
        table.add_row(spec.hf_id, f"[{style}]{text}[/{style}]")
        if v.status != "ok":
            if v.have == 0:
                missing.append(spec.hf_id)
            else:
                incomplete.append(spec.hf_id)

    console.print(table)
    n = len(MODELS)
    issues = missing + incomplete
    console.print(
        f"[bold]{n - len(issues)}/{n} ok[/bold], "
        f"[red]{len(issues)} issues[/red] "
        f"([red]{len(missing)} missing[/red], [yellow]{len(incomplete)} incomplete[/yellow])"
    )

    # Use plain print for resubmit commands — Rich hard-wraps long lines by
    # default and these must stay on one line for copy-paste.
    def _emit(header: str, hf_ids: list[str]) -> None:
        if not hf_ids:
            return
        console.print(f"[bold]{header}:[/bold]")
        for hf_id in hf_ids:
            print(
                f"  uv run --package pipette-calibration --group gpu"
                f" pipette-calibration slurm-completions submit"
                f" --eval {eval_id} --dataset {dataset} --model {hf_id}"
            )

    _emit("Resubmit missing", missing)
    _emit("Resubmit incomplete", incomplete)
    return not issues
