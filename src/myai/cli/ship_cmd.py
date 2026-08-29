import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

from ..core.paths import require_project_root
from ..evaluation.ship_gate import run_ship_gate

console = Console()

def ship(
    base: Optional[Path] = typer.Option(None, "--base", help="Base model directory"),
    adapter: Optional[Path] = typer.Option(None, "--adapter", help="Trained LoRA adapter directory"),
    task_eval: Optional[Path] = typer.Option(None, "--task-eval", help="Custom task evaluation JSONL"),
    emit_evidence: bool = typer.Option(True, "--emit-evidence/--no-evidence", help="Write cryptographic evidence report"),
):
    """Executes Leg-2 offline regression suites and issues a SHIP / DON'T SHIP verdict."""
    root = require_project_root()
    if not adapter:
        # Default to latest trained adapter in project
        candidate_adapter = root / "models" / "trained" / root.name / "adapter"
        if candidate_adapter.exists():
            adapter = candidate_adapter

    console.print("\n[bold cyan]🛡️  MYAI LEG-2 REGRESSION & SANITY GATE (myai ship)[/bold cyan]")
    console.print(f"Project: [bold]{root.name}[/bold]")
    if adapter:
        console.print(f"Adapter: [cyan]{adapter}[/cyan]\n")

    evidence_dir = root / "evaluations" if emit_evidence else None
    verdict = run_ship_gate(
        base_model_path=base,
        adapter_path=adapter,
        custom_eval_file=task_eval,
        output_evidence_dir=evidence_dir,
    )

    t = Table(title="Offline Regression Suites", box=None)
    t.add_column("Suite", style="bold")
    t.add_column("Passed", justify="right")
    t.add_column("Score", justify="right")
    t.add_column("Status", justify="center")

    for s in verdict.suites:
        status_styled = f"[green]{s.status}[/green]" if s.status == "PASS" else f"[red]{s.status}[/red]"
        score_styled = f"{s.score:.0%}"
        t.add_row(s.name.replace("_", " ").title(), f"{s.passed_tests}/{s.total_tests}", score_styled, status_styled)

    console.print(t)
    console.print(f"\n[dim]Adapter Hash: {verdict.adapter_hash} · Timestamp: {verdict.timestamp}[/dim]")

    if verdict.verdict == "SHIP":
        console.print("\n" + "═" * 50)
        console.print(f"[bold green]🚀 VERDICT: {verdict.verdict} (Overall: {verdict.overall_score:.0%})[/bold green]")
        console.print("All offline sanity and safety regression gates satisfied.")
        console.print("═" * 50 + "\n")
        if verdict.evidence_path:
            console.print(f"[dim]Evidence receipt: {verdict.evidence_path}[/dim]\n")
        raise typer.Exit(0)
    else:
        console.print("\n" + "═" * 50)
        console.print(f"[bold red]⛔ VERDICT: {verdict.verdict} (Overall: {verdict.overall_score:.0%})[/bold red]")
        console.print("Model regressed on core reasoning or safety checks. Export blocked.")
        console.print("═" * 50 + "\n")
        raise typer.Exit(2)
