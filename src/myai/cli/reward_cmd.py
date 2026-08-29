import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

from ..evaluation.reward_synth import synthesize_reward_script

reward_app = typer.Typer(help="Deterministic reward and verifier synthesis", no_args_is_help=True)
console = Console()

@reward_app.command("synth")
def synth_reward(
    references: Path = typer.Argument(..., help="Path to JSONL/JSON containing reference responses"),
    output: Path = typer.Option(Path("reward.py"), "-o", "--output", help="Output Python reward file path"),
    output_report: Optional[Path] = typer.Option(None, "--output-report", help="Output JSON calibration report"),
):
    """Infers deterministic verifiers and synthesizes a calibrated reward.py script."""
    if not references.exists():
        console.print(f"[bold red]Error:[/bold red] Reference file '{references}' does not exist.")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]🧠 SYNTHESIZING DETERMINISTIC VERIFIER[/bold cyan]")
    console.print(f"Analyzing references from: [cyan]{references}[/cyan]\n")

    try:
        out_file, report = synthesize_reward_script(references, output, output_report)
    except Exception as e:
        console.print(f"[bold red]Synthesis failed:[/bold red] {e}")
        raise typer.Exit(1)

    # Display Calibration Table
    t = Table(title="Calibration Report", box=None)
    t.add_column("Property", style="dim")
    t.add_column("Value", style="bold")
    t.add_row("Verifier Family", f"[cyan]{report.family.upper()}[/cyan]")
    t.add_row("Samples Analyzed", str(report.samples_analyzed))
    t.add_row("Reference Pass Rate", f"[green]{report.reference_pass_rate:.1%}[/green]")
    t.add_row("Negative Control Rejection", f"[green]{report.negative_rejection_rate:.1%}[/green]")
    t.add_row("Calibration Status", f"[bold green]{report.verdict}[/bold green]" if report.verdict == "CALIBRATED" else f"[bold red]{report.verdict}[/bold red]")
    console.print(t)

    console.print(f"\n[bold green]✓ Verifier Synthesized:[/bold green] [cyan]{out_file}[/cyan]")
    if output_report:
        console.print(f"[dim]Calibration report written to: {output_report}[/dim]\n")
