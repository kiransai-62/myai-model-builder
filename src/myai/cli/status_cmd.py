"""MYAI Project Status CLI Command (Report §4, §10).

Displays comprehensive project state awareness and recommended next actions.
"""
from pathlib import Path
import typer
from rich.panel import Panel
from rich.table import Table

from ..core.console import console
from ..core.paths import get_active_project_dir
from ..core.state import inspect_project_state, ProjectState


def status():
    """Display current project lifecycle state, artifacts, and next steps."""
    project_dir = get_active_project_dir()
    proj_status = inspect_project_state(project_dir)

    console.print(f"\n[bold cyan]📊 MYAI PROJECT STATUS — {proj_status.project_name}[/bold cyan]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="bold")
    table.add_column("Value")

    state_colors = {
        ProjectState.INITIALIZED: "yellow",
        ProjectState.DATA_READY: "cyan",
        ProjectState.TRAINED: "blue",
        ProjectState.EVALUATED: "magenta",
        ProjectState.READY_TO_EXPORT: "green",
        ProjectState.EXPORTED: "bold green",
    }
    color = state_colors.get(proj_status.state, "white")
    table.add_row("Lifecycle State", f"[{color}]{proj_status.state.value.upper()}[/{color}]")
    table.add_row("Project Root", str(proj_status.project_dir))
    table.add_row("Goal Intent", f"{proj_status.goal.task.value} ({proj_status.goal.domain.value})")

    # Data
    data_desc = f"[green]✓ Ready[/green] ({proj_status.data_samples:,} samples)" if proj_status.has_data else "[red]✗ Not registered[/red]"
    table.add_row("Training Data", data_desc)

    # Model
    model_desc = f"[green]✓ Trained[/green] (latest: {proj_status.latest_run_id})" if proj_status.has_trained_model else "[dim]Not trained yet[/dim]"
    table.add_row("Model Artifacts", model_desc)

    # Eval
    eval_desc = f"[green]✓ Evaluated[/green] (Score: {proj_status.best_score:.1f}/100)" if proj_status.has_evaluation and proj_status.best_score else (
        "[yellow]Evaluated (unranked)[/yellow]" if proj_status.has_evaluation else "[dim]Pending evaluation[/dim]"
    )
    table.add_row("Leaderboard Eval", eval_desc)

    # Export
    export_desc = f"[bold green]✓ Exported[/bold green] ({Path(proj_status.export_path).name})" if proj_status.has_export else "[dim]Not exported[/dim]"
    table.add_row("Standalone Export", export_desc)

    console.print(table)

    # Action box
    console.print(Panel(
        f"[bold]Next Step:[/bold] {proj_status.next_step}\n"
        f"[dim]Run command:[/dim] [bold cyan]{proj_status.actionable_command}[/bold cyan]",
        title="🎯 Recommendation",
        border_style="cyan",
    ))
