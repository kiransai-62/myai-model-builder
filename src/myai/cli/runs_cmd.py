from rich.table import Table
from ..core.home import ensure_home
from ..core.console import console
from ..training.runs import RunManager

def list_runs():
    """List all training runs and their execution status."""
    table = Table(title="Training Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Model")
    table.add_column("Dataset")
    table.add_column("Status")
    for r in RunManager(ensure_home()).list():
        status = r["result"].get("status", "RUNNING")
        status_styled = (
            f"[green]{status}[/green]"
            if status == "SUCCESS"
            else (f"[red]{status}[/red]" if status in ("FAILED", "INTERRUPTED") else status)
        )
        table.add_row(
            r["run_id"],
            r["config"].get("base_model", "?"),
            r["config"].get("dataset_id", "?"),
            status_styled,
        )
    console.print(table)

def info(run_id: str):
    """Display frozen config and result metadata for a specific run ID."""
    run = RunManager(ensure_home()).get(run_id)
    if run:
        console.print_json(data={"config": run.read_config(), "result": run.read_result()})
    else:
        console.print(f"[red]Run '{run_id}' not found.[/red]")

def best():
    """Show the experiment leaderboard ranked by goal-weighted composite score."""
    from ..core.paths import find_project_root
    from ..core.config import load_config
    from ..core.goal import GoalProfile
    from ..models.leaderboard import Leaderboard

    home = ensure_home()
    root = find_project_root()
    cfg = load_config(root) if root else load_config(home)

    # Load goal profile from project config
    goal_data = cfg.get("goal", {})
    if not goal_data:
        console.print("[yellow]No goal profile found. Run: myai init[/yellow]")
        return
    goal = GoalProfile.from_dict(goal_data)

    lb = Leaderboard(goal, runs_dir=home / "leaderboard")
    if not lb.runs:
        console.print("[yellow]No evaluated runs yet. Run: myai train && myai evaluate[/yellow]")
        return
    lb.print_board()
