import typer
from myai.core.goal import GoalProfile
from myai.core.paths import get_active_project_dir
from myai.hardware.detector import detect_hardware
from myai.models.leaderboard import Leaderboard
from myai.optimizer.engine import OptimizerEngine, print_report
from myai.training.engine import run_training

def optimize(
    max_iters: int = typer.Option(3, "--max-iters", help="Bounded loop (§18)"),
    min_delta: float = typer.Option(2.0, "--min-delta", help="Minimum justified Δ"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview mutations only"),
):
    project = get_active_project_dir()
    goal = GoalProfile.from_yaml(project / "myai.yaml")
    board = Leaderboard(goal, project / "experiments" / "runs")
    hw = detect_hardware()

    engine = OptimizerEngine(
        goal,
        board,
        train_fn=lambda strategy: run_training(project, strategy, hw),
        min_delta=min_delta,
        max_iters=max_iters,
    )
    print_report(engine.run(dry_run=dry_run))
