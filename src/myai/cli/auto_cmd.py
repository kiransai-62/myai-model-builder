import typer
from typing import List
from myai.autopilot.orchestrator import Autopilot, print_autopilot
from myai.core.paths import get_active_project_dir
from myai.training.engine import run_training
from myai.export.packager import export_package

def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def auto(
    export: bool = typer.Option(True, "--export/--no-export", help="Export standalone package on completion"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview plan without executing training"),
    model: str = typer.Option(None, "--model", help="Explicit model override (§18)"),
    override: List[str] = typer.Option([], "--override", help="key=value, e.g. lora_rank=32"),
    opt_iters: int = typer.Option(2, "--opt-iters", help="Bounded optimizer loop iterations"),
):
    """Goal-to-deployment autonomous build (Report Phase 15)."""
    project = get_active_project_dir()
    pilot = Autopilot(
        project,
        train_fn=lambda s: run_training(project, s),
        export_fn=lambda run_id: export_package(project, run_id),
        export=export,
        dry_run=dry_run,
        model_override=model,
        strategy_override={k: _coerce(v) for k, v in (o.split("=") for o in override)} or None,
        max_opt_iters=opt_iters,
    )
    print_autopilot(pilot.run(), project.name)
