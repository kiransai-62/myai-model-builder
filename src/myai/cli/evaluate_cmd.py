import json
import typer
from pathlib import Path
from ..core.console import console, print_error, print_success
from ..core.home import ensure_home
from ..core.paths import require_project_root
from ..core.config import ProjectConfig
from ..data.manager import resolve_dataset_source
from ..models.trained_registry import list_trained, register_trained
from ..training.runs import RunManager
from ..evaluation.runner import run_evaluation

def evaluate(args: list[str] = typer.Argument(None)):
    """Run evaluation: myai evaluate | myai evaluate <model-id> | myai evaluate list | myai evaluate info <eval-id>"""
    home = ensure_home()

    if args and args[0] == "list":
        _list_evals(home)
        return
    if args and args[0] == "info":
        _info_eval(home, args[1] if len(args) > 1 else None)
        return

    root = require_project_root()
    cfg = ProjectConfig.load(root)
    model_id = args[0] if args else cfg.name
    cfg.name = model_id

    # Resolve run + adapter: registered model first, else latest SUCCESS run
    trained = {m.get("id"): m for m in list_trained(home)}
    if model_id in trained:
        run = RunManager(home).get(trained[model_id].get("run_id", ""))
        adapter = Path(trained[model_id].get("adapter_path", ""))
        already_registered = True
    else:
        runs = [
            r for r in RunManager(home).list()
            if r["config"].get("project") == cfg.name and r["result"].get("status") == "SUCCESS"
        ]
        if not runs:
            runs = [r for r in RunManager(home).list() if r["result"].get("status") == "SUCCESS"]
        if not runs:
            print_error("No successful training run found. Run `myai train` first.")
            raise typer.Exit(1)
        run = RunManager(home).get(runs[0]["run_id"])
        adapter = run.root / "adapter"
        already_registered = False

    source = resolve_dataset_source(root, cfg)
    report = run_evaluation(home, root, cfg, run, adapter, source)

    if report.status == "PASS" and not already_registered:
        dst = register_trained(
            home,
            cfg.name,
            cfg.model_id,
            cfg.dataset_id,
            run.run_id,
            adapter,
            report,
            cfg.training.method,
            root=root,
            eval_score=report.overall,
        )
        console.print(f"\n[green]✓ Registered:[/green] {dst}")
        console.print(f"[bold]myai model list[/bold] now shows [cyan]{cfg.name}[/cyan] as READY.")

def _list_evals(home):
    table_rows = []
    for run_dir in sorted((home / "runs").glob("run_*")):
        for eval_dir in sorted((run_dir / "evaluation").glob("eval_*")):
            if (eval_dir / "report.json").exists():
                try:
                    rep = json.loads((eval_dir / "report.json").read_text(encoding="utf-8"))
                    table_rows.append((
                        rep.get("eval_id", eval_dir.name),
                        rep.get("model_id", "?"),
                        f"{int(rep.get('overall', 0) * 100)}%",
                        rep.get("status", "FAILED"),
                    ))
                except Exception:
                    pass
    console.print("\n[bold cyan]EVALUATIONS[/bold cyan]")
    if not table_rows:
        console.print("[dim]No evaluations found.[/dim]")
        return
    for row in table_rows:
        icon = "[green]PASS[/green]" if row[3] == "PASS" else "[red]FAILED[/red]"
        console.print(f"{row[0]}   {row[1]}   {row[2]}   {icon}")

def _info_eval(home, eval_id):
    if not eval_id:
        print_error("Please specify an evaluation ID.")
        return
    for rep_file in (home / "runs").rglob(f"evaluation/{eval_id}/report.json"):
        console.print_json(data=json.loads(rep_file.read_text(encoding="utf-8")))
        return
    print_error(f"Evaluation not found: {eval_id}")