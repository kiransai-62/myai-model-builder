import json
import shutil
import time
import typer
from pathlib import Path

from ..core.console import console, print_success, print_error, print_warning
from ..core.home import ensure_home
from ..core.paths import require_project_root
from ..core.config import ProjectConfig
from ..data.prompt import prompt_data_path, prompt_output_path
from ..data.scanner import scan_directory, is_readable, has_supported_format
from ..data.validator import validate_data
from ..data.manager import DatasetManager, resolve_dataset_source
from ..hardware.detector import detect_hardware
from ..models.registry import get_registry_models
from ..system.storage import estimate_storage, print_budget

def _est_minutes(hw, billions, epochs, examples) -> float:
    tput = {"T0": 60, "T1": 900, "T2": 2400, "T3": 6000}.get(hw.tier, 900)
    tput *= (3.0 / max(0.5, billions)) ** 0.5
    tokens = max(1, examples) * 120 * epochs
    return round(tokens / max(1, tput) / 60, 1)

def train(
    data: str = typer.Option(None, "--data", "-d", help="Skip the Step-1 prompt"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Automatically accept confirmation prompts"),
    model: str = typer.Option(None, "--model", "-m", help="Explicitly select base model"),
    auto: bool = typer.Option(False, "--auto", "-a", help="Launch autonomous Goal-to-Deployment build (Phase 15)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview autonomous plan without training"),
    opt_iters: int = typer.Option(2, "--opt-iters", help="Maximum optimizer rounds in auto mode"),
    stream_layers: bool = typer.Option(False, "--stream-layers", help="Enable Exact Layer Streaming for 4GB VRAM training"),
    task: str = typer.Option("sft", "--task", help="Training task mode (sft, dpo, orpo, simpo, kto)"),
):
    root = require_project_root()
    cfg = ProjectConfig.load(root)
    if stream_layers:
        cfg.training.stream_layers = True
    if task:
        cfg.training.task = task
    home = ensure_home()
    selection_mode = "user_selected"

    # ── AUTONOMOUS AUTOPILOT DISPATCH ───────────────────────────
    if auto:
        from ..autopilot.orchestrator import Autopilot, print_autopilot
        from ..training.engine import run_training
        from ..export.packager import export_package

        pilot = Autopilot(
            root,
            train_fn=lambda s: run_training(root, s),
            export_fn=lambda run_id: export_package(root, run_id),
            export=True,
            dry_run=dry_run,
            model_override=model,
            max_opt_iters=opt_iters,
        )
        report = pilot.run()
        print_autopilot(report, cfg.name or root.name)
        return

    console.print("\n[bold cyan]MYAI TRAINER[/bold cyan]")
    console.print("─" * 28 + "\n")

    # ── STEP 1/5 — TRAINING DATA ────────────────────────────────
    console.print("[bold]Step 1/5 — Training Data[/bold]\n")
    console.print("Enter your training data path.\nYou can provide a file or folder.\n")

    reg_src = resolve_dataset_source(root, cfg)
    has_reg = reg_src and reg_src.exists()

    if data:
        src = Path(data.strip('"')).expanduser().resolve()
        if not src.exists():
            src = prompt_data_path(default=reg_src if has_reg else None)
    elif yes:
        # Non-interactive / test fallback
        src = reg_src
        if not src.exists():
            src = prompt_data_path()
    else:
        src = prompt_data_path(default=reg_src if has_reg else None)

    # ── STEP 2/5 — DATA ANALYSIS ────────────────────────────────
    console.print("\n[bold]Step 2/5 — Data Analysis[/bold]\n")
    console.print(f"Source: {src}\n", highlight=False)

    scan = scan_directory(src)
    ok_fmt, fmt_label = has_supported_format(scan)
    report = validate_data(src)

    console.print(f"[green]✓[/green] Path exists")
    console.print(f"[green]✓[/green] Readable" if is_readable(src) else "[red]✗ Not readable[/red]")
    console.print(f"[green]✓[/green] Supported format ({fmt_label})" if ok_fmt
                  else f"[red]✗ Unsupported format ({fmt_label})[/red]", highlight=False)
    console.print(f"[green]✓[/green] {report.examples} examples detected\n")

    if not (ok_fmt and report.examples > 0):
        print_error("Dataset is not usable for training.")
        raise typer.Exit(1)
    if not yes and not typer.confirm("Continue?", default=True):
        raise typer.Exit(0)

    # Register (quiet) so provenance holds; reuse if same source
    manager = DatasetManager(home)
    meta = manager.get(cfg.dataset_id) if cfg.dataset_id else None
    if not meta or Path(meta.get("source", "")).resolve() != src:
        from ..cli.data_cmd import register_dataset
        meta = register_dataset(src, quiet=True)
        cfg.dataset_id = meta["dataset_id"]
        cfg.save(root)

    # ── STEP 3/5 — HARDWARE ANALYSIS ────────────────────────────
    console.print("\n[bold]Step 3/5 — Hardware Analysis[/bold]\n")
    hw = detect_hardware()
    console.print(f"CPU {hw.cpu}   RAM {hw.ram_gb} GB", highlight=False)
    console.print(f"GPU {hw.gpu}   VRAM {hw.vram_gb if hw.vram_gb else 'N/A'} GB", highlight=False)
    console.print(f"Compute Tier {hw.tier}   Free disk {hw.disk_gb} GB\n", highlight=False)

    # ── STEP 4/5 — MODEL RECOMMENDATION ─────────────────────────
    console.print("[bold]Step 4/5 — Model Recommendation[/bold]\n")
    models = get_registry_models()
    
    if model:
        spec = next((m for m in models if m.id == model), None)
        if not spec:
            print_error(f"Model '{model}' not found in registry.")
            raise typer.Exit(1)
        selection_mode = "user_selected"
        console.print(f"User Selected: [bold]{spec.name}[/bold] ({spec.id})\n", highlight=False)
    elif cfg.model_id and any(m.id == cfg.model_id for m in models):
        spec = next(m for m in models if m.id == cfg.model_id)
        selection_mode = "user_selected"
        console.print(f"Active Model: [bold]{spec.name}[/bold] ({spec.id})\n", highlight=False)
        from ..models.recommender import recommend_models
        recs = recommend_models(hw, report, models, goal=cfg.goal)
        rec = recs[0].model if recs else models[0]
        selection_mode = "recommended"
        console.print("\n[bold cyan]MODEL AUTO-RECOMMENDATION[/bold cyan]")
        console.print(f"Recommended model: {rec.name} ({rec.id})", highlight=False)
        console.print(f"Recommended: [bold]{rec.name}[/bold] ({rec.parameters})\n", highlight=False)

        if not yes and not typer.confirm(f"Use {rec.name}?", default=True):
            for i, m in enumerate(models, 1):
                mark = "✓" if (not hw.vram_gb or hw.vram_gb >= m.vram_min) else "✗"
                console.print(f"  {i}. {mark} {m.name} ({m.parameters}, min {m.vram_min} GB)", highlight=False)
            try:
                rec = models[int(typer.prompt("Select number", default="1")) - 1]
            except (ValueError, IndexError):
                print_error("Invalid selection.")
                raise typer.Exit(1)
        spec = rec

    cfg.model_id = spec.id

    # ── STEP 5/5 — TRAINING CONFIGURATION ───────────────────────
    console.print("\n[bold]Step 5/5 — Training Configuration[/bold]\n")
    cfg.training.method = "lora" if (hw.vram_gb or 0) >= spec.vram_min + 4 or (hw.vram_gb == 0 and hw.tier == "T1") else "qlora"
    console.print(f"Method          {cfg.training.method.upper()}")
    console.print(f"Epochs          {cfg.training.epochs}")
    console.print(f"Batch size      {cfg.training.batch_size} (accum {cfg.training.grad_accum})")
    console.print(f"Learning rate   {cfg.training.learning_rate}\n")

    if not yes and typer.confirm("Edit configuration?", default=False):
        cfg.training.method = typer.prompt("Method (qlora/lora)", default=cfg.training.method)
        cfg.training.epochs = int(typer.prompt("Epochs", default=str(cfg.training.epochs)))
        cfg.training.batch_size = int(typer.prompt("Batch size", default=str(cfg.training.batch_size)))
        cfg.training.learning_rate = float(typer.prompt("Learning rate", default=str(cfg.training.learning_rate)))
    cfg.save(root)

    src_files = [src] if src.is_file() else list(src.rglob("*"))
    total_bytes = meta.get("total_bytes", 0) if meta else sum(f.stat().st_size for f in src_files if f.is_file())
    budget = estimate_storage(total_bytes, spec.parameters_billions or 3.0,
                              cfg.training.method, cfg.training.epochs)
    mins = _est_minutes(hw, spec.parameters_billions or 3.0, cfg.training.epochs, report.examples)

    console.print("\n[bold]TRAINING APPROVAL[/bold]\n")
    console.print(f"Data:   {src}", highlight=False)
    console.print(f"Model:  {spec.name}", highlight=False)
    console.print(f"Method: {cfg.training.method.upper()}")
    console.print(f"Estimated time: ~{mins:.0f} min  [dim](estimate, not a guarantee)[/dim]\n")

    console.print("\n[bold cyan]STORAGE BUDGET & DISK PROTECTION[/bold cyan]\n")
    if not print_budget(budget, hw.disk_gb, console):
        raise typer.Exit(1)
    if not yes and not typer.confirm("Start training?", default=False):
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit(0)

    # ── RUN (with resume detection) ─────────────────────────────
    from ..training.runs import RunManager
    runman = RunManager(home)
    resume_ckpt = None
    run = runman.find_resumable(cfg.name, cfg.model_id or spec.id, cfg.dataset_id)

    if run:
        console.print(f"\nMYAI detected previous run [cyan]{run.run_id}[/cyan]")
        last_ckpt = run.latest_checkpoint()
        console.print(f"Last checkpoint: {last_ckpt.name if last_ckpt else 'None'}\n")
        if not yes and typer.confirm("[Y] Resume   [N] Start over", default=True):
            resume_ckpt = last_ckpt
        else:
            run = None
    if run is None:
        run = runman.create({
            "project": cfg.name,
            "dataset_id": cfg.dataset_id,
            "base_model": spec.id,
            "selection_mode": selection_mode,
            "training_method": cfg.training.method,
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate
        })

    from ..training.engine import run_training_engine
    result = run_training_engine(run, {
        "cfg": cfg,
        "spec": spec,
        "source": src,
        "home": home,
        "root": root,
        "selection_mode": selection_mode,
        "stream_layers": cfg.training.stream_layers,
        "task": cfg.training.task,
        "budget_gb": budget.additional_gb,
        "resume_ckpt": resume_ckpt
    })

    # ── EVALUATION → REGISTRY ────────────────────────────────────
    from ..evaluation.runner import run_evaluation
    from ..models.trained_registry import register_trained, list_trained

    console.print("\n[bold cyan]RUNNING EVALUATION[/bold cyan]")
    eval_report = run_evaluation(home, root, cfg, run, run.root / "adapter", src)

    registered = any(m.get("id") == cfg.name for m in list_trained(home))
    if eval_report.status == "PASS" and not registered:
        register_trained(
            home,
            cfg.name,
            spec.id,
            cfg.dataset_id,
            run.run_id,
            run.root / "adapter",
            eval_report,
            cfg.training.method,
            selection_mode=selection_mode,
            eval_score=eval_report.overall,
            root=root
        )

    # ── POST-TRAIN: MODEL READY UI ─────────────────────────────────
    from ..cli.post_training_ui import run_post_training_ui

    run_post_training_ui(
        home=home,
        meta={
            "id": cfg.name,
            "base_model": spec.id,
            "base_model_name": getattr(spec, "name", spec.id),
            "method": cfg.training.method.upper(),
            "dataset": cfg.dataset_id,
            "run_id": run.run_id,
            "evaluation": f"{int(eval_report.overall * 100)}%",
            "adapter_path": str(run.root / "adapter"),
            "steps": result.get("steps", 0),
            "duration_seconds": result.get("duration_seconds", 0),
        },
        eval_report=eval_report,
        run=run,
        spec=spec,
        cfg=cfg,
        yes=yes,
    )