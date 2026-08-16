"""
Post-training export UI — professional Rich CLI screens and state machine
that guide the user from MODEL READY through export, validation, and
completion.

This module contains ONLY presentation logic. All business logic lives in
  export/packager.py  — build_package, build_zip_package
  export/validator.py — validate_package
"""

import time
import enum
from pathlib import Path

import typer
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.console import Group

from ..core.console import console
from ..export.packager import (
    build_zip_package,
    estimate_package_size,
    _dir_size_mb,
)
from ..export.validator import validate_package


# ═══════════════════════════════════════════════════════════════════
# State Machine
# ═══════════════════════════════════════════════════════════════════

class ExportState(enum.Enum):
    MODEL_READY = "MODEL_READY"
    EXPORT_CONFIRMATION = "EXPORT_CONFIRMATION"
    EXPORTING = "EXPORTING"
    VALIDATING = "VALIDATING"
    EXPORT_SUCCESS = "EXPORT_SUCCESS"
    EXPORT_FAILED = "EXPORT_FAILED"
    VALIDATION_FAILED = "VALIDATION_FAILED"


# ═══════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════

def _fmt_size(size_bytes: int) -> str:
    """Format byte count as human-readable size."""
    if size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.2f} GB"
    if size_bytes >= 1e6:
        return f"{size_bytes / 1e6:.1f} MB"
    if size_bytes >= 1e3:
        return f"{size_bytes / 1e3:.1f} KB"
    return f"{size_bytes} B"


def _fmt_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hours = mins // 60
    mins = mins % 60
    return f"{hours}h {mins:02d}m"


def _sanitize_filename(name: str) -> str:
    """Produce a safe ZIP filename from a model name."""
    safe = name.replace(" ", "-").replace("/", "-").replace("\\", "-")
    # Remove any characters that are not alphanumeric, dash, underscore, or dot
    safe = "".join(c for c in safe if c.isalnum() or c in "-_.")
    return safe or "model"


# ═══════════════════════════════════════════════════════════════════
# Rich Renderables (Screens)
# ═══════════════════════════════════════════════════════════════════

def _model_ready_panel(meta: dict, eval_report=None) -> Panel:
    """Render the MODEL READY ✓ screen."""
    t = Text()
    t.append("MODEL READY ✓\n\n", style="bold green")

    model_name = meta.get("id", "Unknown")
    t.append(f"  {model_name}\n", style="bold white")
    t.append("  Successfully trained and evaluated\n\n", style="dim")

    # Details table
    details = [
        ("Base Model", meta.get("base_model_name", meta.get("base_model", "—"))),
        ("Training Method", (meta.get("method") or meta.get("training_method", "—")).upper()),
        ("Dataset", meta.get("dataset", "—")),
        ("Training Steps", f"{meta.get('steps', 0):,}"),
        ("Evaluation", meta.get("evaluation", "—")),
    ]

    duration = meta.get("duration_seconds")
    if duration:
        details.append(("Duration", _fmt_duration(float(duration))))

    # Model size from adapter path
    adapter_path = Path(meta.get("adapter_path", ""))
    if adapter_path.exists():
        size = sum(f.stat().st_size for f in adapter_path.rglob("*") if f.is_file())
        details.append(("Model Size", _fmt_size(size)))

    details.append(("Run ID", meta.get("run_id", "—")))
    details.append(("Training", "✓ Complete"))

    eval_status = "✓ Passed"
    if eval_report:
        eval_status = f"✓ Passed ({int(eval_report.overall * 100)}%)" if eval_report.overall_pass else "✗ Failed"
    details.append(("Evaluation", eval_status))

    for label, value in details:
        t.append(f"  {label:<20s}", style="dim")
        t.append(f"{value}\n")

    return Panel(
        t,
        border_style="green",
        padding=(1, 2),
        width=52,
    )


def _export_config_panel(meta: dict, est_size: int) -> Panel:
    """Render the EXPORT TRAINED MODEL configuration screen."""
    t = Text()
    t.append("EXPORT TRAINED MODEL\n\n", style="bold cyan")

    model_name = meta.get("id", "Unknown")
    t.append(f"  Model:  {model_name}\n")
    t.append(f"  Format: ZIP\n")
    t.append(f"  Size:   ~{_fmt_size(est_size)}\n\n")

    t.append("  Package includes:\n", style="bold")
    included = [
        "Model adapter",
        "Tokenizer",
        "Metadata",
        "Evaluation report",
        "README",
        "Standalone loader",
        "Standalone Chat UI",
    ]
    for item in included:
        t.append(f"    ✓ {item}\n", style="green")

    t.append("\n")
    t.append("  Package does NOT include:\n", style="bold")
    excluded = [
        "MYAI source code",
        "Training engine",
        "Original dataset",
        "API keys / secrets",
        ".git directory",
        "Virtual environment",
    ]
    for item in excluded:
        t.append(f"    ✗ {item}\n", style="dim red")

    return Panel(
        t,
        border_style="cyan",
        padding=(1, 2),
        width=52,
    )


class ExportProgressDisplay:
    """Live-updating Rich renderable for export progress."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.stage = "Preparing..."
        self.percent = 0
        self._complete = False

    def update(self, stage: str, percent: int):
        self.stage = stage
        self.percent = min(percent, 100)
        if percent >= 100:
            self._complete = True

    def __rich__(self):
        t = Text()
        t.append("EXPORTING MODEL\n\n", style="bold cyan")
        t.append(f"  Creating standalone model package...\n\n")

        # Progress bar
        bar_width = 24
        filled = int(self.percent / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        t.append(f"  {bar}  {self.percent}%\n\n")

        t.append(f"  {self.stage}\n")
        if not self._complete:
            t.append("  Please do not close the application.\n", style="dim")

        panel = Panel(
            t,
            border_style="cyan",
            padding=(1, 2),
            width=52,
        )
        return Group(panel)


def _export_complete_panel(
    meta: dict,
    zip_path: Path,
    zip_size: int,
    validation_passed: bool,
) -> Panel:
    """Render the EXPORT COMPLETE ✓ screen."""
    t = Text()
    t.append("EXPORT COMPLETE ✓\n\n", style="bold green")

    model_name = meta.get("id", "Unknown")
    t.append(f"  Model:       {model_name}\n")
    t.append(f"  File:        {zip_path.name}\n")
    t.append(f"  Size:        {_fmt_size(zip_size)}\n")
    t.append(f"  Location:    {zip_path.parent}\n")
    t.append(f"  Run ID:      {meta.get('run_id', '—')}\n\n")

    if validation_passed:
        t.append("  Package validation: ✓ PASSED\n", style="bold green")
    else:
        t.append("  Package validation: ⚠ WARNINGS\n", style="bold yellow")

    t.append(f"\n  {zip_path}\n", style="dim")

    return Panel(
        t,
        border_style="green",
        padding=(1, 2),
        width=52,
    )


def _export_failed_panel(reason: str) -> Panel:
    """Render the EXPORT FAILED ✕ screen."""
    t = Text()
    t.append("EXPORT FAILED ✕\n\n", style="bold red")

    t.append("  The trained model was not exported.\n\n")
    t.append("  Reason:\n", style="bold")
    t.append(f"  {reason}\n\n")
    t.append("  Your trained model remains safely stored\n", style="green")
    t.append("  in the MYAI model registry.\n", style="green")

    return Panel(
        t,
        border_style="red",
        padding=(1, 2),
        width=52,
    )


def _validation_failed_panel(validation_result) -> Panel:
    """Render the VALIDATION FAILED screen."""
    t = Text()
    t.append("VALIDATION FAILED ✕\n\n", style="bold red")

    t.append("  The exported package did not pass validation.\n\n")
    t.append("  Failed checks:\n", style="bold")
    for check in validation_result.failed_checks:
        t.append(f"    ✗ {check.name}\n", style="red")
        if check.detail:
            t.append(f"      {check.detail}\n", style="dim")

    if validation_result.warnings:
        t.append("\n  Warnings:\n", style="bold yellow")
        for w in validation_result.warnings:
            t.append(f"    ⚠ {w}\n", style="yellow")

    t.append("\n  Your trained model remains safely stored\n", style="green")
    t.append("  in the MYAI model registry.\n", style="green")

    return Panel(
        t,
        border_style="red",
        padding=(1, 2),
        width=52,
    )


# ═══════════════════════════════════════════════════════════════════
# State Machine Orchestrator
# ═══════════════════════════════════════════════════════════════════

def run_post_training_ui(
    home,
    meta: dict,
    eval_report=None,
    run=None,
    spec=None,
    cfg=None,
    yes: bool = False,
):
    """
    Drive the post-training export UI flow.

    State transitions:
        MODEL_READY → EXPORT_CONFIRMATION → EXPORTING → VALIDATING →
        EXPORT_SUCCESS  (or EXPORT_FAILED / VALIDATION_FAILED)

    Args:
        home: MYAI home directory path.
        meta: Model metadata dict with keys: id, base_model, method,
              dataset, run_id, evaluation, adapter_path, steps,
              duration_seconds, base_model_name (optional).
        eval_report: EvaluationReport instance (optional).
        run: Run instance (optional).
        spec: Model spec from registry (optional).
        cfg: ProjectConfig instance (optional).
        yes: Auto-accept all prompts (non-interactive mode).
    """
    home = Path(home)
    state = ExportState.MODEL_READY

    while True:
        # ── MODEL_READY ──────────────────────────────────────────
        if state == ExportState.MODEL_READY:
            console.print()
            console.print(_model_ready_panel(meta, eval_report))
            console.print()

            if yes:
                state = ExportState.EXPORT_CONFIRMATION
                continue

            console.print("  [bold][1][/bold] Export Trained Model")
            console.print("  [bold][2][/bold] Exit\n")

            choice = typer.prompt("Select", default="1").strip()
            if choice == "2":
                console.print("\n[dim]Exiting. Your model is stored in the MYAI registry.[/dim]")
                console.print("[dim]Run [cyan]myai export[/cyan] anytime to export it.[/dim]\n")
                return
            state = ExportState.EXPORT_CONFIRMATION

        # ── EXPORT_CONFIRMATION ──────────────────────────────────
        elif state == ExportState.EXPORT_CONFIRMATION:
            est_size = estimate_package_size(home, meta)
            console.print()
            console.print(_export_config_panel(meta, est_size))
            console.print()

            if yes:
                # Auto mode: export to project root or cwd
                root = Path(cfg.data_path).parent if cfg else Path.cwd()
                export_dir = root / "exports" if root != Path.cwd() else Path.cwd() / "exports"
                export_dir = Path.cwd() / "exports"
                export_dir.mkdir(parents=True, exist_ok=True)
                zip_name = f"{_sanitize_filename(meta.get('id', 'model'))}.zip"
                zip_path = export_dir / zip_name
            else:
                console.print("  [bold][1][/bold] Export as ZIP")
                console.print("  [bold][2][/bold] Cancel\n")

                choice = typer.prompt("Select", default="1").strip()
                if choice == "2":
                    state = ExportState.MODEL_READY
                    continue

                # Prompt for location
                from ..data.prompt import prompt_output_path
                dest = prompt_output_path("Export location (folder):")
                zip_name = f"{_sanitize_filename(meta.get('id', 'model'))}.zip"
                zip_path = dest / zip_name

            state = ExportState.EXPORTING

        # ── EXPORTING ────────────────────────────────────────────
        elif state == ExportState.EXPORTING:
            display = ExportProgressDisplay(meta.get("id", "Model"))

            try:
                with Live(display, console=console, refresh_per_second=8):
                    def progress_cb(stage: str, pct: int):
                        display.update(stage, pct)
                        # Small delay so progress is visible
                        if pct < 100:
                            time.sleep(0.05)

                    build_zip_package(
                        home=home,
                        meta=meta,
                        zip_path=zip_path,
                        progress_callback=progress_cb,
                    )

                state = ExportState.VALIDATING

            except Exception as exc:
                console.print()
                console.print(_export_failed_panel(str(exc)))
                console.print()

                if yes:
                    return

                console.print("  [bold][1][/bold] Try Again")
                console.print("  [bold][2][/bold] Choose Another Location")
                console.print("  [bold][3][/bold] Back to Model\n")

                choice = typer.prompt("Select", default="1").strip()
                if choice == "1":
                    state = ExportState.EXPORTING
                elif choice == "2":
                    state = ExportState.EXPORT_CONFIRMATION
                else:
                    state = ExportState.MODEL_READY
                continue

        # ── VALIDATING ───────────────────────────────────────────
        elif state == ExportState.VALIDATING:
            console.print("\n  [cyan]●[/cyan] Validating package...")

            validation = validate_package(zip_path)

            if validation.passed:
                state = ExportState.EXPORT_SUCCESS
            else:
                console.print()
                console.print(_validation_failed_panel(validation))
                console.print()

                # Clean up the invalid ZIP
                try:
                    zip_path.unlink(missing_ok=True)
                except Exception:
                    pass

                if yes:
                    return

                console.print("  [bold][1][/bold] Try Again")
                console.print("  [bold][2][/bold] Back to Model\n")

                choice = typer.prompt("Select", default="1").strip()
                if choice == "1":
                    state = ExportState.EXPORTING
                else:
                    state = ExportState.MODEL_READY
                continue

        # ── EXPORT_SUCCESS ───────────────────────────────────────
        elif state == ExportState.EXPORT_SUCCESS:
            zip_size = zip_path.stat().st_size if zip_path.exists() else 0

            console.print()
            console.print(_export_complete_panel(meta, zip_path, zip_size, True))
            console.print()

            if not yes:
                console.print(f"  [dim]Path: {zip_path}[/dim]\n")
                console.print("  [bold][1][/bold] Done")
                console.print("  [bold][2][/bold] Export Another Copy\n")

                choice = typer.prompt("Select", default="1").strip()
                if choice == "2":
                    state = ExportState.EXPORT_CONFIRMATION
                    continue

            return

        # ── EXPORT_FAILED / VALIDATION_FAILED ────────────────────
        # (handled inline above in EXPORTING and VALIDATING blocks)
        else:
            return


# ═══════════════════════════════════════════════════════════════════
# Standalone export UI (used by export_cmd.py)
# ═══════════════════════════════════════════════════════════════════

def run_export_ui(home, meta: dict, yes: bool = False):
    """
    Simplified export UI for standalone `myai export` command.
    Skips the MODEL_READY screen and goes straight to export config.

    Args:
        home: MYAI home directory.
        meta: Model metadata dict.
        yes: Auto-accept prompts.
    """
    home = Path(home)
    est_size = estimate_package_size(home, meta)

    # Show export config
    console.print()
    console.print(_export_config_panel(meta, est_size))
    console.print()

    if not yes:
        console.print("  [bold][1][/bold] Export as ZIP")
        console.print("  [bold][2][/bold] Cancel\n")

        choice = typer.prompt("Select", default="1").strip()
        if choice == "2":
            console.print("[dim]Export cancelled.[/dim]")
            return

    # Get destination
    if yes:
        dest = Path.cwd() / "exports"
        dest.mkdir(parents=True, exist_ok=True)
    else:
        from ..data.prompt import prompt_output_path
        dest = prompt_output_path("Export location (folder):")

    zip_name = f"{_sanitize_filename(meta.get('id', 'model'))}.zip"
    zip_path = dest / zip_name

    # Export with progress
    display = ExportProgressDisplay(meta.get("id", "Model"))

    try:
        with Live(display, console=console, refresh_per_second=8):
            def progress_cb(stage: str, pct: int):
                display.update(stage, pct)
                if pct < 100:
                    time.sleep(0.05)

            build_zip_package(
                home=home,
                meta=meta,
                zip_path=zip_path,
                progress_callback=progress_cb,
            )
    except Exception as exc:
        console.print()
        console.print(_export_failed_panel(str(exc)))
        return

    # Validate
    console.print("\n  [cyan]●[/cyan] Validating package...")
    validation = validate_package(zip_path)

    if not validation.passed:
        console.print()
        console.print(_validation_failed_panel(validation))
        try:
            zip_path.unlink(missing_ok=True)
        except Exception:
            pass
        return

    # Success
    zip_size = zip_path.stat().st_size if zip_path.exists() else 0
    console.print()
    console.print(_export_complete_panel(meta, zip_path, zip_size, validation.passed))
    console.print()
    console.print(f"  [dim]Path: {zip_path}[/dim]\n")
