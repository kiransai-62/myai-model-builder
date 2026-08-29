"""
MYAI Recommendation CLI Command.

Renders rich terminal UI panels with System Compatibility breakdowns,
Multi-Dimension Fit Scores, dynamic hardware metrics, and explainability cards.
"""
import sys
import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from ..core.paths import require_project_root
from ..core.config import ProjectConfig
from ..models.recommender import recommend_models, get_top_recommendation
from ..core.console import console, print_success, print_info


def recommend(
    apply: bool = typer.Option(False, "--apply", "-a", help="Automatically set top recommended model as project model"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive prompts")
):
    """Analyze hardware and data to recommend the optimal base model with explainable fit scores."""
    root = require_project_root()
    cfg = ProjectConfig.load(root)
    
    console.print("\n[bold cyan]MYAI RECOMMENDATION ENGINE — HARDWARE INTELLIGENCE[/bold cyan]")
    
    top_rec, hw, report = get_top_recommendation(root, cfg)
    
    if not top_rec:
        console.print("[red]No compatible models found.[/red]")
        return

    m = top_rec.model
    dm = top_rec.dynamic_memory
    fb = top_rec.fit_breakdown
    sb = top_rec.hw_breakdown
    
    # ── 1. Top Recommendation Header Box ────────────────────────
    header_content = (
        f"[bold]Model:[/bold]         [cyan]{m.name}[/cyan] ({m.id})\n"
        f"[bold]Quantization:[/bold]  [green]GGUF_Q4_K_M[/green] / [dim]FP16 Base[/dim]\n"
        f"[bold]Verdict:[/bold]       {top_rec.verdict_badge}\n"
        f"[bold]Overall Score:[/bold] [bold yellow]{top_rec.score}/100[/bold yellow]\n"
        f"[bold]Confidence:[/bold]    [bold cyan]{top_rec.confidence:.2f}[/bold cyan]"
    )
    console.print(Panel(header_content, title="🏆 [bold cyan]MYAI MODEL RECOMMENDATION[/bold cyan]", expand=False, border_style="cyan"))

    # ── 2. System Compatibility Breakdown (8 Factors) ───────────
    console.print("\n[bold]SYSTEM COMPATIBILITY[/bold]")
    if sb:
        sys_table = Table(show_header=False, box=None, padding=(0, 2))
        sys_table.add_column("Factor", style="bold")
        sys_table.add_column("Score", style="yellow")
        sys_table.add_column("Status")

        factors = [
            ("VRAM", sb.vram_score),
            ("RAM", sb.ram_score),
            ("GPU Compute", sb.gpu_compute_score),
            ("CPU", sb.cpu_score),
            ("Storage", sb.storage_score),
            ("Throughput", sb.throughput_score),
            ("Context", sb.context_score),
            ("Runtime", sb.runtime_score),
        ]
        for name, sc in factors:
            mark = "[green]✅[/green]" if sc >= 75.0 else ("[yellow]⚠️[/yellow]" if sc >= 50.0 else "[red]❌[/red]")
            sys_table.add_row(f"  {name:<14}", f"{int(sc)}%", mark)
        console.print(sys_table)

    # ── 3. Multi-Dimension Fit Scores ───────────────────────────
    if fb:
        console.print("")
        console.print(f"[bold]DATA FIT[/bold]           {int(fb.dataset_fit)}%")
        console.print(f"[bold]TASK FIT[/bold]           {int(fb.task_fit)}%")
        console.print(f"[bold]TRAINING FIT[/bold]       {int(fb.training_fit)}%")
        console.print(f"[bold]DEPLOYMENT FIT[/bold]     {int(fb.deployment_fit)}%")

    # ── 4. Dynamic Hardware & Context Metrics ───────────────────
    console.print("\n" + "─" * 47)
    console.print(f"Estimated inference: [cyan]{top_rec.predicted_tokens_per_sec} tok/s[/cyan]")
    vram_display = f"{dm.total_peak_vram_gb} GB / {hw.vram_gb} GB" if (dm and hw.vram_gb > 0) else f"{m.inference.q4_vram_gb} GB (CPU mode)"
    console.print(f"Estimated VRAM:      [cyan]{vram_display}[/cyan]")
    console.print(f"Estimated training:  [cyan]{m.training.lora_vram_gb} GB VRAM[/cyan] ([dim]{top_rec.method}[/dim])")
    console.print(f"Estimated storage:   [cyan]{m.training.workspace_storage_gb} GB[/cyan]")
    console.print(f"Recommended context: [bold green]{top_rec.recommended_context:,} tokens[/bold green]")
    console.print("─" * 47)

    # ── 5. Explainability ("Why this model?") ───────────────────
    console.print("\n[bold]WHY THIS MODEL?[/bold]")
    for r in top_rec.why_this_model[:5]:
        console.print(f"  [green]✓[/green] {r}")

    # ── 6. Alternative Model Card ───────────────────────────────
    if top_rec.alternative_model:
        alt = top_rec.alternative_model
        console.print(f"\n[bold]ALTERNATIVE[/bold]")
        console.print(f"[bold cyan]{alt.model.name}[/bold cyan] ({alt.model.id})")
        console.print(f"  {alt.verdict_badge} — [yellow]Score {alt.score}/100[/yellow] · {alt.method} · ~{alt.predicted_tokens_per_sec} tok/s")
        if alt.reasons:
            console.print(f"  [dim]{alt.reasons[0]}[/dim]")

    console.print("")

    # ── 7. Interactive Apply Confirmation ───────────────────────
    should_apply = apply
    if not should_apply and not yes and sys.stdin.isatty():
        if cfg.model_id != top_rec.model.id:
            prompt_text = f"Apply recommended model '{top_rec.model.id}' as project active model?"
            should_apply = typer.confirm(prompt_text, default=True)

    if should_apply:
        cfg.model_id = top_rec.model.id
        cfg.training.method = top_rec.method.lower()
        cfg.save(root)
        print_success(f"Project active model updated to: [bold cyan]{top_rec.model.id}[/bold cyan] ({top_rec.method})")
        print_info(f"To download base weights: `myai model add {top_rec.model.id}`")
        print_info("To train on this model:   `myai train`")