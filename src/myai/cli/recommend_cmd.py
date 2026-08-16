import sys
import typer
from ..core.paths import require_project_root
from ..core.config import ProjectConfig
from ..models.recommender import recommend_models, get_top_recommendation
from ..core.console import console, print_success, print_info

def recommend(
    apply: bool = typer.Option(False, "--apply", "-a", help="Automatically set top recommended model as project model"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive prompts")
):
    """Analyze hardware and data to recommend the optimal base model."""
    root = require_project_root()
    cfg = ProjectConfig.load(root)
    
    console.print("\n[bold cyan]MYAI RECOMMENDATION ENGINE[/bold cyan]")
    
    top_rec, hw, report = get_top_recommendation(root, cfg)
    
    console.print("\n[bold]SYSTEM & DATA ANALYSIS[/bold]")
    console.print(f"Examples         : {report.examples:,}")
    console.print(f"Estimated tokens : {report.tokens_approx:,}")
    vram_display = f"{hw.vram_gb} GB" if hw.vram_gb > 0 else f"{hw.ram_gb} GB RAM (CPU mode)"
    console.print(f"Hardware Compute : {hw.gpu} ({vram_display}) [Tier: {hw.tier}]")
    if cfg.model_id:
        console.print(f"Current Model    : [cyan]{cfg.model_id}[/cyan]")
    
    from ..models.registry import get_registry_models
    recs = recommend_models(hw, report, get_registry_models())
    
    console.print("\n[bold]RECOMMENDED MODELS[/bold]\n")
    
    for i, rec in enumerate(recs[:4], 1):
        m = rec.model
        status_icon = "[green]✓ Fits Hardware[/green]" if rec.fits_vram else "[red]✗ Requires More VRAM[/red]"
        is_current = " [bold cyan](Active)[/bold cyan]" if m.id == cfg.model_id else ""
        
        console.print(f"{i}. [bold]{m.name}[/bold] ({m.id}){is_current}")
        console.print(f"   Score        : [bold yellow]{rec.score}/100[/bold yellow]")
        console.print(f"   Method       : {rec.method}")
        console.print(f"   Min VRAM     : ~{m.vram_min} GB")
        console.print(f"   Compatibility: {status_icon}")
        if rec.reasons:
            console.print(f"   Rationale    : {', '.join(rec.reasons[:2])}")
        console.print("")

    if not top_rec:
        return

    should_apply = apply
    if not should_apply and not yes and sys.stdin.isatty():
        if cfg.model_id != top_rec.model.id:
            prompt_text = f"Apply top recommended model '{top_rec.model.id}' as project active model?"
            should_apply = typer.confirm(prompt_text, default=True)

    if should_apply:
        cfg.model_id = top_rec.model.id
        cfg.training.method = top_rec.method.lower()
        cfg.save(root)
        print_success(f"Project active model updated to: [bold cyan]{top_rec.model.id}[/bold cyan] ({top_rec.method})")
        print_info(f"To download base weights: `myai model add {top_rec.model.id}`")
        print_info("To train on this model:   `myai train`")