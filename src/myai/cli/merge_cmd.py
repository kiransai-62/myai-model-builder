import typer
from pathlib import Path
from typing import Optional
from rich.console import Console

from ..core.paths import require_project_root
from ..export.merger import merge_adapter

console = Console()

def merge(
    adapter: Optional[Path] = typer.Option(None, "--adapter", "-a", help="Path to adapter directory"),
    base: Optional[Path] = typer.Option(None, "--base", "-b", help="Path to base model directory"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for merged model"),
):
    """Merges LoRA adapter into base model weights into a standalone checkpoint."""
    root = require_project_root()
    if not adapter:
        candidate_adapter = root / "models" / "trained" / root.name / "adapter"
        if candidate_adapter.exists():
            adapter = candidate_adapter
        else:
            console.print("[bold red]Error:[/bold red] Please specify `--adapter <path>` to merge.")
            raise typer.Exit(1)

    if not base:
        base = root / "models" / "base"
        if not base.exists():
            base = adapter

    if not output:
        output = root / "models" / "merged" / root.name

    console.print(f"\n[bold cyan]🔄 MERGING LORA ADAPTER INTO BASE MODEL[/bold cyan]")
    console.print(f"Adapter: [cyan]{adapter}[/cyan]")
    console.print(f"Base:    [cyan]{base}[/cyan]")
    console.print(f"Output:  [cyan]{output}[/cyan]\n")

    out_dir = merge_adapter(base, adapter, output)
    console.print(f"[bold green]✓ Standalone merged model created at:[/bold green] [cyan]{out_dir}[/cyan]\n")
