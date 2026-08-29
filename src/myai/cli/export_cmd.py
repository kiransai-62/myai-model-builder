import typer
from pathlib import Path
from typing import Optional
from ..core.console import console, print_error
from ..core.home import ensure_home
from ..models.trained_registry import list_trained
from ..export.gguf_exporter import export_to_gguf
from ..export.merger import merge_adapter

def export(
    model_id: str = typer.Argument(None, help="Trained model id (optional)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-accept prompts"),
    format: str = typer.Option("zip", "--format", "-f", help="Export format: 'zip' (standalone web app), 'gguf' (Ollama), 'merged' (weights)"),
    quant: str = typer.Option("q4_k_m", "--quant", "-q", help="GGUF quantization level (q4_k_m, q8_0, f16)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Custom output path"),
):
    """Export the trained model into standalone ZIP, GGUF (Ollama), or Merged weights."""
    home = ensure_home()
    trained = list_trained(home)

    if not trained:
        print_error("No trained models yet. Run: myai train")
        raise typer.Exit(1)

    if model_id:
        meta = next((m for m in trained if m["id"] == model_id), None)
        if not meta:
            print_error(f"Unknown trained model: {model_id}")
            raise typer.Exit(1)
    elif len(trained) == 1:
        meta = trained[0]
    else:
        console.print("\n[bold]Trained models:[/bold]")
        for i, m in enumerate(trained, 1):
            console.print(f"  {i}. {m['id']}  ({m['base_model']}, {m.get('evaluation', 'n/a')})")
        try:
            meta = trained[int(typer.prompt("Select number", default="1")) - 1]
        except (ValueError, IndexError):
            print_error("Invalid selection.")
            raise typer.Exit(1)

    adapter_path = Path(meta.get("adapter_path", ""))
    if not adapter_path.exists():
        adapter_path = home / "models" / "trained" / meta["id"] / "adapter"

    if format.lower() == "gguf":
        out_file = output or Path(f"{meta['id']}-{quant}.gguf")
        console.print(f"\n[bold cyan]📦 EXPORTING GGUF & OLLAMA MODELFILE[/bold cyan]")
        console.print(f"Model:        [cyan]{meta['id']}[/cyan]")
        console.print(f"Quantization: [cyan]{quant.upper()}[/cyan]")
        console.print(f"Output:       [cyan]{out_file}[/cyan]\n")
        export_to_gguf(adapter_path, out_file, quant=quant)
        console.print(f"[bold green]✓ GGUF Export Complete:[/bold green] [cyan]{out_file}[/cyan]")
        console.print(f"To run in Ollama: [cyan]ollama create {meta['id']} -f Modelfile.{out_file.stem}[/cyan]\n")
        return

    elif format.lower() == "merged":
        out_dir = output or Path(f"{meta['id']}-merged")
        base_dir = home / "models" / "base" / meta.get("base_model", "")
        console.print(f"\n[bold cyan]🔄 MERGING LORA INTO STANDALONE WEIGHTS[/bold cyan]")
        merge_adapter(base_dir, adapter_path, out_dir)
        console.print(f"[bold green]✓ Merged Model Complete:[/bold green] [cyan]{out_dir}[/cyan]\n")
        return

    # Default ZIP Standalone Web Chat runtime export
    from ..cli.post_training_ui import run_export_ui
    run_export_ui(home, meta, yes=yes)