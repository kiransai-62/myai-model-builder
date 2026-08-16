import typer
from pathlib import Path
from ..core.console import console, print_error
from ..core.home import ensure_home
from ..models.trained_registry import list_trained

def export(
    model_id: str = typer.Argument(None, help="Trained model id (optional)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-accept prompts"),
):
    """Export ONLY the trained model folder — never the MYAI project."""
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

    from ..cli.post_training_ui import run_export_ui
    run_export_ui(home, meta, yes=yes)