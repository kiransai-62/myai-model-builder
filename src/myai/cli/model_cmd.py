import json
from pathlib import Path
from datetime import datetime
import typer
from rich.table import Table
from ..models.registry import get_registry_models
from ..models.downloader import download_model
from ..models.manifest import write_manifest, list_installed
from ..hardware.detector import detect_hardware
from ..core.console import console, print_success, print_error, print_info
from ..core.paths import require_project_root
from ..core.config import ProjectConfig

from ..models.trained_registry import list_trained as list_trained_models
from ..core.home import ensure_home

def list_models():
    models = get_registry_models()
    hw = detect_hardware()
    
    console.print("\n[bold cyan]AVAILABLE BASE MODELS[/bold cyan]\n")
    table = Table(title="Model Registry")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Size")
    table.add_column("Min VRAM")
    table.add_column("Your VRAM")
    
    for m in models:
        your_vram = f"{hw.vram_gb} GB" if hw.vram_gb > 0 else "N/A"
        status = "[green]✓[/green]" if hw.vram_gb >= m.vram_min else "[red]✗[/red]"
        table.add_row(m.id, m.name, m.parameters, f"{m.vram_min} GB", f"{your_vram} {status}")
        
    console.print(table)

    trained = list_trained_models(ensure_home())
    if trained:
        t = Table(title="Model Registry (Trained)")
        t.add_column("ID", style="cyan")
        t.add_column("Base Model")
        t.add_column("Dataset")
        t.add_column("Training")
        t.add_column("Evaluation")
        t.add_column("Status")
        for m in trained:
            t.add_row(
                m.get("id", "?"),
                m.get("base_model", "?"),
                m.get("dataset", "?"),
                m.get("method", "QLoRA"),
                m.get("evaluation", "—"),
                f"[green]{m.get('status', 'READY')}[/green]",
            )
        console.print(t)

def add(model_id: str):
    root = require_project_root()
    models = get_registry_models()
    spec = next((m for m in models if m.id == model_id), None)
    
    if not spec:
        print_error(f"Model '{model_id}' not found in registry.")
        raise typer.Exit(1)
        
    hw = detect_hardware()
    if hw.vram_gb > 0 and hw.vram_gb < spec.vram_min:
        print_error(f"BLOCKED: Your VRAM ({hw.vram_gb} GB) is below the minimum requirement ({spec.vram_min} GB).")
        if not typer.confirm("Force download anyway?", default=False):
            raise typer.Exit(1)
            
    dest_dir = root / "models" / "base" / spec.id
    if (dest_dir / "model_manifest.json").exists():
        print_info(f"Model {spec.id} is already installed.")
        return

    success = download_model(spec.repository, dest_dir)
    if success:
        write_manifest(dest_dir, {
            "id": spec.id,
            "name": spec.name,
            "repository": spec.repository,
            "parameters": spec.parameters,
            "license": spec.license
        })
        print_success(f"Model {spec.name} installed successfully.")

def use(model_id: str):
    root = require_project_root()
    cfg = ProjectConfig.load(root)
    
    installed = list_installed(root)
    if not any(m["id"] == model_id for m in installed):
        print_error(f"Model {model_id} is not installed. Run `myai model add {model_id}` first.")
        raise typer.Exit(1)
        
    cfg.model_id = model_id
    cfg.save(root)
    print_success(f"Project default model set to: {model_id}")

def list_trained():
    """List all locally trained model versions."""
    root = require_project_root()
    trained_dir = root / "models" / "trained"
    
    if not trained_dir.exists():
        print_info("No trained models found.")
        return

    console.print("\n[bold cyan]TRAINED MODELS[/bold cyan]\n")
    table = Table(title="Local Adapters")
    table.add_column("Name", style="cyan")
    table.add_column("Base Model")
    table.add_column("Method")
    table.add_column("Steps")
    table.add_column("Duration")
    table.add_column("Active")
    
    cfg = ProjectConfig.load(root)
    
    for d in sorted(trained_dir.iterdir()):
        if d.is_dir() and (d / "metadata.json").exists():
            meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
            is_active = "✓" if d.name == cfg.name else ""
            
            table.add_row(
                d.name,
                meta.get("base_model_id", "?"),
                meta.get("method", "?"),
                str(meta.get("steps", "?")),
                f"{meta.get('duration_seconds', 0) // 60}m",
                f"[green]{is_active}[/green]"
            )
            
    console.print(table)

def use_trained(name: str):
    """Switch the active project to a different trained adapter."""
    root = require_project_root()
    cfg = ProjectConfig.load(root)
    
    target_dir = root / "models" / "trained" / name
    if not target_dir.exists():
        print_error(f"Trained model '{name}' not found.")
        raise typer.Exit(1)
        
    cfg.name = name
    cfg.save(root)
    print_success(f"Active trained model switched to: {name}")