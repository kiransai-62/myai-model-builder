from pathlib import Path
from typing import Optional
import typer
from ..core.paths import require_project_root
from ..core.config import ProjectConfig
from ..core.console import console, print_success, print_error, print_info
from ..knowledge.index_builder import build_index

def build():
    """Build or refresh the knowledge index from project documents."""
    root = require_project_root()
    cfg = ProjectConfig.load(root)
    console.print("\n[bold cyan]BUILDING KNOWLEDGE INDEX[/bold cyan]\n")
    count = build_index(root, cfg)
    if count == 0:
        print_error("No text found to index.")
        raise typer.Exit(1)
    print_success(f"Indexed {count:,} chunks into indexes/chunks.jsonl")

def add_document(
    path: str = typer.Argument(..., help="Path to document or folder to index"),
):
    """Add a file or directory to the project's knowledge documents."""
    root = require_project_root()
    src = Path(path).expanduser().resolve()
    if not src.exists():
        print_error(f"Path does not exist: {src}")
        raise typer.Exit(1)
    
    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    if src.is_file():
        import shutil
        dest = docs_dir / src.name
        shutil.copy2(src, dest)
        print_success(f"Added {src.name} to project docs.")
    else:
        import shutil
        dest = docs_dir / src.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        print_success(f"Added directory {src.name} to project docs.")
    
    # Auto-rebuild index
    cfg = ProjectConfig.load(root)
    count = build_index(root, cfg)
    print_info(f"Knowledge index updated with {count:,} total chunks.")

def list_indexes():
    """List existing knowledge indexes in the project."""
    root = require_project_root()
    idx_file = root / "indexes" / "chunks.jsonl"
    if not idx_file.exists():
        console.print("[yellow]No knowledge index found. Use `myai index build` or `myai index add <path>`.[/yellow]")
        return
    
    lines = [l for l in idx_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    console.print(f"\n[bold cyan]KNOWLEDGE INDEX[/bold cyan]")
    console.print(f"Index Path   : {idx_file}")
    console.print(f"Total Chunks : [bold green]{len(lines):,}[/bold green]\n")