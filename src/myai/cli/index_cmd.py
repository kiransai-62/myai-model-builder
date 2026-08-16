import typer
from ..core.paths import require_project_root
from ..core.config import ProjectConfig
from ..core.console import console, print_success, print_error
from ..knowledge.index_builder import build_index

def build():
    root = require_project_root()
    cfg = ProjectConfig.load(root)
    console.print("\n[bold cyan]BUILDING KNOWLEDGE INDEX[/bold cyan]\n")
    count = build_index(root, cfg)
    if count == 0:
        print_error("No text found to index.")
        raise typer.Exit(1)
    print_success(f"Indexed {count:,} chunks into indexes/chunks.jsonl")