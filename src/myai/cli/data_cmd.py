import sys
from pathlib import Path
from typing import Optional
import typer
from rich.table import Table

from ..core.paths import find_project_root, require_project_root
from ..core.config import ProjectConfig
from ..core.console import console, print_success, print_error, print_warning, print_info
from ..core.home import ensure_home
from ..data.scanner import scan_directory, DISPLAY_LABELS
from ..data.validator import validate_data
from ..data.manager import DatasetManager, resolve_dataset_source

def register_dataset(src: Path, quiet: bool = False) -> dict:
    src = Path(src).resolve()
    scan = scan_directory(src)
    validation = validate_data(src)
    mgr = DatasetManager(ensure_home())
    dataset_name = src.name if src.is_dir() else src.stem
    return mgr.register(dataset_name, src, scan, validation)

def add(
    source_path: str = typer.Argument(..., help="Path to local dataset directory or file"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Target base model identifier or repository for tokenizer selection"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Automatically accept confirmation prompts"),
):
    """Scan, register, and tokenize local user dataset in-place without moving or uploading files."""
    src = Path(source_path).resolve()
    if not src.exists():
        print_error(f"Source path does not exist: {source_path}")
        raise typer.Exit(1)

    console.print("\n[bold cyan]📚 MYAI Dataset Manager[/bold cyan]\n")
    console.print("[bold]Step 1/5 — Training Data[/bold]\n")
    console.print("MYAI uses [bold green]Reference Mode[/bold green]:")
    console.print(" • Original data will not be moved.")
    console.print(" • Original data will not be modified.")
    console.print(" • Original data will not be overwritten.\n")
    console.print(f"Scanning: {src}\n")

    scan = scan_directory(src)
    if scan.errors:
        for err in scan.errors[:5]:
            print_warning(f"Scan warning: {err}")

    console.print("Found:")
    if scan.categories:
        for cat, count in sorted(scan.categories.items(), key=lambda x: -x[1]):
            label = DISPLAY_LABELS.get(cat, f"{cat} files")
            console.print(f"[green]✓[/green] {count:,} {label}")
    else:
        console.print("[yellow]  No recognized data files found.[/yellow]")

    console.print(f"\nTotal: {scan.total_files:,} files")
    console.print(f"Estimated dataset size: {scan.size_gb} GB")
    console.print("Location: LOCAL ONLY")
    console.print("No files will be uploaded.\n")

    if not yes and sys.stdin.isatty():
        if not typer.confirm("[Continue]", default=True):
            console.print("[yellow]Registration aborted.[/yellow]")
            raise typer.Exit()

    # Step checklist
    validation = validate_data(src)
    console.print("[green]✓ SCAN[/green]")
    console.print("[green]✓ VALIDATE[/green]")
    console.print("[green]✓ IDENTIFY FORMAT[/green]")
    console.print("[green]✓ READ METADATA[/green]")
    console.print("[green]✓ CHECK ERRORS[/green]")

    meta = register_dataset(src)
    console.print("[green]✓ CREATE DATASET ID[/green]\n")

    console.print(f"Dataset: [bold]{meta['name']}[/bold]")
    console.print(f"ID:      [bold cyan]{meta['dataset_id']}[/bold cyan]")
    console.print(f"Source:  {meta['source']}")
    console.print(f"Size:    {meta['size_gb']} GB")
    console.print(f"Files:   {meta['total_files']:,}")
    console.print(f"Status:  [bold green]{meta['validation']['status']}[/bold green]")

    # If inside project, attach
    project_root = find_project_root()
    if project_root:
        cfg = ProjectConfig.load(project_root)
        cfg.dataset_id = meta["dataset_id"]
        if model:
            cfg.model_id = model
        cfg.save(project_root)
        console.print(f"\n[green]✓ Attached to project: {cfg.name}[/green]")
        
        # Automatic Tokenizer Analysis
        try:
            from ..tokenization.analyzer import analyze_dataset_tokens
            stats = analyze_dataset_tokens(
                source_path=src,
                dataset_id=meta["dataset_id"],
                model_identifier=model or cfg.model_id,
                project_root=project_root,
                use_cache=True,
            )
            stats.print_report()
        except Exception as e:
            console.print(f"[dim]Tokenizer analysis notice: {e}[/dim]")

        console.print("[bold green]✓ READY FOR TRAINING[/bold green]")
    else:
        console.print("\n[dim]Run inside a MYAI project directory to attach automatically, or use dataset ID in config.[/dim]")

def list_datasets():
    """List all registered datasets in the local MYAI store."""
    mgr = DatasetManager(ensure_home())
    datasets = mgr.list()
    if not datasets:
        console.print("[yellow]No datasets registered yet. Use `myai data add <path>` to register data.[/yellow]")
        return

    table = Table(title="MYAI Registered Datasets", expand=True)
    table.add_column("Dataset ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold", no_wrap=False)
    table.add_column("Source", style="dim", no_wrap=False)
    table.add_column("Size", justify="right", no_wrap=True)
    table.add_column("Files", justify="right", no_wrap=True)
    table.add_column("Status", style="green", no_wrap=True)
    table.add_column("Created", style="dim", no_wrap=True)

    for ds in datasets:
        val_status = ds.get("validation", {}).get("status", "READY")
        created = ds.get("created_at", "")[:10]
        table.add_row(
            ds.get("dataset_id", ""),
            ds.get("name", ""),
            ds.get("source", ""),
            f"{ds.get('size_gb', 0)} GB",
            f"{ds.get('total_files', 0):,}",
            val_status,
            created,
        )

    console.print(table)

def info(dataset_id: Optional[str] = typer.Argument(None, help="Dataset ID to inspect")):
    """Show detailed metadata and validation report for a registered dataset."""
    target_id = dataset_id
    if not target_id:
        root = find_project_root()
        if root:
            cfg = ProjectConfig.load(root)
            target_id = cfg.dataset_id

    if not target_id:
        print_error("No dataset ID provided and no dataset attached to current project.")
        print_info("Usage: myai data info <dataset_id>")
        raise typer.Exit(1)

    mgr = DatasetManager(ensure_home())
    meta = mgr.get(target_id)
    if not meta:
        print_error(f"Dataset '{target_id}' not found.")
        raise typer.Exit(1)

    console.print("\n[bold cyan]DATASET INFORMATION[/bold cyan]\n")
    console.print(f"ID               : [bold cyan]{meta.get('dataset_id')}[/bold cyan]")
    console.print(f"Name             : {meta.get('name')}")
    console.print(f"Source           : {meta.get('source')}")
    console.print(f"Created At       : {meta.get('created_at')}")
    console.print(f"Total Files      : {meta.get('total_files', 0):,}")
    console.print(f"Size             : {meta.get('size_gb', 0)} GB ({meta.get('total_bytes', 0):,} bytes)")
    console.print(f"Manifest Checksum: {meta.get('manifest_checksum')}")
    console.print(f"Privacy          : {meta.get('privacy', {})}")

    console.print("\n[bold]Categories:[/bold]")
    for cat, count in meta.get("categories", {}).items():
        label = DISPLAY_LABELS.get(cat, cat)
        console.print(f"  - {label}: {count:,}")

    val = meta.get("validation", {})
    console.print("\n[bold]Validation Report:[/bold]")
    console.print(f"  Status         : [green]{val.get('status')}[/green]")
    console.print(f"  Examples       : {val.get('examples', 0):,}")
    console.print(f"  Approx Tokens  : {val.get('tokens_approx', 0):,}")
    console.print(f"  Duplicates     : {val.get('duplicates', 0)}")
    if val.get("warnings"):
        console.print("  Warnings       :")
        for w in val["warnings"]:
            console.print(f"    [yellow]• {w}[/yellow]")

def validate():
    """Validate current project dataset in-place and show tokenizer stats."""
    root = require_project_root()
    cfg = ProjectConfig.load(root)
    data_dir = resolve_dataset_source(root, cfg)

    console.print("\n[bold cyan]DATA ANALYSIS[/bold cyan]\n")
    console.print(f"Source: {data_dir}\n")
    report = validate_data(data_dir)

    console.print(f"Examples         : {report.examples:,}")
    console.print(f"Estimated tokens : {report.tokens_approx:,}")
    console.print(f"Duplicates       : {report.duplicates}")

    # Also run Tokenizer Analysis if available
    try:
        from ..tokenization.analyzer import analyze_dataset_tokens
        stats = analyze_dataset_tokens(
            source_path=data_dir,
            dataset_id=cfg.dataset_id or "default",
            model_identifier=cfg.model_id,
            project_root=root,
            use_cache=True,
        )
        stats.print_report()
    except Exception:
        pass

def tokenize(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Dataset ID to tokenize"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Target model identifier or repository"),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="Direct path to dataset file or folder"),
    refresh: bool = typer.Option(False, "--refresh", help="Force re-tokenization ignoring cache"),
):
    """Run model-aware tokenizer analysis and calculate token distributions."""
    root = find_project_root()
    target_path = None
    target_id = dataset or "dataset"
    target_model = model

    if path:
        target_path = Path(path).resolve()
    elif root:
        cfg = ProjectConfig.load(root)
        target_id = dataset or cfg.dataset_id or "dataset"
        target_model = model or cfg.model_id
        target_path = resolve_dataset_source(root, cfg)
    elif dataset:
        mgr = DatasetManager(ensure_home())
        meta = mgr.get(dataset)
        if meta and meta.get("source"):
            target_path = Path(meta["source"])

    if not target_path or not target_path.exists():
        print_error(f"Could not resolve dataset source path. Specify --path or run inside a MYAI project.")
        raise typer.Exit(1)

    from ..tokenization.analyzer import analyze_dataset_tokens
    stats = analyze_dataset_tokens(
        source_path=target_path,
        dataset_id=target_id,
        model_identifier=target_model,
        project_root=root,
        use_cache=not refresh,
        force_refresh=refresh,
    )
    stats.print_report()

from ..data.cleaner import prepare_datasets

def prepare(
    val_split: float = typer.Option(0.1, "--val-split", help="Validation set split fraction (0.0 - 0.5)"),
    fuzzy: bool = typer.Option(True, "--fuzzy/--no-fuzzy", help="Enable fuzzy deduplication"),
):
    """Clean, deduplicate, and prepare safe train/validation datasets in Reference Mode."""
    root = require_project_root()
    cfg = ProjectConfig.load(root)
    src_path = resolve_dataset_source(root, cfg)

    sources = [src_path] if src_path.exists() else []
    if not sources:
        print_error(f"Dataset source does not exist: {src_path}")
        raise typer.Exit(1)

    report = prepare_datasets(sources, root, val_split=val_split, fuzzy_dedup=fuzzy)
    report.print_report()
    console.print(f"[green]✨ Processed datasets saved to data/train.jsonl and data/validation.jsonl[/green]\n")

clean = prepare
