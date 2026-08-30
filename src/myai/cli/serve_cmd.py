import typer
from typing import Optional
from ..core.paths import require_project_root
from ..core.console import console, print_info, print_error

def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind (default: loopback 127.0.0.1)"),
    port: int = typer.Option(8000, help="Port to bind"),
    reload: bool = typer.Option(False, help="Enable auto-reload (dev mode)"),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="Require this API key on /ask, /ask-stream, and /info endpoints. "
             "Alternatively set the MYAI_API_KEY environment variable.",
        envvar="MYAI_API_KEY",
    ),
    max_concurrent: int = typer.Option(
        2,
        "--max-concurrent",
        help="Maximum simultaneous inference requests (protects GPU/CPU memory)",
    ),
    rate_limit: int = typer.Option(
        60,
        "--rate-limit",
        help="Maximum requests per minute per client IP",
    ),
):
    """Start the MYAI FastAPI inference server (local-first, binds to 127.0.0.1 by default)."""
    root = require_project_root()

    try:
        import uvicorn
    except ImportError:
        print_error("Serving dependencies missing. Run: pip install -e '.[serve]' or pip install uvicorn fastapi")
        raise typer.Exit(1)

    console.print("\n[bold cyan]STARTING MYAI RUNTIME[/bold cyan]\n")
    print_info(f"Loading from: {root}")
    print_info(f"Server will be available at: http://{host}:{port}")
    if api_key:
        print_info("Authentication: [bold green]API key required[/bold green]")
    else:
        print_info("Authentication: [yellow]None (local dev mode)[/yellow]")
    print_info("Press Ctrl+C to stop\n")

    # Import here to avoid loading torch at CLI startup
    from ..serving.app import create_app

    app = create_app(root, api_key=api_key, max_concurrent=max_concurrent, rate_limit_per_min=rate_limit)

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )