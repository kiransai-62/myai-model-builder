import typer
from ..core.paths import require_project_root
from ..core.console import console, print_info, print_error

def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    reload: bool = typer.Option(False, help="Enable auto-reload (dev mode)")
):
    """Start the FastAPI server."""
    root = require_project_root()
    
    try:
        import uvicorn
    except ImportError:
        print_error("Serving dependencies missing. Run: pip install -e '.[serve]' or pip install uvicorn fastapi")
        raise typer.Exit(1)

    console.print("\n[bold cyan]STARTING MYAI RUNTIME[/bold cyan]\n")
    print_info(f"Loading from: {root}")
    print_info(f"Server will be available at: http://{host}:{port}")
    print_info("Press Ctrl+C to stop\n")
    
    # Import here to avoid loading torch at CLI startup
    from ..serving.app import create_app
    
    app = create_app(root)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )