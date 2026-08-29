import typer
from rich.table import Table
from ..hardware.detector import detect_hardware
from ..core.console import console

def check():
    hw = detect_hardware()
    
    console.print("\n[bold cyan]MYAI SYSTEM ANALYSIS[/bold cyan]\n")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")
    
    table.add_row("CPU", hw.cpu)
    table.add_row("RAM", f"{hw.ram_gb} GB")
    table.add_row("GPU", hw.gpu)
    table.add_row("VRAM", f"{hw.vram_gb} GB" if hw.vram_gb > 0 else "N/A")
    table.add_row("Storage", f"{hw.disk_gb} GB free")
    table.add_row("Compute Tier", hw.tier)
    
    status = "[green][PASS] Compatible[/green]" if hw.tier != "T0" else "[red][FAIL] Insufficient[/red]"
    table.add_row("Status", status)
    console.print(table)


def benchmark(
    steps: int = typer.Option(10, "--steps", "-s", help="Number of calibration steps to run"),
):
    """Run live hardware compute and memory throughput benchmark (Phase 12)."""
    from ..hardware.benchmark import run_hardware_benchmark, print_benchmark
    console.print("[dim]Benchmarking host hardware throughput...[/dim]")
    res = run_hardware_benchmark(steps=steps)
    print_benchmark(res)