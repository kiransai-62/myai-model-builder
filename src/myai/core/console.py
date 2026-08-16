import sys
import os

# Force UTF-8 on Windows to prevent UnicodeEncodeError with Rich's special characters
if os.name == "nt":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

from rich.console import Console

console = Console(force_terminal=True)

def print_success(msg: str): console.print(f"[bold green][OK][/bold green] {msg}")
def print_error(msg: str): console.print(f"[bold red][ERROR][/bold red] {msg}")
def print_warning(msg: str): console.print(f"[bold yellow][WARNING][/bold yellow] {msg}")
def print_info(msg: str): console.print(f"[cyan][INFO][/cyan] {msg}")