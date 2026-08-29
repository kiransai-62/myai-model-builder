import os
from pathlib import Path
from ..core.console import print_error

from typing import Optional

def prompt_data_path(message: str = "Path:", default: Optional[Path | str] = None) -> Path:
    """Ask the user for the data path. Re-asks until valid. Supports optional default."""
    first = True
    while True:
        if not first:
            print_error("Path not found.\n")
        prompt_msg = f"\n{message if first else 'Please enter a valid data file or folder path:'}"
        if default and first:
            prompt_msg += f" [press Enter for {default}]"
        raw = input(f"{prompt_msg}\n> ")
        first = False
        raw = raw.strip().strip('"').strip("'")

        if not raw and default:
            p = Path(default).expanduser().resolve()
            if p.exists():
                return p

        if raw.lower() in ("q", "quit", "exit"):
            raise SystemExit(0)
        if not raw:
            continue

        p = Path(raw).expanduser().resolve()
        if p.exists() and (p.is_dir() or p.is_file()):
            return p
        # loop re-asks

def prompt_output_path(message: str = "Output path:") -> Path:
    """Ask where to save the exported model. Creates the directory."""
    while True:
        raw = input(f"\n{message}\n> ").strip().strip('"').strip("'")
        if not raw:
            continue
        p = Path(raw).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p.resolve()
        except Exception as e:
            print_error(f"Cannot create directory: {e}")
