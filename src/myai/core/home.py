import os
from pathlib import Path

SUBDIRS = [
    "config",
    "datasets",
    "models/base",
    "models/adapters",
    "models/trained",
    "runs",
    "cache",
    "logs",
]

def get_home() -> Path:
    override = os.environ.get("MYAI_HOME")
    return Path(override) if override else Path.home() / ".myai"

def ensure_home() -> Path:
    home = get_home()
    for sub in SUBDIRS:
        (home / sub).mkdir(parents=True, exist_ok=True)
    return home
