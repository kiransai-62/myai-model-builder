from pathlib import Path

CONFIG_FILE = "myai.yaml"

def find_project_root() -> Path | None:
    current = Path.cwd().resolve()
    for p in [current, *current.parents]:
        if (p / CONFIG_FILE).exists():
            return p
    return None

def require_project_root() -> Path:
    root = find_project_root()
    if not root:
        from .console import print_error
        print_error(f"No {CONFIG_FILE} found. Run `myai init <project>` first.")
        raise SystemExit(1)
    return root

get_active_project_dir = require_project_root