import json
from pathlib import Path

MANIFEST_FILE = "model_manifest.json"

def write_manifest(model_dir: Path, data: dict):
    path = model_dir / MANIFEST_FILE
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def read_manifest(model_dir: Path) -> dict | None:
    path = model_dir / MANIFEST_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None

def list_installed(project_root: Path) -> list[dict]:
    base_dir = project_root / "models" / "base"
    installed = []
    if base_dir.exists():
        for d in base_dir.iterdir():
            if d.is_dir():
                manifest = read_manifest(d)
                if manifest:
                    installed.append(manifest)
    return installed