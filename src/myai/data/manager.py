import hashlib
import json
import time
from pathlib import Path

from ..core.config import ProjectConfig
from .scanner import ScanResult

PROCESSING_VERSION = "1"

def make_dataset_id() -> str:
    return "ds_" + hashlib.sha256(f"{time.time_ns()}".encode()).hexdigest()[:8]

def _manifest_checksum(src: Path) -> str:
    """Lightweight checksum of (path|size|mtime) — detects changes without re-reading content."""
    entries = []
    if src.is_file():
        st = src.stat()
        entries.append(f"{src.name}|{st.st_size}|{int(st.st_mtime)}")
    else:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                st = f.stat()
                entries.append(f"{f.relative_to(src).as_posix()}|{st.st_size}|{int(st.st_mtime)}")
    digest = hashlib.sha256("\n".join(entries).encode()).hexdigest()[:16]
    return f"sha256:{digest}"

class DatasetManager:
    def __init__(self, home: Path):
        self.datasets_dir = Path(home) / "datasets"

    def register(self, name: str, source: Path, scan: ScanResult, validation) -> dict:
        dataset_id = make_dataset_id()
        ds_dir = self.datasets_dir / dataset_id
        (ds_dir / "processed").mkdir(parents=True, exist_ok=True)
        (ds_dir / "cache").mkdir(parents=True, exist_ok=True)

        metadata = {
            "dataset_id": dataset_id,
            "name": name,
            "source": str(source),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_files": scan.total_files,
            "total_bytes": scan.total_bytes,
            "size_gb": scan.size_gb,
            "categories": scan.categories,
            "validation": {
                "status": "READY" if validation.examples > 0 else "INDEXED",
                "examples": validation.examples,
                "tokens_approx": validation.tokens_approx,
                "duplicates": validation.duplicates,
                "warnings": validation.warnings if hasattr(validation, "warnings") else [],
            },
            "processing_version": PROCESSING_VERSION,
            "manifest_checksum": _manifest_checksum(source),
            "privacy": {"upload": False, "location": "local"},
        }

        (ds_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    def list(self) -> list:
        out = []
        if self.datasets_dir.exists():
            for meta_file in sorted(self.datasets_dir.glob("*/metadata.json")):
                out.append(json.loads(meta_file.read_text(encoding="utf-8")))
        return out

    def get(self, dataset_id: str):
        meta_file = self.datasets_dir / dataset_id / "metadata.json"
        if meta_file.exists():
            return json.loads(meta_file.read_text(encoding="utf-8"))
        return None

def resolve_dataset_source(root: Path, cfg: ProjectConfig) -> Path:
    """Training/index read the ORIGINAL location — never a copy."""
    if cfg.dataset_id:
        meta = DatasetManager(ensure_home()).get(cfg.dataset_id)
        if meta:
            return Path(meta["source"])
    return root / cfg.data_path

def ensure_home():
    from ..core.home import ensure_home as _e
    return _e()
