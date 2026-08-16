from dataclasses import dataclass, field
from pathlib import Path

CATEGORY_BY_EXT = {
    ".csv": "csv", ".json": "json", ".jsonl": "jsonl",
    ".txt": "text", ".md": "text",
    ".parquet": "parquet",
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".webp": "image", ".bmp": "image",
    ".wav": "audio", ".mp3": "audio", ".flac": "audio",
    ".mp4": "video", ".mkv": "video",
}

DISPLAY_LABELS = {
    "csv": "CSV", "json": "JSON files", "jsonl": "JSONL files",
    "text": "text documents", "image": "images", "audio": "audio files",
    "video": "video files", "parquet": "parquet files", "other": "other files",
}

@dataclass
class ScanResult:
    source: str
    total_files: int = 0
    total_bytes: int = 0
    categories: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    @property
    def size_gb(self) -> float:
        return round(self.total_bytes / 1024**3, 2)

def scan_directory(path: Path) -> ScanResult:
    result = ScanResult(source=str(path))
    p = Path(path)

    if p.is_file():
        try:
            result.total_files = 1
            result.total_bytes = p.stat().st_size
            cat = CATEGORY_BY_EXT.get(p.suffix.lower(), "other")
            result.categories[cat] = 1
        except OSError as e:
            result.errors.append(f"{p}: {e}")
        return result

    for f in p.rglob("*"):
        if not f.is_file():
            continue
        try:
            result.total_files += 1
            result.total_bytes += f.stat().st_size
            cat = CATEGORY_BY_EXT.get(f.suffix.lower(), "other")
            result.categories[cat] = result.categories.get(cat, 0) + 1
        except OSError as e:
            result.errors.append(f"{f}: {e}")

    return result

def is_readable(path: Path) -> bool:
    try:
        path = Path(path)
        if path.is_file():
            with open(path, "rb") as f:
                f.read(1)
            return True
        for child in Path(path).rglob("*"):
            if child.is_file():
                with open(child, "rb") as f:
                    f.read(1)
                return True
        return True                      # empty folder: readable, but no data
    except OSError:
        return False

def has_supported_format(scan) -> tuple:
    found = [DISPLAY_LABELS[c] for c in ("csv", "json", "jsonl", "text", "pdf")
             if scan.categories.get(c)]
    return (bool(found), ", ".join(found) or "none detected")
