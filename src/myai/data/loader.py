import json
import csv
from pathlib import Path

def load_file(file_path: Path) -> list[dict]:
    ext = file_path.suffix.lower()
    if ext == ".jsonl":
        return [json.loads(line) for line in file_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    elif ext == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, list) else [data]
    elif ext == ".csv":
        with open(file_path, "r", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    return []