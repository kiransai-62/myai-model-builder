import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from .loader import load_file

@dataclass
class DataReport:
    examples: int = 0
    tokens_approx: int = 0
    duplicates: int = 0
    warnings: list[str] = field(default_factory=list)

def validate_data(data_dir: Path) -> DataReport:
    report = DataReport()
    seen_hashes = set()
    
    files = [data_dir] if data_dir.is_file() else list(data_dir.rglob("*"))
    for file_path in files:
        if file_path.is_file() and file_path.suffix.lower() in [".json", ".jsonl", ".csv"]:
            examples = load_file(file_path)
            for ex in examples:
                text = str(ex.get("text", ex.get("prompt", "")) + ex.get("response", ""))
                if not text: continue
                
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in seen_hashes:
                    report.duplicates += 1
                    continue
                    
                seen_hashes.add(text_hash)
                report.examples += 1
                report.tokens_approx += len(text) // 4  # Rough char-to-token ratio
                
    return report