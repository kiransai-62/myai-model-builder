import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from .loader import load_file
from ..tokenization.formatter import format_record

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
        if file_path.is_file() and file_path.suffix.lower() in [".json", ".jsonl", ".csv", ".txt"]:
            try:
                examples = load_file(file_path)
            except Exception as e:
                report.warnings.append(f"Failed to load {file_path.name}: {e}")
                continue

            for ex in examples:
                sample = format_record(ex)
                text = sample.full_text.strip()
                if not text:
                    continue
                
                text_hash = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
                if text_hash in seen_hashes:
                    report.duplicates += 1
                    continue
                    
                seen_hashes.add(text_hash)
                report.examples += 1
                report.tokens_approx += max(1, len(text) // 4)
                
    return report