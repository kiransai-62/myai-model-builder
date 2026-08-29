"""MYAI Streaming Tokenizer Analyzer.

Streams records incrementally across supported formats (.json, .jsonl, .csv, .txt),
extracts training representations, counts tokens with the model tokenizer,
and derives comprehensive token statistics and context fit reports.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from .formatter import format_record, detect_record_schema
from .stats import TokenStats, compute_token_stats
from .tokenizer import TokenizerEngine, get_tokenizer, resolve_model_repo
from .cache import TokenizationCache


SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".csv", ".txt"}


def discover_data_files(source_path: Path) -> List[Path]:
    """Recursively discovers all supported training files (.json, .jsonl, .csv, .txt)."""
    p = Path(source_path)
    if not p.exists():
        return []
    if p.is_file():
        return [p] if p.suffix.lower() in SUPPORTED_EXTENSIONS else []
    
    files: List[Path] = []
    for f in sorted(p.rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(f)
    return files


def stream_records_from_path(source_path: Path) -> Generator[Dict[str, Any], None, None]:
    """Generator that streams records from file or directory without loading all into RAM."""
    files = discover_data_files(source_path)

    for f in files:
        ext = f.suffix.lower()
        try:
            if ext == ".jsonl":
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                pass

            elif ext == ".json":
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    try:
                        data = json.load(fh)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    yield item
                                else:
                                    yield {"text": str(item)}
                        elif isinstance(data, dict):
                            yield data
                    except json.JSONDecodeError:
                        pass

            elif ext == ".csv":
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        yield dict(row)

            elif ext == ".txt":
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    # Treat non-empty lines or whole content as chunks
                    for line in fh:
                        line = line.strip()
                        if line:
                            yield {"text": line}
        except Exception:
            continue


def analyze_dataset_tokens(
    source_path: Path,
    dataset_id: str = "dataset",
    model_identifier: Optional[str] = None,
    project_root: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    use_cache: bool = True,
    force_refresh: bool = False,
    model_context_length: int = 4096,
) -> TokenStats:
    """Main analysis engine: Streams records, counts tokens via model tokenizer, and returns TokenStats."""
    source_path = Path(source_path).resolve()
    resolved_model = resolve_model_repo(model_identifier, project_root)
    cache = TokenizationCache()

    # 1. Check Cache
    if use_cache and not force_refresh:
        cached_stats = cache.load(dataset_id, source_path, resolved_model, project_root)
        if cached_stats:
            return cached_stats

    # 2. Load Model Tokenizer
    tokenizer = get_tokenizer(resolved_model, project_root)

    full_tokens: List[int] = []
    input_tokens: List[int] = []
    output_tokens: List[int] = []
    total_chars = 0
    total_words = 0
    dominant_schema = "instruction"

    start_time = time.time()
    sample_count = 0

    # 3. Stream & Tokenize
    for record in stream_records_from_path(source_path):
        sample = format_record(record, tokenizer)
        if sample_count == 0:
            dominant_schema = sample.schema

        # Calculate exact or tokenizer token counts
        full_tok_count = tokenizer.count_tokens(sample.full_text)
        inp_tok_count = tokenizer.count_tokens(sample.input_text)
        out_tok_count = tokenizer.count_tokens(sample.output_text)

        full_tokens.append(full_tok_count)
        input_tokens.append(inp_tok_count)
        output_tokens.append(out_tok_count)

        total_chars += sample.char_count
        total_words += sample.word_count
        sample_count += 1

        if progress_callback and sample_count % 500 == 0:
            elapsed = time.time() - start_time
            speed = sample_count / elapsed if elapsed > 0 else 0.0
            progress_callback(sample_count, sum(full_tokens), speed)

    # 4. Compute comprehensive stats
    stats = compute_token_stats(
        dataset_id=dataset_id,
        model_id=resolved_model,
        tokenizer_name=tokenizer.name,
        full_token_counts=full_tokens,
        input_token_counts=input_tokens,
        output_token_counts=output_tokens,
        total_chars=total_chars,
        total_words=total_words,
        schema_detected=dominant_schema,
        model_context_length=model_context_length,
    )

    # 5. Persist to cache
    if use_cache:
        cache.save(stats, source_path, project_root)

    return stats
