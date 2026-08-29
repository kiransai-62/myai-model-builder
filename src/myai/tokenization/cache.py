"""MYAI Tokenization Cache & Metadata Persistence.

Persists derived tokenization metadata and statistical distributions
with deterministic source fingerprint invalidation without ever storing raw dataset copies.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .stats import TokenStats
from ..core.home import ensure_home


def calculate_source_fingerprint(source_path: Path) -> str:
    """Computes a deterministic hash of source file paths, sizes, and mtimes."""
    entries = []
    p = Path(source_path)
    if not p.exists():
        return "empty"

    if p.is_file():
        st = p.stat()
        entries.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
    else:
        for f in sorted(p.rglob("*")):
            if f.is_file():
                st = f.stat()
                entries.append(f"{f.relative_to(p).as_posix()}:{st.st_size}:{int(st.st_mtime)}")

    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()[:16]


class TokenizationCache:
    """Manages reading and persisting tokenization metadata and stats."""

    def __init__(self, home: Optional[Path] = None):
        self.home = home or ensure_home()
        self.global_cache_dir = self.home / "tokenization"

    def get_cache_dir(self, dataset_id: str, project_dir: Optional[Path] = None) -> Path:
        if project_dir and (project_dir / "myai.yaml").exists():
            return project_dir / ".myai" / "tokenization" / dataset_id
        return self.global_cache_dir / dataset_id

    def load(self, dataset_id: str, source_path: Path, model_id: str, project_dir: Optional[Path] = None) -> Optional[TokenStats]:
        """Loads cached TokenStats if available and source fingerprint matches."""
        cache_dir = self.get_cache_dir(dataset_id, project_dir)
        meta_file = cache_dir / "metadata.json"
        stats_file = cache_dir / "stats.json"

        if not meta_file.exists() or not stats_file.exists():
            # Also check global cache fallback
            if project_dir:
                alt_dir = self.global_cache_dir / dataset_id
                meta_file = alt_dir / "metadata.json"
                stats_file = alt_dir / "stats.json"
                if not meta_file.exists() or not stats_file.exists():
                    return None

        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            current_fp = calculate_source_fingerprint(source_path)
            
            # Invalidation check: fingerprint & target model must match
            if meta.get("fingerprint") != current_fp or meta.get("model_id") != model_id:
                return None

            stats_data = json.loads(stats_file.read_text(encoding="utf-8"))
            return TokenStats.from_dict(stats_data)
        except Exception:
            return None

    def save(self, stats: TokenStats, source_path: Path, project_dir: Optional[Path] = None) -> None:
        """Persists TokenStats and metadata into project and global cache directories."""
        current_fp = calculate_source_fingerprint(source_path)
        metadata = {
            "dataset_id": stats.dataset_id,
            "model_id": stats.model_id,
            "tokenizer_name": stats.tokenizer_name,
            "fingerprint": current_fp,
            "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_samples": stats.total_samples,
            "total_tokens": stats.total_tokens,
            "avg_tokens": stats.avg_tokens,
            "max_tokens": stats.max_tokens,
            "min_tokens": stats.min_tokens,
        }

        # Save to both project and global locations
        targets = [self.get_cache_dir(stats.dataset_id, project_dir)]
        if project_dir:
            targets.append(self.global_cache_dir / stats.dataset_id)

        for d in targets:
            try:
                d.mkdir(parents=True, exist_ok=True)
                (d / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                (d / "stats.json").write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
            except Exception:
                pass
