"""MYAI Dataset Scorer & Intelligence Analyzer (Report §9, §6.2).

Performs quality scoring, duplicate percentage estimation, and issue diagnostics
for datasets to drive model recommendation and training strategy feasibility.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

from .cleaner import _extract_samples_from_path, _scrub_pii


@dataclass
class DatasetSummary:
    num_samples: int = 0
    avg_tokens: int = 0
    tokens_approx: int = 0
    quality_score: int = 80
    dup_pct: float = 0.0
    issues: List[str] = field(default_factory=list)
    exact_duplicates: int = 0
    pii_count: int = 0
    total_bytes: int = 0
    max_tokens: int = 0
    token_stats: Optional[Any] = None



def analyze_dataset(data_path: Path) -> DatasetSummary:
    """Analyze a dataset file or directory, returning comprehensive intelligence summary."""
    if not data_path or not data_path.exists():
        return DatasetSummary(
            num_samples=0,
            avg_tokens=0,
            tokens_approx=0,
            quality_score=0,
            dup_pct=0.0,
            issues=["Data source does not exist or is empty."],
        )

    raw_samples = _extract_samples_from_path(data_path)
    total_samples = len(raw_samples)
    if total_samples == 0:
        return DatasetSummary(
            num_samples=0,
            avg_tokens=0,
            tokens_approx=0,
            quality_score=0,
            dup_pct=0.0,
            issues=["No valid JSON, JSONL, or CSV prompt/response samples found."],
        )

    seen_hashes = set()
    exact_dups = 0
    total_chars = 0
    pii_count = 0
    empty_count = 0

    for s in raw_samples:
        prompt = str(s.get("prompt") or s.get("instruction") or s.get("input") or "")
        resp = str(s.get("response") or s.get("output") or s.get("completion") or "")

        if not prompt.strip() or not resp.strip():
            empty_count += 1
            continue

        p_hash = hashlib.md5(prompt.strip().lower().encode("utf-8")).hexdigest()
        if p_hash in seen_hashes:
            exact_dups += 1
        else:
            seen_hashes.add(p_hash)

        _, p_pii = _scrub_pii(prompt)
        _, r_pii = _scrub_pii(resp)
        pii_count += (p_pii + r_pii)

        total_chars += len(prompt) + len(resp)

    valid_samples = max(1, total_samples - empty_count)
    tokens_approx = total_chars // 4
    avg_tokens = tokens_approx // valid_samples if valid_samples else 0
    dup_pct = round((exact_dups / total_samples) * 100.0, 1) if total_samples else 0.0

    # Calculate quality score (0..100)
    score = 100
    issues: List[str] = []

    if dup_pct > 15.0:
        score -= 20
        issues.append(f"High duplication rate ({dup_pct}% exact duplicates).")
    elif dup_pct > 5.0:
        score -= 10
        issues.append(f"Moderate duplication rate ({dup_pct}% duplicates).")

    if empty_count > 0:
        pct_empty = (empty_count / total_samples) * 100.0
        score -= min(30, int(pct_empty * 2))
        issues.append(f"{empty_count} empty or malformed samples detected ({pct_empty:.1f}%).")

    if pii_count > 0:
        score -= min(15, pii_count * 2)
        issues.append(f"{pii_count} PII/secret instances detected.")

    if avg_tokens < 20:
        score -= 15
        issues.append(f"Short average prompt/response length (~{avg_tokens} tokens).")

    score = max(10, min(100, score))

    src_files = [data_path] if data_path.is_file() else list(data_path.rglob("*"))
    total_bytes = sum(f.stat().st_size for f in src_files if f.is_file())

    return DatasetSummary(
        num_samples=total_samples,
        avg_tokens=avg_tokens,
        tokens_approx=tokens_approx,
        quality_score=score,
        dup_pct=dup_pct,
        issues=issues,
        exact_duplicates=exact_dups,
        pii_count=pii_count,
        total_bytes=total_bytes,
    )
