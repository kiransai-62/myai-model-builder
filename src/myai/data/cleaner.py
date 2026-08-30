"""MYAI Dataset Intelligence — Cleaner, Deduplicator & Leakage Detector (Report §9, §6.2).

Processes data in-place (Reference Mode) and generates safe, processed copies 
in the project workspace without mutating the original source files.

Capabilities:
1. PII & Secret Scrubbing (Emails, Phone numbers, API keys)
2. Exact & Fuzzy Deduplication
3. Train/Validation Contamination (Leakage) Detection
4. Transparent Cleaning Report (Report §18: Show reasoning/assumptions)
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


# --- PII & Secret Regex Patterns ---
PII_PATTERNS = [
    # Email addresses
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]"),
    # Phone numbers (E.164 and common North American formats)
    (r"\b(?:\+?1[-.\\s]?)?\(?[0-9]{3}\)?[-.\\s]?[0-9]{3}[-.\\s]?[0-9]{4}\b", "[PHONE_REDACTED]"),
    # OpenAI / Anthropic / HuggingFace / GitHub / AWS-AKIA style tokens
    (r"\b(?:sk-|pk-|ant-|ghp_|gho_|ghu_|ghs_|ghr_|hf_|AKIA|ABIA|ACCA)[A-Za-z0-9_\-]{16,}\b", "[SECRET_REDACTED]"),
    # AWS Secret Access Key (40 chars, base64)
    (r"(?i)aws[_\-\s]?secret[_\-\s]?access[_\-\s]?key\s*[:=]\s*[A-Za-z0-9/+]{40}", "[SECRET_REDACTED]"),
    # Generic key=value credential patterns (api_key, auth_token, access_token, secret_key, password)
    (r"(?i)\b(?:api[_\-]?key|secret[_\-]?key|access[_\-]?token|auth[_\-]?token|passwd|password)\s*[:=]\s*['\"]?[A-Za-z0-9_.+/\-]{16,}['\"]?", "[CREDENTIAL_REDACTED]"),
    # JWT Bearer tokens (three base64url segments separated by dots)
    (r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b", "[JWT_REDACTED]"),
    # PEM private key headers (RSA, EC, PKCS8)
    (r"-----BEGIN (?:RSA |EC |OPENSSH |PRIVATE KEY|ENCRYPTED PRIVATE KEY).*?PRIVATE KEY-----", "[PRIVATE_KEY_REDACTED]"),
]


@dataclass
class CleaningReport:
    total_raw_samples: int = 0
    exact_duplicates_removed: int = 0
    fuzzy_duplicates_removed: int = 0
    pii_redactions_made: int = 0
    empty_malformed_removed: int = 0
    leakage_detected: bool = False
    leakage_samples: List[str] = field(default_factory=list)
    train_samples_count: int = 0
    val_samples_count: int = 0

    def print_report(self) -> None:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.print("\n[bold cyan]🧹 Dataset Intelligence & Cleaning Report[/bold cyan]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim")
        table.add_column("Count / Status")

        table.add_row("Raw Samples Ingested", str(self.total_raw_samples))
        table.add_row("Exact Duplicates Removed", f"[red]{self.exact_duplicates_removed}[/red]")
        table.add_row("Fuzzy Duplicates Removed", f"[yellow]{self.fuzzy_duplicates_removed}[/yellow]")
        table.add_row("PII / Secrets Redacted", f"[yellow]{self.pii_redactions_made}[/yellow]")
        table.add_row("Empty / Malformed Dropped", f"[red]{self.empty_malformed_removed}[/red]")

        leak_status = "[red]⚠️ YES (Filtered)[/red]" if self.leakage_detected else "[green]✅ NONE[/green]"
        table.add_row("Train/Val Leakage", leak_status)
        table.add_row("Final Clean Train Samples", f"[bold green]{self.train_samples_count}[/bold green]")
        table.add_row("Final Clean Val Samples", f"[bold green]{self.val_samples_count}[/bold green]")

        console.print(table)

        if self.leakage_detected:
            console.print("\n[bold red]⚠️ Contamination Warning:[/bold red] Evaluation prompts were found in the training set.")
            console.print("MYAI has automatically removed them from the training set to prevent false-positive evaluation scores.\n")


def _hash_text(text: str) -> str:
    """Deterministic hash for exact deduplication."""
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()


def _scrub_pii(text: str) -> Tuple[str, int]:
    """Replaces PII and secrets with redaction tokens."""
    redaction_count = 0
    for pattern, replacement in PII_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            redaction_count += len(matches)
            text = re.sub(pattern, replacement, text)
    return text, redaction_count


def _is_fuzzy_duplicate(new_prompt: str, existing_prompts: List[str], threshold: float = 0.85) -> bool:
    """Lightweight fuzzy matching using SequenceMatcher."""
    for existing in existing_prompts:
        if SequenceMatcher(None, new_prompt, existing).ratio() > threshold:
            return True
    return False


def _extract_samples_from_path(src: Path) -> List[Dict[str, Any]]:
    """Extract prompt/response or instruction/input/output samples from file or directory."""
    raw_samples: List[Dict[str, Any]] = []
    if not src.exists():
        return raw_samples

    files = [src] if src.is_file() else [f for f in src.rglob("*") if f.is_file()]

    for f in files:
        suffix = f.suffix.lower()
        try:
            if suffix == ".jsonl":
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                raw_samples.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            elif suffix == ".json":
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    try:
                        data = json.load(fh)
                        if isinstance(data, list):
                            raw_samples.extend(data)
                        elif isinstance(data, dict):
                            raw_samples.append(data)
                    except json.JSONDecodeError:
                        pass
            elif suffix == ".csv":
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        raw_samples.append(row)
        except Exception:
            continue

    return raw_samples


def prepare_datasets(
    sources: List[Path],
    project_dir: Path,
    val_split: float = 0.1,
    fuzzy_dedup: bool = True,
    seed: int = 42,
) -> CleaningReport:
    """
    Main orchestrator: Reads reference sources -> Cleans -> Splits -> Saves to workspace.
    Strictly respects Reference Mode (Report §6.2). Original source files remain untouched.
    """
    raw_samples: List[Dict[str, Any]] = []

    # 1. Ingest from Reference Sources
    for src in sources:
        raw_samples.extend(_extract_samples_from_path(src))

    report = CleaningReport(
        total_raw_samples=len(raw_samples),
        exact_duplicates_removed=0,
        fuzzy_duplicates_removed=0,
        pii_redactions_made=0,
        empty_malformed_removed=0,
        leakage_detected=False,
    )

    cleaned_samples: List[Dict[str, Any]] = []
    seen_hashes: Set[str] = set()
    seen_prompts: List[str] = []

    # 2. Clean, Scrub, and Deduplicate
    for sample in raw_samples:
        prompt = (
            sample.get("prompt")
            or sample.get("instruction")
            or sample.get("input")
            or sample.get("question")
            or ""
        )
        response = (
            sample.get("response")
            or sample.get("output")
            or sample.get("completion")
            or sample.get("answer")
            or ""
        )

        if not isinstance(prompt, str) or not isinstance(response, str):
            report.empty_malformed_removed += 1
            continue

        prompt = prompt.strip()
        response = response.strip()

        # Drop empty/malformed
        if not prompt or not response:
            report.empty_malformed_removed += 1
            continue

        # PII & Secret Scrubbing
        prompt, p_redactions = _scrub_pii(prompt)
        response, r_redactions = _scrub_pii(response)
        report.pii_redactions_made += (p_redactions + r_redactions)

        # Exact Deduplication
        p_hash = _hash_text(prompt)
        if p_hash in seen_hashes:
            report.exact_duplicates_removed += 1
            continue

        # Fuzzy Deduplication
        if fuzzy_dedup and _is_fuzzy_duplicate(prompt, seen_prompts):
            report.fuzzy_duplicates_removed += 1
            continue

        seen_hashes.add(p_hash)
        seen_prompts.append(prompt)

        # Build standardized cleaned record
        cleaned_record = dict(sample)
        cleaned_record["prompt"] = prompt
        cleaned_record["response"] = response
        cleaned_samples.append(cleaned_record)

    # 3. Train / Validation Split
    rng = random.Random(seed)
    rng.shuffle(cleaned_samples)

    split_idx = int(len(cleaned_samples) * (1 - val_split)) if len(cleaned_samples) > 1 else len(cleaned_samples)
    train_set = cleaned_samples[:split_idx]
    val_set = cleaned_samples[split_idx:] if split_idx < len(cleaned_samples) else []

    # 4. Leakage (Contamination) Detection
    val_hashes = {_hash_text(s["prompt"]) for s in val_set}
    clean_train_set: List[Dict[str, Any]] = []

    for sample in train_set:
        if _hash_text(sample["prompt"]) in val_hashes:
            report.leakage_detected = True
            report.leakage_samples.append(sample["prompt"][:50] + "...")
        else:
            clean_train_set.append(sample)

    report.train_samples_count = len(clean_train_set)
    report.val_samples_count = len(val_set)

    # 5. Save Processed Copies to Workspace (Reference Mode Safe)
    out_dir = project_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "validation").mkdir(parents=True, exist_ok=True)

    # Write root train.jsonl / validation.jsonl
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for s in clean_train_set:
            f.write(json.dumps(s) + "\n")

    with open(out_dir / "validation.jsonl", "w", encoding="utf-8") as f:
        for s in val_set:
            f.write(json.dumps(s) + "\n")

    # Also mirror into data/train/train.jsonl for compatibility with standard loader
    with open(out_dir / "train" / "train.jsonl", "w", encoding="utf-8") as f:
        for s in clean_train_set:
            f.write(json.dumps(s) + "\n")

    if val_set:
        with open(out_dir / "validation" / "validation.jsonl", "w", encoding="utf-8") as f:
            for s in val_set:
                f.write(json.dumps(s) + "\n")

    return report
