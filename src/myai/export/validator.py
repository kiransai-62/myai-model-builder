"""
Export package validation — verifies a MYAI model ZIP is complete,
clean, and contains no project internals or sensitive data.
"""

import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationResult:
    """Aggregate result of all package validation checks."""
    passed: bool
    checks: List[CheckResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def failed_checks(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed]


# ── Patterns for detecting sensitive content ─────────────────────

_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token)\s*[:=]\s*\S+"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),       # OpenAI-style key
    re.compile(r"hf_[a-zA-Z0-9]{20,}"),        # Hugging Face token
    re.compile(r"ghp_[a-zA-Z0-9]{20,}"),       # GitHub PAT
]

_ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"[A-Z]:\\\\"),                  # Windows: C:\\
    re.compile(r"[A-Z]:\\[^\\]"),               # Windows: C:\Users
    re.compile(r'"/(?:home|Users|root|tmp)/'),  # Unix absolute paths in JSON strings
]

# MYAI source directories that must never appear in an export
_MYAI_SOURCE_DIRS = [
    "myai/cli/", "myai/core/", "myai/training/", "myai/evaluation/",
    "myai/export/", "myai/hardware/", "myai/models/", "myai/system/",
    "myai/data/", "myai/knowledge/", "myai/registry/", "myai/serving/",
    "src/myai/",
]

# Temporary / unwanted file patterns
_TEMP_PATTERNS = [
    "__pycache__/", ".pyc", ".pyo", ".tmp", ".temp",
    ".DS_Store", "Thumbs.db", "desktop.ini",
]

# Dataset file extensions that should not be in the package
_DATASET_EXTENSIONS = {".csv", ".tsv", ".jsonl", ".parquet", ".arrow"}


def validate_package(zip_path: Path) -> ValidationResult:
    """
    Run all validation checks against an exported model ZIP.

    This function opens but does not extract the ZIP. All checks are
    performed by inspecting the ZIP's namelist and reading small files
    (metadata.json) in memory.
    """
    zip_path = Path(zip_path)
    checks: List[CheckResult] = []
    warnings: List[str] = []

    # 1. ZIP exists
    checks.append(CheckResult(
        "ZIP file exists",
        zip_path.exists(),
        "" if zip_path.exists() else f"File not found: {zip_path}",
    ))
    if not zip_path.exists():
        return ValidationResult(passed=False, checks=checks, warnings=warnings)

    # 2. ZIP can be opened
    try:
        zf = zipfile.ZipFile(zip_path, "r")
        checks.append(CheckResult("ZIP is valid archive", True))
    except (zipfile.BadZipFile, Exception) as e:
        checks.append(CheckResult("ZIP is valid archive", False, str(e)))
        return ValidationResult(passed=False, checks=checks, warnings=warnings)

    with zf:
        names = zf.namelist()

        # Helper: check if any entry starts with or contains a prefix
        def _has_prefix(prefix: str) -> bool:
            return any(n == prefix or n.startswith(prefix) for n in names)

        # Determine the top-level prefix (ZIP may contain a root folder)
        top_prefix = ""
        if names:
            first = names[0]
            if "/" in first:
                candidate = first.split("/")[0] + "/"
                if all(n.startswith(candidate) or n == candidate.rstrip("/") for n in names):
                    top_prefix = candidate

        def _resolve(name: str) -> str:
            return top_prefix + name

        # 3. Required model files exist
        model_files = [n for n in names if "/model/" in n or n.startswith("model/")]
        has_model = bool(model_files)
        checks.append(CheckResult(
            "Model files present",
            has_model,
            "" if has_model else "No model/ directory found in archive",
        ))

        # 4. Tokenizer files exist
        tok_files = [n for n in names if "/tokenizer/" in n or n.startswith("tokenizer/")]
        has_tokenizer = bool(tok_files)
        checks.append(CheckResult(
            "Tokenizer files present",
            has_tokenizer,
            "" if has_tokenizer else "No tokenizer/ directory found in archive",
        ))

        # 5. metadata.json exists
        has_metadata = any("metadata.json" in n for n in names)
        checks.append(CheckResult(
            "metadata.json present",
            has_metadata,
            "" if has_metadata else "metadata.json not found",
        ))

        # 6. evaluation.json exists
        has_eval = any("evaluation.json" in n for n in names)
        checks.append(CheckResult(
            "evaluation.json present",
            has_eval,
            "" if has_eval else "evaluation.json not found",
        ))

        # 7. README.md exists
        has_readme = any("README.md" in n for n in names)
        checks.append(CheckResult(
            "README.md present",
            has_readme,
            "" if has_readme else "README.md not found",
        ))

        # 8. loader.py exists
        has_loader = any("loader.py" in n for n in names)
        checks.append(CheckResult(
            "loader.py present",
            has_loader,
            "" if has_loader else "loader.py not found",
        ))

        # 8b. Standalone Chat UI exists (chat/app.py, chat/ui.py, chat/config.json, chat/web/index.html)
        has_chat_app = any("chat/app.py" in n for n in names)
        has_chat_ui = any("chat/ui.py" in n for n in names)
        has_chat_cfg = any("chat/config.json" in n for n in names)
        has_chat_html = any("chat/web/index.html" in n for n in names)
        has_chat = has_chat_app and has_chat_ui and has_chat_cfg and has_chat_html
        chat_missing = []
        if not has_chat_app: chat_missing.append("chat/app.py")
        if not has_chat_ui: chat_missing.append("chat/ui.py")
        if not has_chat_cfg: chat_missing.append("chat/config.json")
        if not has_chat_html: chat_missing.append("chat/web/index.html")
        checks.append(CheckResult(
            "Standalone Chat UI present",
            has_chat,
            f"Missing chat files: {', '.join(chat_missing)}" if chat_missing else "",
        ))

        # 9. No MYAI source code
        found_src = [n for n in names if any(s in n for s in _MYAI_SOURCE_DIRS)]
        checks.append(CheckResult(
            "No MYAI source code",
            not found_src,
            f"Found MYAI source files: {', '.join(found_src[:5])}" if found_src else "",
        ))

        # 10. No .git directory
        has_git = any(".git/" in n or n == ".git" for n in names)
        checks.append(CheckResult(
            "No .git directory",
            not has_git,
            "Found .git directory in archive" if has_git else "",
        ))

        # 11. No .env files
        env_files = [n for n in names if n.endswith(".env") or "/.env" in n or n == ".env"]
        checks.append(CheckResult(
            "No .env files",
            not env_files,
            f"Found: {', '.join(env_files)}" if env_files else "",
        ))

        # 12. No API keys / secrets
        secrets_found = []
        if has_metadata:
            meta_entry = next((n for n in names if n.endswith("metadata.json")), None)
            if meta_entry:
                try:
                    content = zf.read(meta_entry).decode("utf-8")
                    for pat in _SECRET_PATTERNS:
                        if pat.search(content):
                            secrets_found.append(f"Potential secret in metadata.json")
                            break
                except Exception:
                    pass
        checks.append(CheckResult(
            "No API keys or secrets",
            not secrets_found,
            "; ".join(secrets_found) if secrets_found else "",
        ))

        # 13. No original training dataset
        dataset_files = [
            n for n in names
            if any(n.endswith(ext) for ext in _DATASET_EXTENSIONS)
            and "evaluation" not in n.lower()
        ]
        # Also check for a data/ directory
        data_dirs = [n for n in names if n.startswith("data/") or "/data/" in n]
        has_dataset = bool(dataset_files) or bool(data_dirs)
        checks.append(CheckResult(
            "No training dataset included",
            not has_dataset,
            f"Found dataset files: {', '.join((dataset_files + data_dirs)[:5])}" if has_dataset else "",
        ))

        # 14. No unrelated models (only the expected model/ directory)
        model_dirs = set()
        for n in names:
            parts = n.replace(top_prefix, "").split("/")
            if len(parts) >= 2 and parts[0] not in ("model", "tokenizer", "") and \
               any(f.endswith((".bin", ".safetensors", ".pt", ".pth")) for f in [parts[-1]]):
                model_dirs.add(parts[0])
        checks.append(CheckResult(
            "No unrelated models",
            not model_dirs,
            f"Unexpected model directories: {', '.join(model_dirs)}" if model_dirs else "",
        ))

        # 15. No temporary files
        temp_files = [
            n for n in names
            if any(t in n for t in _TEMP_PATTERNS)
        ]
        checks.append(CheckResult(
            "No temporary files",
            not temp_files,
            f"Found: {', '.join(temp_files[:5])}" if temp_files else "",
        ))

        # 16. No absolute filesystem paths in metadata
        abs_paths_found = []
        if has_metadata:
            meta_entry = next((n for n in names if n.endswith("metadata.json")), None)
            if meta_entry:
                try:
                    content = zf.read(meta_entry).decode("utf-8")
                    for pat in _ABSOLUTE_PATH_PATTERNS:
                        match = pat.search(content)
                        if match:
                            abs_paths_found.append(match.group())
                except Exception:
                    pass
        # Absolute paths are a warning, not a hard failure, since
        # adapter_path in metadata may contain the original training path
        if abs_paths_found:
            warnings.append(
                f"Absolute paths detected in metadata: {', '.join(abs_paths_found[:3])}"
            )
        checks.append(CheckResult(
            "No embedded absolute paths",
            not abs_paths_found,
            f"Found: {', '.join(abs_paths_found[:3])}" if abs_paths_found else "",
        ))

        # 17. No path traversal entries
        traversal = [n for n in names if n.startswith("..") or "/../" in n or n.startswith("/")]
        checks.append(CheckResult(
            "No path traversal entries",
            not traversal,
            f"Dangerous entries: {', '.join(traversal[:5])}" if traversal else "",
        ))

    passed = all(c.passed for c in checks)
    return ValidationResult(passed=passed, checks=checks, warnings=warnings)
