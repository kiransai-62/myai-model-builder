import json
import random
import re
from pathlib import Path
from ..data.loader import load_file

def split_pairs(pairs, eval_fraction: float = 0.1, seed: int = 42):
    """90/10 by default; configurable. Deterministic via seed."""
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n = int(len(pairs) * eval_fraction)
    if len(pairs) < 4 or n < 1:
        return pairs, []
    return pairs[n:], pairs[:n]

def write_holdout(path: Path, pairs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8")

def read_holdout(path: Path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

FACT_RE = re.compile(r"\$[\d,.]+|\b\d+(?:\.\d+)?\s?(?:days|hours|minutes|weeks|months|GB|%)?\b|[\w.+-]+@[\w-]+\.[\w.]+")

def facts_from_text(text: str, max_facts: int = 4) -> list:
    """Extract testable facts: numbers, currency, emails, durations, then key words."""
    facts = list(dict.fromkeys(FACT_RE.findall(text)))
    words = sorted(re.findall(r"[A-Za-z][A-Za-z-]{4,}", text), key=len, reverse=True)
    facts += [w for w in words if w.lower() not in {f.lower() for f in facts}]
    return facts[:max_facts]

def load_eval_cases(source: Path) -> list:
    """Explicit knowledge cases from <source>/evaluation/ (required_facts format)."""
    cases = []
    p = Path(source)
    eval_dir = (p.parent if p.is_file() else p) / "evaluation"
    if eval_dir.exists() and eval_dir.is_dir():
        for f in eval_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in (".jsonl", ".json"):
                cases.extend(load_file(f))
    return [c for c in cases if c.get("prompt")]

def knowledge_cases_from_holdout(holdout: list) -> list:
    """Fallback: derive knowledge tests from held-out examples (never training data)."""
    return [{
        "prompt": p["prompt"],
        "required_facts": facts_from_text(p["response"]),
        "must_not_claim": [],
        "expected_meaning": p["response"],
    } for p in holdout]
