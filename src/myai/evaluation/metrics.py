import re

def _tok(t: str) -> set:
    return set(re.findall(r"\w+", (t or "").lower()))

def knowledge_case_score(answer: str, case: dict) -> tuple:
    required = case.get("required_facts", [])
    a = (answer or "").lower()
    found = [f for f in required if f.lower() in a]
    banned = case.get("must_not_claim", [])
    violations = [b for b in banned if b.lower() in a]
    score = (len(found) / len(required)) if required else 1.0
    if violations:
        score = max(0.0, score - 0.5 * len(violations))
    return score, found, violations

def task_score(answer: str, expected: str) -> float:
    a, e = _tok(answer), _tok(expected)
    if not a or not e:
        return 0.0
    inter = len(a & e)
    if inter == 0:
        return 0.0
    prec, rec = inter / len(a), inter / len(e)
    f1 = 2 * prec * rec / (prec + rec)
    return min(1.0, f1 * 1.4)          # F1 underestimates paraphrases

def quality_score(answer: str) -> float:
    """Simple-English proxy: sentence length + jargon penalty."""
    words = (answer or "").split()
    if not words:
        return 0.0
    avg_len = len(words) / max(1, answer.count(".") + answer.count("!") + answer.count("?"))
    jargon = sum(1 for w in words if w.lower().strip(".,!?") in
                 {"utilize", "leverage", "synergy", "paradigm", "holistic", "robust", "scalable"})
    return max(0.0, min(1.0, 1.0 - max(0, avg_len - 25) / 50 - jargon * 0.1))

def overall_score(knowledge: float, task: float, regression: float, quality: float) -> float:
    return round(0.4 * knowledge + 0.3 * task + 0.2 * regression + 0.1 * quality, 3)
