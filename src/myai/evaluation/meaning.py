"""Meaning preservation: check entailment + prohibition of false claims."""
import re
from .report import MetricResult

def _simple_entails(answer: str, expected: str) -> float:
    """Lightweight entailment proxy: token overlap between expected meaning and answer."""
    if not expected:
        return 1.0
    
    def tokens(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower()))
    
    a_tokens = tokens(answer)
    e_tokens = tokens(expected)
    if not e_tokens:
        return 1.0
    
    overlap = len(a_tokens & e_tokens) / len(e_tokens)
    return min(1.0, overlap * 1.5)  # Scale up since overlap underestimates entailment

def _check_prohibitions(answer: str, must_not_claim: list[str]) -> float:
    """Return 1.0 if no prohibited claims appear, else a penalty."""
    if not must_not_claim:
        return 1.0
    
    answer_lower = answer.lower()
    violations = sum(1 for claim in must_not_claim if claim.lower() in answer_lower)
    
    if violations == 0:
        return 1.0
    return max(0.0, 1.0 - (violations / len(must_not_claim)))

def evaluate_meaning(answer: str, eval_cases: list[dict]) -> MetricResult:
    if not eval_cases:
        return MetricResult("meaning_preservation", 1.0, 0.8, True, "No evaluation cases")
    
    scores = []
    for case in eval_cases:
        entailment = _simple_entails(answer, case.get("expected_meaning", ""))
        prohibition = _check_prohibitions(answer, case.get("must_not_claim", []))
        scores.append((entailment + prohibition) / 2)
    
    score = sum(scores) / len(scores)
    threshold = 0.8
    
    return MetricResult(
        name="meaning_preservation",
        score=score,
        threshold=threshold,
        passed=score >= threshold,
        detail=f"Average entailment/prohibition score across {len(scores)} cases"
    )