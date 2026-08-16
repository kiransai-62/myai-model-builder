"""Simple English: readability score + jargon detection."""
from .report import MetricResult

JARGON_WORDS = {
    "utilize", "leverage", "synergy", "paradigm", "holistic",
    "proactive", "scalable", "ecosystem", "optimize", "robust"
}

def _flesch_reading_ease(text: str) -> float:
    """Flesch Reading Ease — higher = easier to read."""
    try:
        import textstat  # type: ignore[import-not-found]
        return textstat.flesch_reading_ease(text)
    except ImportError:
        # Minimal fallback
        sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
        words = len(text.split())
        if words == 0:
            return 0.0
        return max(0, min(100, 100 - (words / sentences) * 2))

def _jargon_ratio(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    jargon_count = sum(1 for w in words if w.strip(".,!?;:") in JARGON_WORDS)
    return jargon_count / len(words)

def evaluate_readability(answer: str, eval_cases: list[dict]) -> MetricResult:
    if not answer.strip():
        return MetricResult("simple_english", 0.0, 0.6, False, "Empty answer")
    
    fre = _flesch_reading_ease(answer)
    jargon = _jargon_ratio(answer)
    
    # Normalize: FRE 0-100, target 60+
    fre_score = max(0.0, min(1.0, fre / 100))
    jargon_penalty = min(1.0, jargon * 10)  # Heavy penalty for jargon
    
    score = max(0.0, fre_score - jargon_penalty)
    threshold = 0.6
    
    return MetricResult(
        name="simple_english",
        score=score,
        threshold=threshold,
        passed=score >= threshold,
        detail=f"Flesch Reading Ease: {fre:.0f}, jargon ratio: {jargon:.3f}"
    )