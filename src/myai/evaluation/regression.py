import re

REGRESSION_PROMPTS = [
    "What is 2 + 2?",
    "Name three primary colors.",
    "What is the capital of France?",
    "What comes after Tuesday?",
    "How many days are in a week?",
    "Name a fruit that is yellow.",
    "What is 10 minus 3?",
    "What do you call a baby dog?",
]

def _quality(text: str) -> float:
    text = (text or "").strip()
    if not text:
        return 0.0
    if any(r in text.lower() for r in ["i cannot", "i can't", "as an ai", "[error"]):
        return 0.3
    return min(1.0, len(re.findall(r"\w+", text)) / 15)

def regression_score(base_runner, ft_runner) -> tuple:
    base = [_quality(base_runner(p)) for p in REGRESSION_PROMPTS]
    ft = [_quality(ft_runner(p)) for p in REGRESSION_PROMPTS]
    b_avg = sum(base) / len(base) if base else 0.0
    f_avg = sum(ft) / len(ft) if ft else 0.0
    delta = max(0.0, (b_avg - f_avg) / b_avg) if b_avg > 0 else 0.0
    return 1.0 - delta, delta