"""Domain accuracy: check if required facts appear in the generated answer."""
from .report import MetricResult

def evaluate_domain(answer: str, eval_cases: list[dict]) -> MetricResult:
    if not eval_cases:
        return MetricResult(
            name="domain_accuracy",
            score=1.0,
            threshold=0.8,
            passed=True,
            detail="No evaluation cases provided"
        )
    
    total_facts = 0
    found_facts = 0
    
    for case in eval_cases:
        required = case.get("required_facts", [])
        if not required:
            continue
        answer_lower = answer.lower()
        for fact in required:
            total_facts += 1
            if fact.lower() in answer_lower:
                found_facts += 1
    
    score = found_facts / total_facts if total_facts > 0 else 1.0
    threshold = 0.8
    
    return MetricResult(
        name="domain_accuracy",
        score=score,
        threshold=threshold,
        passed=score >= threshold,
        detail=f"{found_facts}/{total_facts} required facts present"
    )