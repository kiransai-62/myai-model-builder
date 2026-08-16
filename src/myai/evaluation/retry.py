"""Automatic retry policy based on evaluation results."""
from dataclasses import dataclass
from ..core.config import ProjectConfig
from ..core.console import console, print_warning, print_info

@dataclass
class RetryDecision:
    action: str  # "pass", "retry", "stop"
    changes: dict
    reason: str

def decide_retry(
    report,
    attempt: int,
    max_retries: int,
    current_cfg: ProjectConfig,
) -> RetryDecision:
    """Decide whether to pass, retry with changes, or stop."""
    
    if report.overall_pass:
        return RetryDecision(
            action="pass",
            changes={},
            reason="All evaluation metrics passed."
        )
    
    if attempt >= max_retries:
        return RetryDecision(
            action="stop",
            changes={},
            reason=f"Maximum retries ({max_retries}) reached. Presenting best checkpoint."
        )
    
    # Identify the failing metric and choose a fix
    changes = {}
    reason_parts = []
    
    if not report.regression.passed:
        regression_delta = 1.0 - report.regression.score
        
        if regression_delta <= 0.15:
            # Moderate regression: halve LR, reduce LoRA rank
            new_lr = current_cfg.training.learning_rate / 2
            changes["learning_rate"] = new_lr
            reason_parts.append(f"Moderate regression ({regression_delta*100:.0f}%). Halving LR to {new_lr:.2e}")
        else:
            # High regression: force LoRA if using QLoRA
            if current_cfg.training.method == "qlora":
                changes["method"] = "lora"
                reason_parts.append(f"High regression ({regression_delta*100:.0f}%). Switching QLoRA → LoRA")
            else:
                new_lr = current_cfg.training.learning_rate / 2
                changes["learning_rate"] = new_lr
                reason_parts.append(f"High regression ({regression_delta*100:.0f}%). Halving LR")
    
    if not report.domain.passed:
        # Domain failure: train longer
        changes["epochs"] = min(current_cfg.training.epochs + 1, 10)
        reason_parts.append(f"Low domain accuracy ({report.domain.score*100:.0f}%). Adding epoch.")
    
    if not report.meaning.passed:
        # Meaning failure: reduce LR
        new_lr = changes.get("learning_rate", current_cfg.training.learning_rate) / 2
        changes["learning_rate"] = new_lr
        reason_parts.append(f"Low meaning preservation ({report.meaning.score*100:.0f}%). Reducing LR.")
    
    if not changes:
        # Generic fallback
        new_lr = current_cfg.training.learning_rate / 2
        changes["learning_rate"] = new_lr
        reason_parts.append("Generic retry with halved LR.")
    
    return RetryDecision(
        action="retry",
        changes=changes,
        reason=" | ".join(reason_parts)
    )


def apply_changes(cfg: ProjectConfig, changes: dict) -> ProjectConfig:
    """Apply retry changes to config (returns modified copy)."""
    if "learning_rate" in changes:
        cfg.training.learning_rate = changes["learning_rate"]
    if "method" in changes:
        cfg.training.method = changes["method"]
    if "epochs" in changes:
        cfg.training.epochs = changes["epochs"]
    return cfg