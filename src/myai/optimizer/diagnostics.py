"""MYAI Optimization Diagnostics (Report §13, §19).

Analyzes evaluation metrics, regression test outcomes, and goal weights
to diagnose training bottlenecks (catastrophic forgetting, underfitting,
overfitting, goal misalignment) and recommend precise hyperparameter adjustments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from ..core.goal import GoalProfile, TaskType
from ..models.leaderboard import RunRecord, RankedRun


class IssueCategory(str, Enum):
    CATASTROPHIC_FORGETTING = "catastrophic_forgetting"
    UNDERFITTING = "underfitting"
    OVERFITTING = "overfitting"
    GOAL_MISALIGNMENT = "goal_misalignment"
    CONVERGED = "converged"


@dataclass
class DiagnosticReport:
    category: IssueCategory
    summary: str
    weakest_metric: str
    weakest_score: float
    recommendations: List[str] = field(default_factory=list)
    suggested_params: Dict[str, Any] = field(default_factory=dict)


def diagnose_run(
    run: RunRecord,
    goal: GoalProfile,
    target_score: float = 85.0,
) -> DiagnosticReport:
    """Analyze a run's evaluation metrics and regression status to diagnose issues.

    Args:
        run: The RunRecord containing metrics and strategy
        goal: The project's GoalProfile
        target_score: Desired composite score (0..100)
    """
    if not goal.eval_weights:
        goal.compute_eval_weights()

    # 1. Catastrophic forgetting / Regression failure
    if not run.regression_passed or run.metrics.get("regression", 1.0) < 0.70:
        curr_lr = float(run.strategy.get("learning_rate", 2e-4))
        curr_epochs = int(run.strategy.get("epochs", 3))
        return DiagnosticReport(
            category=IssueCategory.CATASTROPHIC_FORGETTING,
            summary="Regression gate failed: model degraded on general reasoning benchmarks.",
            weakest_metric="regression",
            weakest_score=run.metrics.get("regression", 0.0),
            recommendations=[
                "Halve learning rate to protect base model capabilities.",
                "Reduce training epochs to prevent catastrophic forgetting.",
                "Maintain conservative LoRA rank.",
            ],
            suggested_params={
                "learning_rate": max(1e-5, curr_lr * 0.5),
                "epochs": max(1, curr_epochs - 1),
            },
        )

    # Calculate weighted contributions
    weighted_scores = {}
    for k, weight in goal.eval_weights.items():
        val = run.metrics.get(k, 0.0)
        weighted_scores[k] = (weight, val)

    # Find the metric with the highest room for goal-weighted improvement
    def penalty_or_gap(item):
        k, (w, val) = item
        return w * (1.0 - val)

    weakest_k, (weakest_w, weakest_val) = max(weighted_scores.items(), key=penalty_or_gap)

    # Compute raw composite
    raw_composite = sum(w * run.metrics.get(k, 0.0) for k, w in goal.eval_weights.items()) * 100.0

    # 2. Converged / Target Met
    if raw_composite >= target_score:
        return DiagnosticReport(
            category=IssueCategory.CONVERGED,
            summary=f"Run achieved target composite score ({raw_composite:.1f} >= {target_score:.1f}).",
            weakest_metric=weakest_k,
            weakest_score=weakest_val,
            recommendations=["Model meets or exceeds quality goals. Ready for export."],
            suggested_params={},
        )

    # 3. Underfitting (low task/knowledge/exact_match while regression passed)
    curr_lr = float(run.strategy.get("learning_rate", 2e-4))
    curr_epochs = int(run.strategy.get("epochs", 3))
    curr_rank = int(run.strategy.get("lora_rank", 16))

    if weakest_val < 0.60:
        return DiagnosticReport(
            category=IssueCategory.UNDERFITTING,
            summary=f"Model underfitting on goal metric '{weakest_k}' ({weakest_val:.2f}).",
            weakest_metric=weakest_k,
            weakest_score=weakest_val,
            recommendations=[
                f"Increase epochs (+1) to allow more gradient updates on {weakest_k}.",
                "Double LoRA rank for higher adapter capacity.",
                "Slightly raise learning rate within stability bounds.",
            ],
            suggested_params={
                "epochs": min(8, curr_epochs + 1),
                "lora_rank": min(64, curr_rank * 2),
                "learning_rate": min(5e-4, curr_lr * 1.25),
            },
        )

    # 4. Fine-tuning goal misalignment / plateau
    return DiagnosticReport(
        category=IssueCategory.GOAL_MISALIGNMENT,
        summary=f"Goal bottleneck is '{weakest_k}' (score {weakest_val:.2f}, weight {weakest_w:.2f}).",
        weakest_metric=weakest_k,
        weakest_score=weakest_val,
        recommendations=[
            f"Tune hyperparameters to maximize '{weakest_k}'.",
            "Adjust learning rate and batch accumulation for smoother convergence.",
        ],
        suggested_params={
            "learning_rate": curr_lr * 0.8,
            "epochs": min(8, curr_epochs + 1),
        },
    )
