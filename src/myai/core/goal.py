"""MYAI Goal & Project Analyzer (Report §5.1, §10).

Captures the user's intent during `myai init` to drive downstream decisions:
- Model Recommendation (Task/Domain fit)
- Evaluation Weighting (Metric prioritization)
- Context Length & Latency requirements
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


class TaskType(str, Enum):
    INSTRUCTION = "instruction-tuning"
    CHAT = "chat"
    DOMAIN_QA = "domain-qa"
    CODE = "code"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"

    @classmethod
    def from_str(cls, val: str) -> "TaskType":
        val_lower = val.lower().strip()
        for member in cls:
            if member.value.lower() == val_lower or member.name.lower() == val_lower:
                return member
        return cls.INSTRUCTION


class Domain(str, Enum):
    GENERAL = "general"
    MEDICAL = "medical"
    FINANCE = "finance"
    FITNESS = "fitness"
    LEGAL = "legal"
    CUSTOMER_SUPPORT = "customer-support"

    @classmethod
    def from_str(cls, val: str) -> "Domain":
        val_lower = val.lower().strip().replace(" ", "-")
        for member in cls:
            if member.value.lower() == val_lower or member.name.lower() == val_lower:
                return member
        return cls.GENERAL


@dataclass
class GoalProfile:
    task: TaskType = TaskType.INSTRUCTION
    domain: Domain = Domain.GENERAL
    context_priority: str = "balanced"        # short, balanced, long-context
    latency_priority: str = "balanced"        # fast, balanced, high-quality
    target_deployment: str = "local-cpu-gpu"   # edge, local-cpu-gpu, server

    # Derived weights for the Evaluation Engine (computed dynamically)
    eval_weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.task, str) and not isinstance(self.task, TaskType):
            self.task = TaskType.from_str(self.task)
        if isinstance(self.domain, str) and not isinstance(self.domain, Domain):
            self.domain = Domain.from_str(self.domain)
        self.compute_eval_weights()

    def compute_eval_weights(self) -> None:
        """Dynamically assigns weights to evaluation metrics based on the goal."""
        weights = {
            "rouge": 0.2,
            "bleu": 0.2,
            "readability": 0.2,
            "domain_accuracy": 0.2,
            "exact_match": 0.2,
        }

        # Task-specific adjustments
        if self.task == TaskType.CODE:
            weights["exact_match"] = 0.6
            weights["readability"] = 0.1
            weights["rouge"] = 0.05
            weights["bleu"] = 0.05
            weights["domain_accuracy"] = 0.2
        elif self.task in (TaskType.CHAT, TaskType.DOMAIN_QA):
            weights["readability"] = 0.4
            weights["domain_accuracy"] = 0.4
            weights["exact_match"] = 0.05
            weights["rouge"] = 0.075
            weights["bleu"] = 0.075
        elif self.task == TaskType.SUMMARIZATION:
            weights["rouge"] = 0.6
            weights["bleu"] = 0.2
            weights["readability"] = 0.1
            weights["domain_accuracy"] = 0.05
            weights["exact_match"] = 0.05
        elif self.task == TaskType.CLASSIFICATION:
            weights["exact_match"] = 0.5
            weights["domain_accuracy"] = 0.3
            weights["readability"] = 0.1
            weights["rouge"] = 0.05
            weights["bleu"] = 0.05

        # Domain-specific adjustments
        if self.domain != Domain.GENERAL:
            weights["domain_accuracy"] = max(weights["domain_accuracy"], 0.4)

        # Normalize to 1.0
        total = sum(weights.values())
        self.eval_weights = {k: round(v / total, 2) for k, v in weights.items()}

    def to_dict(self, include_weights: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for YAML persistence."""
        self.compute_eval_weights()
        d: Dict[str, Any] = {
            "task": self.task.value,
            "domain": self.domain.value,
            "context_priority": self.context_priority,
            "latency_priority": self.latency_priority,
            "target_deployment": self.target_deployment,
        }
        if include_weights:
            d["eval_weights"] = self.eval_weights
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoalProfile":
        """Builds GoalProfile from dictionary."""
        task_str = data.get("task", "instruction-tuning")
        domain_str = data.get("domain", "general")
        profile = cls(
            task=TaskType.from_str(task_str),
            domain=Domain.from_str(domain_str),
            context_priority=data.get("context_priority", "balanced"),
            latency_priority=data.get("latency_priority", "balanced"),
            target_deployment=data.get("target_deployment", "local-cpu-gpu"),
        )
        profile.compute_eval_weights()
        return profile

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "GoalProfile":
        """Loads the Goal Profile from the project's myai.yaml."""
        if not yaml_path.exists():
            return cls()
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            goal_data = data.get("goal", {})
            return cls.from_dict(goal_data)
        except Exception:
            return cls()


def prompt_for_goal(
    non_interactive: bool = False,
    task: Optional[str] = None,
    domain: Optional[str] = None,
    context_priority: Optional[str] = None,
    latency_priority: Optional[str] = None,
    target_deployment: Optional[str] = None,
) -> GoalProfile:
    """Interactive CLI prompt using Rich to capture the goal during `myai init`, with non-interactive fallback."""
    import sys
    is_interactive = False
    try:
        is_interactive = sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        is_interactive = False

    if non_interactive or (task and domain) or not is_interactive:
        selected_task = TaskType.from_str(task) if task else TaskType.INSTRUCTION
        selected_domain = Domain.from_str(domain) if domain else Domain.GENERAL
        profile = GoalProfile(
            task=selected_task,
            domain=selected_domain,
            context_priority=context_priority or "balanced",
            latency_priority=latency_priority or "balanced",
            target_deployment=target_deployment or "local-cpu-gpu",
        )
        profile.compute_eval_weights()
        return profile

    try:
        from rich.console import Console
        from rich.prompt import Prompt, IntPrompt

        console = Console()
        console.print("\n[bold cyan]🎯 Goal Understanding[/bold cyan]")
        console.print("Tell MYAI what you want to build so it can optimize model selection and evaluation.\n")

        # 1. Task Type
        console.print("[bold]1. What is the primary task of this AI?[/bold]")
        for i, t in enumerate(TaskType, 1):
            console.print(f"   [cyan]{i}[/cyan]. {t.value}")
        task_idx = IntPrompt.ask("   Select task", default=1, choices=[str(i) for i in range(1, len(TaskType) + 1)])
        chosen_task = list(TaskType)[task_idx - 1]

        # 2. Domain
        console.print("\n[bold]2. What domain will this AI operate in?[/bold]")
        for i, dom in enumerate(Domain, 1):
            console.print(f"   [cyan]{i}[/cyan]. {dom.value}")
        dom_idx = IntPrompt.ask("   Select domain", default=1, choices=[str(i) for i in range(1, len(Domain) + 1)])
        chosen_domain = list(Domain)[dom_idx - 1]

        # 3. Priorities
        console.print("\n[bold]3. Architectural Priorities[/bold]")
        context = Prompt.ask("   Context length priority", choices=["short", "balanced", "long-context"], default="balanced")
        latency = Prompt.ask("   Latency vs Quality priority", choices=["fast", "balanced", "high-quality"], default="balanced")
        deployment = Prompt.ask("   Target deployment", choices=["edge", "local-cpu-gpu", "server"], default="local-cpu-gpu")

        profile = GoalProfile(
            task=chosen_task,
            domain=chosen_domain,
            context_priority=context,
            latency_priority=latency,
            target_deployment=deployment,
        )
        profile.compute_eval_weights()

        console.print("\n[green]✓ Goal profile captured. Evaluation weights auto-calculated.[/green]")
        return profile
    except (EOFError, KeyboardInterrupt, Exception):
        selected_task = TaskType.from_str(task) if task else TaskType.INSTRUCTION
        selected_domain = Domain.from_str(domain) if domain else Domain.GENERAL
        profile = GoalProfile(
            task=selected_task,
            domain=selected_domain,
            context_priority=context_priority or "balanced",
            latency_priority=latency_priority or "balanced",
            target_deployment=target_deployment or "local-cpu-gpu",
        )
        profile.compute_eval_weights()
        return profile
