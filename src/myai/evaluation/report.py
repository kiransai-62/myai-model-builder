from dataclasses import dataclass, field
from ..core.console import console

@dataclass
class MetricResult:
    name: str
    score: float          # 0.0 - 1.0
    threshold: float
    passed: bool
    detail: str = ""

@dataclass
class EvaluationReport:
    eval_id: str = "eval_init"
    model_id: str = ""
    dataset_id: str = ""
    run_id: str = ""
    validation: list = field(default_factory=list)
    knowledge: dict = field(default_factory=dict)   # {passed,total,score}
    task: dict = field(default_factory=dict)
    regression: dict = field(default_factory=dict)  # {score,delta,passed}
    quality: float = 0.0
    overall: float = 0.0
    status: str = "FAILED"
    reason: str = ""
    recommended_action: str = ""

    @property
    def overall_pass(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict:
        return {
            "eval_id": self.eval_id,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "run_id": self.run_id,
            "validation": [c.__dict__ if hasattr(c, "__dict__") else c for c in self.validation],
            "knowledge": self.knowledge,
            "task": self.task,
            "regression": self.regression,
            "quality": round(self.quality, 3),
            "overall": round(self.overall, 3),
            "status": self.status,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
        }

    def print_summary(self):
        console.print("\n[bold cyan]MYAI EVALUATION[/bold cyan]\n")
        console.print(f"Model:\n{self.model_id}\n")
        console.print(f"Dataset:\n{self.dataset_id}\n")
        console.print("─" * 28 + "\n")

        console.print("[bold]Validation[/bold]")
        for c in self.validation:
            passed = c.passed if hasattr(c, "passed") else c.get("passed", False)
            name = c.name if hasattr(c, "name") else c.get("name", "check")
            icon = "[green]✓[/green]" if passed else "[red]✗[/red]"
            console.print(f"{icon} {name}")

        k, t, r = self.knowledge, self.task, self.regression
        ki = "[green]✓[/green]" if k.get("passed") else "[red]✗[/red]"
        ti = "[green]✓[/green]" if t.get("passed") else "[red]✗[/red]"
        ri = "[green]✓[/green]" if r.get("passed") else "[red]✗[/red]"
        console.print(f"\n[bold]Knowledge[/bold]\n{ki} {k.get('passed_cases', 0)} / {k.get('total_cases', 0)}")
        console.print(f"\n[bold]Task Quality[/bold]\n{ti} {int(t.get('score', 0) * 100)} / 100")
        console.print(f"\n[bold]Regression[/bold]\n{ri} {'PASS' if r.get('passed') else 'FAIL'}")

        console.print("\n" + "─" * 28 + "\n")
        console.print(f"[bold]Overall Score: {int(self.overall * 100)}%[/bold]\n")
        if self.status == "PASS":
            console.print("[bold green]Status: PASS[/bold green]\n")
            console.print("Model is ready for registration.")
        else:
            console.print("[bold red]Status: FAILED[/bold red]\n")
            if self.reason:
                console.print(f"Reason:\n{self.reason}\n")
            if self.recommended_action:
                console.print(f"Recommended action:\n{self.recommended_action}")