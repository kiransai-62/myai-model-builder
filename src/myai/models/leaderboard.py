"""MYAI Experiment Leaderboard — Best-Model Ranking Matrix (Report §12, §19).

Ranks historical training runs into a composite, goal-weighted score and
designates the release candidate for `myai export`.

Design rules (Report §18 / §19):
- Best model is chosen on measurable evidence, not recency.
- Regressed runs are penalized and can NEVER become release candidates.
- Weights come from the project GoalProfile, so "best" is goal-relative.
- `myai export --run <id>` remains the explicit override.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core.goal import GoalProfile

METRIC_KEYS = ["rouge", "bleu", "readability", "domain_accuracy", "exact_match"]


@dataclass
class RunRecord:
    run_id: str
    model_name: str
    timestamp: str
    strategy: Dict                        # rank, lr, quantization, epochs...
    metrics: Dict[str, float]             # normalized 0..1
    regression_passed: bool
    vram_peak_gb: float = 0.0
    train_minutes: float = 0.0
    dataset_hash: str = ""


@dataclass
class RankedRun:
    run: RunRecord
    composite: float                      # 0..100
    stable: bool
    breakdown: Dict[str, float]           # weighted contribution per metric


class Leaderboard:
    """Goal-weighted experiment leaderboard.

    Scores runs against the project's GoalProfile eval_weights, penalizes
    regressed models, and designates the top stable run as the release
    candidate for ``myai export``.
    """

    def __init__(self, goal: GoalProfile, runs_dir: Optional[Path] = None):
        self.goal = goal
        if not self.goal.eval_weights:
            self.goal.compute_eval_weights()
        self.runs_dir = runs_dir
        self.runs: List[RunRecord] = []
        if runs_dir and runs_dir.exists():
            self.load()

    # ------------------------------------------------ persistence
    def load(self) -> None:
        """Load all persisted RunRecord JSON files from the runs directory."""
        if not self.runs_dir or not self.runs_dir.exists():
            return
        for p in sorted(self.runs_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                self.runs.append(RunRecord(**data))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

    def add_run(self, run: RunRecord) -> None:
        """Register a run and persist to disk if a runs_dir is configured."""
        self.runs.append(run)
        if self.runs_dir:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            (self.runs_dir / f"{run.run_id}.json").write_text(
                json.dumps(asdict(run), indent=2), encoding="utf-8"
            )

    # ------------------------------------------------ scoring
    @staticmethod
    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    def score(self, run: RunRecord) -> RankedRun:
        """Compute a goal-weighted composite score for a single run."""
        w = self.goal.eval_weights
        breakdown: Dict[str, float] = {}
        for k in METRIC_KEYS:
            weight = w.get(k, 0.0)
            metric = self._clamp(run.metrics.get(k, 0.0))
            breakdown[k] = round(weight * metric, 4)

        composite = sum(breakdown.values())

        # §12: regression gate — regressed models are penalized and blocked
        if not run.regression_passed:
            composite *= 0.5

        return RankedRun(
            run=run,
            composite=round(composite * 100, 1),
            stable=run.regression_passed,
            breakdown=breakdown,
        )

    def rank(self) -> List[RankedRun]:
        """Return all runs sorted by (stable, composite) descending."""
        return sorted(
            (self.score(r) for r in self.runs),
            key=lambda rr: (rr.stable, rr.composite),
            reverse=True,
        )

    def release_candidate(self) -> Optional[RankedRun]:
        """Return the highest-scoring stable run, or None if no stable run exists."""
        stable = [rr for rr in self.rank() if rr.stable]
        return stable[0] if stable else None

    def compare(self, id_a: str, id_b: str) -> Tuple[RankedRun, float]:
        """A/B comparison helper for the Stage-E optimizer loop.

        Returns ``(winner, delta)`` where delta is the absolute composite
        score difference. The optimizer should only promote a retrain if
        ``delta >= improvement_threshold`` (e.g., +2.0 points).
        """
        scored = {r.run_id: self.score(r) for r in self.runs}
        a = scored[id_a]
        b = scored[id_b]
        winner = a if a.composite >= b.composite else b
        return winner, abs(a.composite - b.composite)

    def explain(self, rr: RankedRun) -> List[str]:
        """Human-readable explanation of what drove this run's score."""
        top = sorted(rr.breakdown.items(), key=lambda kv: kv[1], reverse=True)[:2]
        lines = [f"{k} contributed {v * 100:.1f} pts" for k, v in top]
        if not rr.stable:
            lines.append("regression gate FAILED → score halved & blocked from release")
        return lines

    # ------------------------------------------------ terminal UI
    def print_board(self) -> None:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        ranked = self.rank()
        rc = self.release_candidate()

        console.print(
            f"\n🏆 [bold cyan]MYAI Leaderboard[/bold cyan] "
            f"(goal weights: {self.goal.task.value} / {self.goal.domain.value})"
        )

        t = Table(header_style="bold magenta")
        t.add_column("#")
        t.add_column("Run")
        t.add_column("Model")
        t.add_column("Composite", justify="right")
        t.add_column("Stable")
        t.add_column("Key config")

        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        for i, rr in enumerate(ranked, 1):
            s = rr.run.strategy
            medal = medals.get(i, "  ")
            cfg_str = (
                f"r{s.get('lora_rank', '?')} "
                f"lr{s.get('learning_rate', '?')} "
                f"{s.get('quantization', '?')}"
            )
            t.add_row(
                f"{medal}{i}",
                rr.run.run_id,
                rr.run.model_name,
                f"{rr.composite:.1f}",
                "✅" if rr.stable else "❌",
                cfg_str,
            )
        console.print(t)

        if rc:
            console.print(
                f"\n[bold green]📦 Release candidate:[/bold green] "
                f"{rc.run.run_id} ({rc.composite:.1f}/100) → default for `myai export`"
            )
            for line in self.explain(rc):
                console.print(f"   ✨ {line}")
        else:
            console.print("\n[bold red]⚠️ No stable run available for release.[/bold red]")

        console.print("[dim]💡 Override: myai export --run <id>[/dim]\n")
