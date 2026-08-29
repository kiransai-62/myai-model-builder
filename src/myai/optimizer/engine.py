"""MYAI Optimizer — Autonomous Retrain/Compare Loop (Report §5.1, §8, §11, §18).

Diagnoses the weakest goal-weighted metrics of the current release candidate,
prescribes a minimal strategy mutation, retrains via an injected train_fn,
and promotes the new run ONLY if:
  1. it passes the regression gate (stable), AND
  2. composite Δ ≥ min_delta  ("Improvement Justified" gate).

Over-automation mitigation (§18): bounded iterations, explicit reasoning on
every decision, --dry-run preview, and CLI knobs (--min-delta, --max-iters).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..models.leaderboard import Leaderboard, RunRecord, RankedRun

MIN_LOST_POINTS = 0.03   # ignore metric gaps contributing < 3 composite points

# param, op, val, min, max, note
PRESCRIPTIONS: Dict[str, List[dict]] = {
    "bleu": [
        {"param": "learning_rate", "op": "mul", "val": 0.5, "min": 5e-5, "max": 3e-4,
         "note": "lower LR for more precise phrasing"},
        {"param": "seq_len", "op": "mul", "val": 2, "min": 512, "max": 8192,
         "note": "longer context for n-gram overlap"},
    ],
    "rouge": [
        {"param": "seq_len", "op": "mul", "val": 2, "min": 512, "max": 8192,
         "note": "longer context to capture fuller responses"},
        {"param": "lora_rank", "op": "mul", "val": 2, "min": 8, "max": 64,
         "note": "more adapter capacity for content coverage"},
    ],
    "domain_accuracy": [
        {"param": "lora_rank", "op": "mul", "val": 2, "min": 8, "max": 64,
         "note": "raise LoRA rank for domain capacity"},
        {"param": "epochs", "op": "add", "val": 1, "min": 1, "max": 4,
         "note": "add one epoch for domain coverage"},
    ],
    "readability": [
        {"param": "epochs", "op": "add", "val": -1, "min": 1, "max": 4,
         "note": "reduce overfitting that hurts fluency"},
    ],
    "exact_match": [
        {"param": "learning_rate", "op": "mul", "val": 0.5, "min": 5e-5, "max": 3e-4,
         "note": "conservative LR for exactness"},
    ],
}


@dataclass
class Diagnosis:
    metric: str
    value: float
    weight: float
    lost_points: float                 # weight * (1 - value), 0..1 scale


@dataclass
class OptimizationStep:
    iteration: int
    mutation: Dict[str, float]
    before: float
    after: Optional[float]
    delta: float
    promoted: bool
    reasoning: List[str] = field(default_factory=list)


@dataclass
class OptimizationReport:
    steps: List[OptimizationStep]
    final_run_id: str
    improved: bool
    confidence: float = 0.8
    assumptions: List[str] = field(default_factory=lambda: [
        "composite weights from project GoalProfile",
        "promotion requires Δ ≥ min_delta AND regression stability",
    ])


class OptimizerEngine:
    def __init__(self, goal, board: Leaderboard,
                 train_fn: Callable[[Dict], RunRecord],
                 min_delta: float = 2.0, max_iters: int = 3):
        self.goal, self.board, self.train_fn = goal, board, train_fn
        self.min_delta, self.max_iters = min_delta, max_iters

    # ---------------------------------------------- intelligence
    def diagnose(self, rr: RankedRun) -> List[Diagnosis]:
        lost = [Diagnosis(m, rr.run.metrics.get(m, 0.0), w, w * (1 - rr.run.metrics.get(m, 0.0)))
                for m, w in self.goal.eval_weights.items() if m in PRESCRIPTIONS]
        lost = [d for d in lost if d.lost_points >= MIN_LOST_POINTS]
        return sorted(lost, key=lambda d: d.lost_points, reverse=True)[:2]

    def prescribe(self, diags: List[Diagnosis], strategy: Dict) -> Tuple[Dict, List[str]]:
        mutation: Dict[str, float] = {}
        reasoning: List[str] = []
        for d in diags:
            for p in PRESCRIPTIONS[d.metric]:
                cur = strategy.get(p["param"])
                if cur is None or p["param"] in mutation:
                    continue
                raw = cur * p["val"] if p["op"] == "mul" else cur + p["val"]
                new = min(p["max"], max(p["min"], raw))
                if new == cur:
                    continue
                mutation[p["param"]] = new
                reasoning.append(
                    f"{d.metric} lost {d.lost_points * 100:.1f} pts (weight {d.weight:.2f}) "
                    f"→ {p['note']} ({p['param']}: {cur} → {new})")
                break
        return mutation, reasoning

    # ---------------------------------------------- the loop
    def run(self, dry_run: bool = False) -> OptimizationReport:
        steps: List[OptimizationStep] = []
        ranked = self.board.rank()
        if not ranked:
            return OptimizationReport([], "", False)
        base = self.board.release_candidate() or ranked[0]
        improved = False

        for i in range(1, self.max_iters + 1):
            diags = self.diagnose(base)
            if not diags:
                steps.append(OptimizationStep(i, {}, base.composite, None, 0.0, False,
                             ["No significant metric gaps remain → converged."]))
                break
            mutation, reasoning = self.prescribe(diags, dict(base.run.strategy))
            if not mutation:
                steps.append(OptimizationStep(i, {}, base.composite, None, 0.0, False,
                             ["All prescriptions exhausted at their safe bounds."]))
                break
            if dry_run:
                steps.append(OptimizationStep(i, mutation, base.composite, None, 0.0, False,
                             reasoning + ["[dry-run] mutation previewed, no training executed."]))
                continue

            strategy = dict(base.run.strategy)
            strategy.update(mutation)
            new_run = self.train_fn(strategy)
            self.board.add_run(new_run)
            new_rr = self.board.score(new_run)
            _, delta = self.board.compare(base.run.run_id, new_run.run_id)
            promoted = (new_rr.composite > base.composite) and delta >= self.min_delta and new_rr.stable

            r = list(reasoning)
            if promoted:
                r.append(f"Δ +{delta:.1f} ≥ {self.min_delta} and regression-stable → promoted {new_run.run_id}.")
                base, improved = new_rr, True
            else:
                r.append(f"Δ {delta:+.1f} below justification threshold ({self.min_delta}) "
                         f"or unstable → kept {base.run.run_id}.")
            steps.append(OptimizationStep(i, mutation, steps_before(base, new_rr), new_rr.composite,
                                          delta, promoted, r))

        return OptimizationReport(steps, base.run.run_id, improved)


def steps_before(base, new_rr) -> float:  # small helper for readability
    return new_rr.composite - (new_rr.composite - base.composite) - (base.composite - base.composite) \
        if False else base.composite


def print_report(rep: OptimizationReport) -> None:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    console.print("\n[bold cyan]🔧 MYAI Optimizer Report[/bold cyan]")
    t = Table(header_style="bold magenta")
    t.add_column("Iter")
    t.add_column("Mutation")
    t.add_column("Δ")
    t.add_column("Result")
    for s in rep.steps:
        mut = ", ".join(f"{k}={v}" for k, v in s.mutation.items()) or "—"
        d = f"{s.delta:+.1f}" if s.after is not None else "—"
        res = "✅ promoted" if s.promoted else ("🚫 not justified" if s.after is not None else "⏸️")
        t.add_row(str(s.iteration), mut, d, res)
    console.print(t)
    for s in rep.steps:
        for line in s.reasoning:
            console.print(f"   ✨ {line}")
    verdict = f"[green]improved → release candidate is now {rep.final_run_id}[/green]" \
        if rep.improved else f"[yellow]no justified improvement → kept {rep.final_run_id}[/yellow]"
    console.print(f"\n🏁 Verdict: {verdict} (confidence {rep.confidence:.0%})")
    console.print(f"[dim]   assumptions: {', '.join(rep.assumptions)}[/dim]")
    console.print("[dim]💡 Knobs: myai optimize --max-iters 5 --min-delta 1.0 --dry-run[/dim]\n")
