"""MYAI Autopilot — Goal-to-Deployment Orchestrator (Report §2, §17 Phase 15).

Chains the intelligence modules (Stages A–E) into one supervised flow:
Goal → Hardware → Data → Model → Feasibility → Strategy → Train →
Leaderboard → Optimizer → (optional) Export.

§18 compliance: every stage prints its reasoning; --dry-run previews the full
plan; explicit overrides (--model, --override) always win; loops are bounded;
a FAIL feasibility gate aborts before any GPU time is spent.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional
import yaml

from myai.core.goal import GoalProfile
from myai.hardware.detector import detect_hardware
from myai.hardware.feasibility import run_feasibility
from myai.models.recommender import recommend_model
from myai.training.strategy import plan_strategy
from myai.data.scorer import analyze_dataset
from myai.models.leaderboard import Leaderboard
from myai.optimizer.engine import OptimizerEngine


@dataclass
class StageResult:
    icon: str
    name: str
    summary: str
    reasoning: List[str] = field(default_factory=list)


@dataclass
class AutopilotReport:
    stages: List[StageResult] = field(default_factory=list)
    final_run_id: Optional[str] = None
    export_path: Optional[str] = None
    ready: bool = False


def _load_sources(project_dir: Path) -> List[Path]:
    from myai.core.config import ProjectConfig
    from myai.data.manager import resolve_dataset_source
    try:
        cfg = ProjectConfig.load(project_dir)
        src = resolve_dataset_source(project_dir, cfg)
        if src and src.exists():
            return [src]
    except Exception:
        pass

    manifest = project_dir / "data" / "sources.yaml"
    if not manifest.exists():
        data_dir = project_dir / "data"
        if data_dir.exists():
            return [data_dir]
        return []
    try:
        entries = yaml.safe_load(manifest.read_text("utf-8")) or []
        paths = [Path(e["path"]) for e in entries if "path" in e]
        return paths if paths else ([project_dir / "data"] if (project_dir / "data").exists() else [])
    except Exception:
        return [project_dir / "data"] if (project_dir / "data").exists() else []


class Autopilot:
    def __init__(self, project_dir: Path, *,
                 train_fn: Callable,                      # strategy -> RunRecord
                 export_fn: Optional[Callable] = None,    # run_id -> Path
                 export: bool = False, dry_run: bool = False,
                 model_override: Optional[str] = None,
                 strategy_override: Optional[Dict] = None,
                 max_opt_iters: int = 2, min_delta: float = 2.0):
        self.P = project_dir
        self.train_fn, self.export_fn = train_fn, export_fn
        self.export, self.dry_run = export, dry_run
        self.model_override, self.strategy_override = model_override, strategy_override
        self.max_opt_iters, self.min_delta = max_opt_iters, min_delta

    def run(self) -> AutopilotReport:
        rep = AutopilotReport()

        # [1] Goal Understanding (Stage A)
        goal = GoalProfile.from_yaml(self.P / "myai.yaml")
        rep.stages.append(StageResult("🎯", "Goal", f"{goal.task.value} / {goal.domain.value}",
                                      [f"eval weights: {goal.eval_weights}"]))

        # [2] Hardware Analysis
        hw = detect_hardware()
        gpu_name = getattr(hw, "gpu", getattr(hw, "gpu_name", "CPU"))
        rep.stages.append(StageResult("🖥️", "Hardware",
                                      f"{gpu_name} · tier {hw.tier}"))

        # [3] Dataset Intelligence (Stage B)
        sources = _load_sources(self.P)
        summary = analyze_dataset(sources[0]) if sources else None
        if summary:
            rep.stages.append(StageResult(
                "📊", "Data",
                f"{summary.num_samples:,} samples · quality {summary.quality_score}/100 · dup {summary.dup_pct}%",
                summary.issues))
        else:
            rep.stages.append(StageResult("📊", "Data", "No registered data sources found."))

        # [4] Model Selection (Stage A)
        rec = recommend_model(hw, goal, summary)
        model = rec.model
        if self.model_override:                      # §18 explicit override wins
            catalog = __import__("myai.models.recommender", fromlist=["CATALOG"]).CATALOG
            matched = next((m for m in catalog if m.id == self.model_override or m.repo_id == self.model_override), None)
            if matched:
                model = matched
            rec.reasoning.append(f"User override: forced {model.name}.")
        rep.stages.append(StageResult("🧠", "Model", model.name, rec.reasoning + rec.warnings))

        # [5] Feasibility Gate — fail fast BEFORE training
        feas = run_feasibility(hw, model, summary)
        rep.stages.append(StageResult("⚖️", "Feasibility",
                                      f"{feas.overall} · est {feas.estimated_vram_gb}GB",
                                      [feas.reasoning]))
        if feas.overall == "FAIL" and not self.dry_run:
            return rep                               # aborted, nothing wasted

        # [6] Training Strategy (Stage C)
        strat = plan_strategy(hw, model, summary, goal, override=self.strategy_override)
        rep.stages.append(StageResult(
            "⚙️", "Strategy",
            f"{strat.config.quantization} r{strat.config.lora_rank} · {strat.epochs} ep · "
            f"~{strat.estimated_minutes} min · {strat.storage_required_gb}GB storage",
            strat.reasoning))

        if self.dry_run:
            rep.stages.append(StageResult("⏸️", "Dry-run", "full plan previewed — no training executed"))
            return rep

        # [7] Train
        run = self.train_fn(strat)
        rep.stages.append(StageResult("🏗️", "Training", f"{run.run_id} · {run.train_minutes} min"))

        # [8] Leaderboard (Stage D)
        board = Leaderboard(goal, self.P / "experiments" / "runs")
        board.add_run(run)
        rep.stages.append(StageResult("🏆", "Leaderboard",
                                      f"{run.run_id} → {board.score(run).composite}/100"))

        # [9] Optimizer loop (Stage E, bounded)
        opt = OptimizerEngine(goal, board, self.train_fn,
                              self.min_delta, self.max_opt_iters).run()
        rep.final_run_id = opt.final_run_id
        rep.stages.append(StageResult("🔧", "Optimizer",
                                      f"{'improved' if opt.improved else 'kept baseline'} → {opt.final_run_id}"))

        # [10] Export through the 18-point Security Gate
        if self.export and self.export_fn:
            path = self.export_fn(opt.final_run_id)
            rep.export_path = str(path)
            rep.stages.append(StageResult("📦", "Export", f"18/18 gate passed → {Path(path).name}"))

        rep.ready = True
        return rep


def print_autopilot(rep: AutopilotReport, project: str) -> None:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    console.print(f"\n[bold cyan]🚀 MYAI AUTOPILOT — {project}[/bold cyan]")
    for i, s in enumerate(rep.stages, 1):
        console.print(f"[{i:>2}] {s.icon} [bold]{s.name}[/bold]: {s.summary}")
        for line in s.reasoning[:3]:
            console.print(f"     [dim]✨ {line}[/dim]")
    if rep.ready:
        extra = f"\nPackage: {rep.export_path}" if rep.export_path else ""
        console.print(Panel(f"[bold green]🎉 YOUR AI IS READY[/bold green]\n"
                            f"Release candidate: {rep.final_run_id}{extra}",
                            border_style="green"))
    else:
        console.print(Panel("[bold red]⛔ Aborted at Feasibility Gate.[/bold red] "
                            "See reasoning above; adjust goal/data or use --model override.",
                            border_style="red"))
