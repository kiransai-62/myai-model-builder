"""MYAI Project Understanding & Lifecycle State Machine (Report §4, §10).

Tracks project lifecycle states and validates operational preconditions:
  INITIALIZED ──▶ DATA_READY ──▶ FEASIBLE ──▶ TRAINED ──▶ EVALUATED ──▶ READY_TO_EXPORT ──▶ EXPORTED
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import ProjectConfig, load_config
from .goal import GoalProfile


class ProjectState(str, Enum):
    INITIALIZED = "initialized"
    DATA_READY = "data_ready"
    FEASIBLE = "feasible"
    TRAINED = "trained"
    EVALUATED = "evaluated"
    OPTIMIZED = "optimized"
    READY_TO_EXPORT = "ready_to_export"
    EXPORTED = "exported"


@dataclass
class ProjectStatus:
    project_name: str
    project_dir: Path
    state: ProjectState
    goal: GoalProfile
    has_data: bool
    data_samples: int
    has_trained_model: bool
    latest_run_id: Optional[str]
    has_evaluation: bool
    best_score: Optional[float]
    has_export: bool
    export_path: Optional[str]
    next_step: str
    actionable_command: str
    missing_prerequisites: List[str] = field(default_factory=list)


def inspect_project_state(project_dir: Path) -> ProjectStatus:
    """Inspects workspace artifacts and computes the current lifecycle state."""
    cfg_file = project_dir / "myai.yaml"
    if not cfg_file.exists():
        return ProjectStatus(
            project_name=project_dir.name,
            project_dir=project_dir,
            state=ProjectState.INITIALIZED,
            goal=GoalProfile(),
            has_data=False,
            data_samples=0,
            has_trained_model=False,
            latest_run_id=None,
            has_evaluation=False,
            best_score=None,
            has_export=False,
            export_path=None,
            next_step="Initialize project configuration",
            actionable_command="myai init",
            missing_prerequisites=["myai.yaml configuration file missing"],
        )

    cfg = ProjectConfig.load(project_dir)
    goal = GoalProfile.from_yaml(cfg_file)

    # 1. Check data
    from ..data.manager import resolve_dataset_source
    src_path = resolve_dataset_source(project_dir, cfg)
    data_dir = project_dir / "data"
    train_file = data_dir / "train.jsonl"
    sources_file = data_dir / "sources.yaml"
    
    has_data = bool(
        (src_path and src_path.exists())
        or (cfg.dataset_id and str(cfg.dataset_id).strip())
        or train_file.exists()
        or sources_file.exists()
        or (data_dir.exists() and (any(data_dir.glob("*.jsonl")) or any(data_dir.glob("*.json"))))
    )

    data_samples = 0
    if src_path and src_path.exists():
        if src_path.is_file():
            try:
                data_samples = sum(1 for _ in src_path.open(encoding="utf-8", errors="ignore"))
            except Exception:
                data_samples = 0
        elif src_path.is_dir():
            for f in src_path.glob("*.jsonl"):
                try:
                    data_samples += sum(1 for _ in f.open(encoding="utf-8", errors="ignore"))
                except Exception:
                    pass
    elif train_file.exists():
        try:
            data_samples = sum(1 for _ in train_file.open(encoding="utf-8", errors="ignore"))
        except Exception:
            data_samples = 0

    # 2. Check trained model & runs for THIS project
    models_dir = project_dir / "models" / "trained"
    has_trained = models_dir.exists() and any(models_dir.iterdir())
    from .home import ensure_home
    from ..training.runs import RunManager
    home = ensure_home()
    runman = RunManager(home)
    project_runs = [r for r in runman.list() if r.get("config", {}).get("project") == cfg.name]
    latest_run = project_runs[0]["run_id"] if project_runs else None
    if project_runs:
        has_trained = True

    # 3. Check evaluation & leaderboard
    lb_dir = home / "leaderboard"
    has_eval = False
    best_score = None
    if lb_dir.exists() and any(lb_dir.glob("*.json")):
        from ..models.leaderboard import Leaderboard
        lb = Leaderboard(goal, runs_dir=lb_dir)
        proj_runs = [r for r in lb.runs if any(pr["run_id"] == r.run_id for pr in project_runs)]
        if proj_runs:
            has_eval = True
            scored = [lb.score(r) for r in proj_runs if r.regression_passed]
            if scored:
                best_score = max(s.composite for s in scored)


    # 4. Check export
    export_dir = project_dir / "export"
    has_export = False
    export_path = None
    if export_dir.exists():
        packages = list(export_dir.glob("*.myai")) + list(export_dir.glob("*.zip"))
        if packages:
            has_export = True
            export_path = str(packages[0])

    # Determine state & next action
    missing = []
    if not has_data:
        state = ProjectState.INITIALIZED
        missing.append("No training dataset registered or prepared")
        next_step = "Add training dataset"
        action_cmd = "myai data add <path>"
    elif not has_trained:
        state = ProjectState.DATA_READY
        next_step = "Train base model on data"
        action_cmd = "myai train"
    elif not has_eval:
        state = ProjectState.TRAINED
        next_step = "Evaluate model quality against goal benchmarks"
        action_cmd = "myai evaluate"
    elif not has_export:
        state = ProjectState.READY_TO_EXPORT
        next_step = "Export standalone model package"
        action_cmd = "myai export"
    else:
        state = ProjectState.EXPORTED
        next_step = "Model is exported and ready to deploy"
        action_cmd = "myai serve"

    return ProjectStatus(
        project_name=cfg.name,
        project_dir=project_dir,
        state=state,
        goal=goal,
        has_data=has_data,
        data_samples=data_samples,
        has_trained_model=has_trained,
        latest_run_id=latest_run,
        has_evaluation=has_eval,
        best_score=best_score,
        has_export=has_export,
        export_path=export_path,
        next_step=next_step,
        actionable_command=action_cmd,
        missing_prerequisites=missing,
    )


def validate_precondition(
    project_dir: Path,
    required_state: ProjectState,
) -> Tuple[bool, str]:
    """Validate that the project has reached a prerequisite state before proceeding."""
    status = inspect_project_state(project_dir)
    state_order = [
        ProjectState.INITIALIZED,
        ProjectState.DATA_READY,
        ProjectState.FEASIBLE,
        ProjectState.TRAINED,
        ProjectState.EVALUATED,
        ProjectState.READY_TO_EXPORT,
        ProjectState.EXPORTED,
    ]

    curr_idx = state_order.index(status.state) if status.state in state_order else 0
    req_idx = state_order.index(required_state) if required_state in state_order else 0

    if curr_idx < req_idx:
        msg = (
            f"Cannot proceed: Project state is [{status.state.value.upper()}], "
            f"but [{required_state.value.upper()}] is required.\n"
            f"👉 Recommended next step: {status.actionable_command}"
        )
        return False, msg
    return True, "Preconditions satisfied."
