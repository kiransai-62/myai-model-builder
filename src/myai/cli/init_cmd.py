from typing import Optional
from pathlib import Path
import typer
from ..core.config import ProjectConfig
from ..core.console import print_success
from ..core.goal import prompt_for_goal, GoalProfile, TaskType, Domain

def init(
    project_name: str,
    task: Optional[str] = typer.Option(None, "--task", "-t", help="Primary task (instruction-tuning, chat, domain-qa, code, etc.)"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Operating domain (general, medical, finance, fitness, etc.)"),
    context_priority: Optional[str] = typer.Option(None, "--context", help="Context length priority (short, balanced, long-context)"),
    latency_priority: Optional[str] = typer.Option(None, "--latency", help="Latency priority (fast, balanced, high-quality)"),
    target_deployment: Optional[str] = typer.Option(None, "--target", help="Target deployment (edge, local-cpu-gpu, server)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Use default goal settings without prompting"),
):
    root = Path(project_name).resolve()
    if root.exists() and any(root.iterdir()):
        print(f"Error: Directory {root} is not empty.")
        raise typer.Exit(1)
        
    root.mkdir(parents=True, exist_ok=True)
    
    dirs = ["data/train", "data/validation", "data/evaluation", "models/base", "models/trained", "indexes", "checkpoints", "dist"]
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
    
    # Prompt or infer goal profile
    goal_profile = prompt_for_goal(
        non_interactive=yes,
        task=task,
        domain=domain,
        context_priority=context_priority,
        latency_priority=latency_priority,
        target_deployment=target_deployment,
    )
    
    cfg = ProjectConfig(name=project_name, goal=goal_profile)
    cfg.save(root)
    
    gitignore = "models/base/\nmodels/trained/\nindexes/\ncheckpoints/\ndist/\n"
    (root / ".gitignore").write_text(gitignore)
    
    print_success(f"Created project: {project_name}")
    print(f"Next steps:\n  cd {project_name}\n  myai system check")