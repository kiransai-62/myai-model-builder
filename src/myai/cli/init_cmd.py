from typing import Optional
from pathlib import Path
import typer
from ..core.config import ProjectConfig
from ..core.console import print_success
from ..core.goal import prompt_for_goal, GoalProfile, TaskType, Domain

def init(
    project_name: str = typer.Argument(".", help="Project name or '.' for current directory"),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="Primary task (instruction-tuning, chat, domain-qa, code, etc.)"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Operating domain (general, medical, finance, fitness, etc.)"),
    context_priority: Optional[str] = typer.Option(None, "--context", help="Context length priority (short, balanced, long-context)"),
    latency_priority: Optional[str] = typer.Option(None, "--latency", help="Latency priority (fast, balanced, high-quality)"),
    target_deployment: Optional[str] = typer.Option(None, "--target", help="Target deployment (edge, local-cpu-gpu, server)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Use default goal settings without prompting"),
):
    root = Path(project_name).resolve()
    actual_project_name = root.name if project_name in (".", "./", "") else project_name
    
    if (root / "myai.yaml").exists():
        print(f"Error: Directory {root} is already an initialized MYAI project.")
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
    
    cfg = ProjectConfig(name=actual_project_name, goal=goal_profile)
    cfg.save(root)
    
    gitignore_path = root / ".gitignore"
    if not gitignore_path.exists():
        gitignore = "models/base/\nmodels/trained/\nindexes/\ncheckpoints/\ndist/\n"
        gitignore_path.write_text(gitignore)
    
    print_success(f"Initialized MYAI project: {actual_project_name}")
    if project_name not in (".", "./", ""):
        print(f"Next steps:\n  cd {project_name}\n  myai system check")
    else:
        print("Next steps:\n  myai data add <path/to/data.jsonl>\n  myai auto --export")