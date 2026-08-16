import typer
from pathlib import Path
from ..core.config import ProjectConfig
from ..core.console import print_success

def init(project_name: str):
    root = Path(project_name).resolve()
    if root.exists() and any(root.iterdir()):
        print(f"Error: Directory {root} is not empty.")
        raise typer.Exit(1)
        
    root.mkdir(parents=True)
    
    dirs = ["data/train", "data/validation", "data/evaluation", "models/base", "models/trained", "indexes", "checkpoints", "dist"]
    for d in dirs: (root / d).mkdir(parents=True)
    
    cfg = ProjectConfig(name=project_name)
    cfg.save(root)
    
    gitignore = "models/base/\nmodels/trained/\nindexes/\ncheckpoints/\ndist/\n"
    (root / ".gitignore").write_text(gitignore)
    
    print_success(f"Created project: {project_name}")
    print(f"Next steps:\n  cd {project_name}\n  myai system check")