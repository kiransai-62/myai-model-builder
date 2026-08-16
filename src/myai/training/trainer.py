import json
import time
from pathlib import Path
from ..core.console import console
from ..data.manager import resolve_dataset_source
from ..core.home import ensure_home
from .runs import RunManager
from .engine import run_training_engine

def run_training(root: Path, cfg, spec, max_retries: int = 3, selection_mode: str = "user_selected") -> dict:
    """Compatibility wrapper around run_training_engine and RunManager."""
    home = ensure_home()
    manager = RunManager(home)
    source = resolve_dataset_source(root, cfg)
    
    run = manager.create({
        "run_id": "",
        "project": cfg.name,
        "dataset_id": getattr(cfg, "dataset_id", ""),
        "base_model": cfg.model_id,
        "selection_mode": selection_mode,
        "training_method": cfg.training.method,
        "epochs": cfg.training.epochs,
        "batch_size": cfg.training.batch_size,
        "grad_accum": cfg.training.grad_accum,
        "learning_rate": cfg.training.learning_rate,
        "seq_length": cfg.training.seq_length,
    })
    
    ctx = {
        "cfg": cfg,
        "spec": spec,
        "source": source,
        "home": home,
        "root": root,
        "selection_mode": selection_mode,
        "budget_gb": 0.0,
        "resume_ckpt": None,
    }
    
    metadata = run_training_engine(run, ctx)
    trained_dir = root / "models" / "trained" / cfg.name
    trained_dir.mkdir(parents=True, exist_ok=True)
    (trained_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata