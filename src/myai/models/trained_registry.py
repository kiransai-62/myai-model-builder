import json
import shutil
import time
from pathlib import Path

def register_trained(
    home,
    name: str,
    base_model_id: str,
    dataset_id: str,
    run_id: str,
    adapter_src: Path,
    eval_report,
    method: str,
    selection_mode: str = "user_selected",
    root: Path | None = None,
    eval_score: float = None,
) -> Path:
    home = Path(home)
    adapters_dst = home / "models" / "adapters" / base_model_id / name
    trained_dst = home / "models" / "trained" / name

    adapters_dst.mkdir(parents=True, exist_ok=True)
    trained_dst.mkdir(parents=True, exist_ok=True)

    if Path(adapter_src).exists():
        shutil.copytree(adapter_src, adapters_dst, dirs_exist_ok=True)
        shutil.copytree(adapter_src, trained_dst / "model", dirs_exist_ok=True)

    metadata = {
        "id": name,
        "name": name,
        "base_model": base_model_id,
        "base_model_id": base_model_id,
        "dataset": dataset_id,
        "dataset_id": dataset_id,
        "run_id": run_id,
        "method": method,
        "selection_mode": selection_mode,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "READY",
        "adapter_path": str(adapters_dst),
    }
    if eval_score is not None:
        metadata["evaluation"] = f"{int(eval_score * 100)}%"
    elif hasattr(eval_report, "overall"):
        metadata["evaluation"] = f"{int(eval_report.overall * 100)}%"

    (trained_dst / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    
    eval_data = eval_report.to_dict() if hasattr(eval_report, "to_dict") else eval_report
    (trained_dst / "evaluation.json").write_text(
        json.dumps(eval_data, indent=2), encoding="utf-8"
    )

    if root:
        proj_trained = Path(root) / "models" / "trained" / name
        proj_trained.mkdir(parents=True, exist_ok=True)
        (proj_trained / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (proj_trained / "evaluation.json").write_text(json.dumps(eval_data, indent=2), encoding="utf-8")
        if Path(adapter_src).exists():
            shutil.copytree(adapter_src, proj_trained / "adapter", dirs_exist_ok=True)

    return trained_dst

def list_trained(home) -> list:
    out = []
    trained_dir = Path(home) / "models" / "trained"
    if trained_dir.exists():
        for meta_file in sorted(trained_dir.glob("*/metadata.json")):
            try:
                out.append(json.loads(meta_file.read_text(encoding="utf-8")))
            except Exception:
                pass
    return out

def resolve_adapter(home, name: str, root: Path | None = None) -> Path | None:
    # 1. Project local adapter first if exists
    if root:
        proj_adapter = Path(root) / "models" / "trained" / name / "adapter"
        if proj_adapter.exists() and any(proj_adapter.iterdir()):
            return proj_adapter
        proj_meta = Path(root) / "models" / "trained" / name / "metadata.json"
        if proj_meta.exists():
            try:
                ad_path = Path(json.loads(proj_meta.read_text(encoding="utf-8")).get("adapter_path", ""))
                if ad_path.exists():
                    return ad_path
            except Exception:
                pass

    # 2. Central registry metadata
    meta_file = Path(home) / "models" / "trained" / name / "metadata.json"
    if meta_file.exists():
        try:
            ad_path = Path(json.loads(meta_file.read_text(encoding="utf-8")).get("adapter_path", ""))
            if ad_path.exists():
                return ad_path
        except Exception:
            pass

    # 3. Direct path in ~/.myai/models/trained/<name>/model
    direct_model = Path(home) / "models" / "trained" / name / "model"
    if direct_model.exists() and any(direct_model.iterdir()):
        return direct_model

    # 4. Direct path in ~/.myai/models/trained/<name>/adapter
    direct_adapter = Path(home) / "models" / "trained" / name / "adapter"
    if direct_adapter.exists() and any(direct_adapter.iterdir()):
        return direct_adapter

    return None
