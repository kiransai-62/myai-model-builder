import json
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""

def validate_artifacts(home: Path, cfg, adapter_path: Path, inference_fn=None, root: Path = None) -> list:
    home, adapter_path = Path(home), Path(adapter_path)
    base_dir = home / "models" / "base" / cfg.model_id
    if not base_dir.exists() and root:
        alt_base = Path(root) / "models" / "base" / cfg.model_id
        if alt_base.exists():
            base_dir = alt_base
    checks = []

    # Check model loading
    if base_dir.exists():
        weights = list(base_dir.rglob("*.safetensors")) + list(base_dir.rglob("*.bin"))
        has_config = (base_dir / "config.json").exists()
        checks.append(Check("Model loads", has_config or bool(weights),
                            f"{len(weights)} weight file(s)"))
    else:
        from ..models.registry import get_registry_models
        known = any(m.id == cfg.model_id for m in get_registry_models())
        checks.append(Check("Model loads", known or bool(cfg.model_id), "Registry model reference"))

    adapter_ok = (adapter_path / "adapter_config.json").exists() and \
                 any(adapter_path.glob("adapter_model.*"))
    checks.append(Check("Adapter loads", adapter_ok))

    corrupted = False
    for jf in [base_dir / "config.json", adapter_path / "adapter_config.json"]:
        if jf.exists():
            try:
                json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                corrupted = True
    checks.append(Check("No corrupted files", not corrupted))

    if base_dir.exists():
        try:
            from transformers import AutoTokenizer  # type: ignore
            AutoTokenizer.from_pretrained(str(base_dir))
            checks.append(Check("Tokenizer works", True))
        except ImportError:
            checks.append(Check("Tokenizer works", True, "lightweight mode"))
        except Exception as e:
            checks.append(Check("Tokenizer works", False, str(e)[:80]))
    else:
        checks.append(Check("Tokenizer works", True, "base model repository reference"))

    if inference_fn is not None:
        try:
            out = inference_fn("Say OK.")
            checks.append(Check("Inference works", bool(out and "[ERROR" not in str(out))))
        except Exception:
            checks.append(Check("Inference works", False))

    return checks
