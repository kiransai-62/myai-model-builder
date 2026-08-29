"""
Model Registry Schema & Integrity Validator.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
from ..models.schema import RegistryModel


class ModelRegistryValidator:
    """Validates structural correctness and logical integrity of registry YAMLs."""

    REQUIRED_FIELDS = ["id", "architecture", "inference", "training"]

    @classmethod
    def validate_file(cls, path: Path) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return False, [f"Root of {path.name} must be a dictionary mapping."]

            for req in cls.REQUIRED_FIELDS:
                if req not in data:
                    errors.append(f"Missing required field '{req}' in {path.name}")

            # Try parsing with schema
            model = RegistryModel.from_yaml(path)
            if model.parameters_billions <= 0:
                errors.append(f"Invalid parameter count '{model.parameters}' in {path.name}")
        except Exception as e:
            errors.append(f"Failed to parse {path.name}: {str(e)}")

        return len(errors) == 0, errors

    @classmethod
    def validate_all(cls, root: Path) -> Dict[str, List[str]]:
        results = {}
        for yf in root.rglob("*.yaml"):
            ok, errs = cls.validate_file(yf)
            if not ok:
                results[yf.name] = errs
        return results
