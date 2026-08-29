"""
Hierarchical Model Registry Loader for MYAI.

Recursively discovers and loads all model YAML specifications from
family subdirectories under `src/myai/registry/models/**/`.
"""
from pathlib import Path
from typing import List, Dict, Optional
from ..models.schema import RegistryModel


def get_model_registry_root() -> Path:
    """Returns the root directory of the model registry YAMLs."""
    return Path(__file__).resolve().parent / "models"


def load_registry_models() -> List[RegistryModel]:
    """
    Recursively scans and loads all model YAML files in family subdirectories.
    """
    root = get_model_registry_root()
    if not root.exists():
        return []

    yaml_files = list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))
    models: List[RegistryModel] = []
    seen_ids = set()

    for yf in sorted(yaml_files):
        try:
            m = RegistryModel.from_yaml(yf)
            if m.id not in seen_ids:
                seen_ids.add(m.id)
                models.append(m)
        except Exception as e:
            # Skip invalid YAML or log warning
            continue

    return models
