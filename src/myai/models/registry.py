from typing import List
from .schema import RegistryModel
from ..registry.loader import load_registry_models


def get_registry_models() -> List[RegistryModel]:
    """Retrieves all registered models from the hierarchical model registry."""
    return load_registry_models()