from pathlib import Path
from importlib import resources
from .schema import RegistryModel

def get_registry_models() -> list[RegistryModel]:
    models = []
    # In a real package, use importlib.resources. For local dev, we find the file relative to this script.
    registry_dir = Path(__file__).parent.parent / "registry" / "models"
    if registry_dir.exists():
        for f in registry_dir.glob("*.yaml"):
            models.append(RegistryModel.from_yaml(f))
    return models