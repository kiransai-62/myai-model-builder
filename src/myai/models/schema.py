from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import yaml

@dataclass
class RegistryModel:
    id: str
    name: str
    parameters: str
    vram_min: float
    methods: List[str]
    repository: str
    license: str
    hidden_size: int = 4096
    num_layers: int = 32
    context_length: int = 4096

    @property
    def repo_id(self) -> str:
        return self.repository or self.id

    @property
    def params_b(self) -> float:
        return self.parameters_billions

    @property
    def parameters_billions(self) -> float:
        val = self.parameters.upper().replace("B", "").strip()
        try:
            return float(val)
        except ValueError:
            return 3.0

    @classmethod
    def from_yaml(cls, path: Path) -> "RegistryModel":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        arch = data.get("architecture", {})
        hw = data.get("hardware", {})
        training = data.get("training", {})
        src = data.get("source", {})
        lic = data.get("license", {})
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            parameters=arch.get("parameters", "3B"),
            vram_min=hw.get("minimum_vram_gb", 8.0),
            methods=training.get("methods", ["QLoRA"]),
            repository=src.get("repository", data["id"]),
            license=lic.get("name", "Apache 2.0"),
            hidden_size=arch.get("hidden_size", 4096),
            num_layers=arch.get("num_layers", 32),
            context_length=arch.get("context_length", 4096),
        )

ModelSpec = RegistryModel