from dataclasses import dataclass
import yaml

@dataclass
class RegistryModel:
    id: str
    name: str
    parameters: str
    vram_min: float
    methods: list[str]
    repository: str
    license: str

    @property
    def parameters_billions(self) -> float:
        val = self.parameters.upper().replace("B", "").strip()
        try:
            return float(val)
        except ValueError:
            return 3.0

    @classmethod
    def from_yaml(cls, path) -> "RegistryModel":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            id=data["id"], name=data["name"], parameters=data["architecture"]["parameters"],
            vram_min=data["hardware"]["minimum_vram_gb"], methods=data["training"]["methods"],
            repository=data["source"]["repository"], license=data["license"]["name"]
        )