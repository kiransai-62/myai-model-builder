from dataclasses import dataclass, field
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
    # Enhanced MYAI Catalog Fields
    family: str = ""
    architecture: str = "Dense"  # Dense | MoE
    active_parameters: str = ""
    modality: str = "Text"  # Text | Text + Vision | Multimodal
    quantizations: List[str] = field(default_factory=lambda: ["FP16", "FP8", "INT8", "INT4", "GGUF"])
    cpu_min_cores: int = 4
    ram_min_gb: float = 8.0
    gpu_min_vram_gb: float = 8.0
    vram_q4_gb: float = 4.0
    vram_fp16_gb: float = 16.0
    storage_gb: float = 8.0
    finetune_vram_gb: float = 12.0
    training_ram_gb: float = 16.0
    has_vision: bool = False
    has_audio: bool = False
    has_tools: bool = True
    has_reasoning: bool = True
    recommended_tier: str = "T1"
    confidence: float = 0.95

    def __post_init__(self):
        if not self.family:
            self.family = self.name.split()[0] if self.name else self.id
        if not self.active_parameters:
            self.active_parameters = self.parameters
        if self.gpu_min_vram_gb == 8.0 and self.vram_min != 8.0:
            self.gpu_min_vram_gb = self.vram_min
        elif self.vram_min == 8.0 and self.gpu_min_vram_gb != 8.0:
            self.vram_min = self.gpu_min_vram_gb

    @property
    def repo_id(self) -> str:
        return self.repository or self.id

    @property
    def params_b(self) -> float:
        return self.parameters_billions

    @property
    def parameters_billions(self) -> float:
        val = self.parameters.upper().replace("B", "").replace("M", "").strip()
        try:
            parsed = float(val)
            if "M" in self.parameters.upper():
                return parsed / 1000.0
            return parsed
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
        caps = data.get("capabilities", {})

        vram_min = float(hw.get("minimum_vram_gb", 8.0))
        params_str = str(arch.get("parameters", "3B"))

        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            family=data.get("family", data.get("name", data["id"]).split()[0]),
            parameters=params_str,
            architecture=arch.get("type", "Dense"),
            active_parameters=str(arch.get("active_parameters", params_str)),
            modality=data.get("modality", "Text"),
            vram_min=vram_min,
            gpu_min_vram_gb=vram_min,
            cpu_min_cores=int(hw.get("cpu_min_cores", 4)),
            ram_min_gb=float(hw.get("ram_min_gb", 8.0)),
            vram_q4_gb=float(hw.get("vram_q4_gb", vram_min)),
            vram_fp16_gb=float(hw.get("vram_fp16_gb", vram_min * 2.0)),
            storage_gb=float(hw.get("storage_gb", 8.0)),
            finetune_vram_gb=float(hw.get("finetune_vram_gb", vram_min * 1.5)),
            training_ram_gb=float(hw.get("training_ram_gb", 16.0)),
            methods=training.get("methods", ["QLoRA", "LoRA"]),
            repository=src.get("repository", data["id"]),
            license=lic.get("name", "Apache 2.0"),
            hidden_size=int(arch.get("hidden_size", 4096)),
            num_layers=int(arch.get("num_layers", 32)),
            context_length=int(arch.get("context_length", 4096)),
            quantizations=data.get("quantizations", ["FP16", "FP8", "INT8", "INT4", "GGUF"]),
            has_vision=bool(caps.get("vision", False)),
            has_audio=bool(caps.get("audio", False)),
            has_tools=bool(caps.get("tools", True)),
            has_reasoning=bool(caps.get("reasoning", True)),
            recommended_tier=data.get("recommended_tier", "T1"),
            confidence=float(data.get("confidence", 0.95)),
        )


ModelSpec = RegistryModel