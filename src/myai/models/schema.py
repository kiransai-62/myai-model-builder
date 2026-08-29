"""
MYAI Enhanced Model Registry & Hardware Schema.

Defines rich architectural, CPU, GPU, multi-GPU, memory, storage, quantization, 
and capability specifications for all catalog model families (0.1B to 675B+).
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml


@dataclass
class CpuRequirements:
    min_cores: int = 4
    rec_cores: int = 8
    architecture: str = "x86_64"  # x86_64 | arm64
    instruction_sets: List[str] = field(default_factory=lambda: ["AVX2"])  # AVX2, AVX-512, ARM_NEON
    est_cpu_tok_per_sec: float = 12.0


@dataclass
class InferenceRequirements:
    q4_vram_gb: float = 4.0
    q8_vram_gb: float = 6.0
    fp16_vram_gb: float = 12.0
    ram_gb: float = 8.0
    est_tokens_per_sec: float = 35.0


@dataclass
class TrainingRequirements:
    lora_vram_gb: float = 14.0
    qlora_vram_gb: float = 8.0
    layer_streaming_vram_gb: float = 3.32
    training_ram_gb: float = 16.0
    download_size_gb: float = 6.0
    runtime_storage_gb: float = 8.0
    workspace_storage_gb: float = 24.0


@dataclass
class MultiGpuTopology:
    min_count: int = 1
    rec_count: int = 1
    vram_per_gpu_gb: float = 8.0
    total_vram_gb: float = 8.0
    interconnect: str = "PCIe"  # PCIe | NVLink | InfiniBand
    parallelism: str = "None"   # None | TP | PP | FSDP


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
    
    # ── Enhanced MYAI 15-Point Hardware Intelligence Fields ─────────
    family: str = ""
    architecture: str = "Dense"  # Dense | MoE
    active_parameters: str = ""
    num_experts: int = 1
    num_active_experts: int = 1
    modality: str = "Text"  # Text | Text + Vision | Multimodal
    
    # Subsystems
    cpu: CpuRequirements = field(default_factory=CpuRequirements)
    inference: InferenceRequirements = field(default_factory=InferenceRequirements)
    training: TrainingRequirements = field(default_factory=TrainingRequirements)
    multi_gpu: MultiGpuTopology = field(default_factory=MultiGpuTopology)
    
    # Quantizations & Training Methods
    quantizations: List[str] = field(
        default_factory=lambda: ["FP16", "FP8", "INT8", "INT4", "GGUF_Q4_K_M", "GGUF_Q8_0", "AWQ", "GPTQ"]
    )
    training_compatibility: Dict[str, bool] = field(
        default_factory=lambda: {
            "sft": True, "lora": True, "qlora": True, "layer_streaming": True,
            "dpo": True, "orpo": True, "simpo": True, "kto": True, "grpo": False,
        }
    )
    
    # Tasks & Capabilities
    tasks: List[str] = field(
        default_factory=lambda: ["chat", "code", "domain_qa", "reasoning", "classification", "summarization", "extraction"]
    )
    capabilities: Dict[str, bool] = field(
        default_factory=lambda: {
            "tool_calling": True, "structured_json": True, "function_calling": True,
            "long_context": True, "reasoning_chain": True, "rag_optimized": True,
        }
    )
    
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
        if self.inference.q4_vram_gb == 4.0 and self.vram_min != 8.0:
            self.inference.q4_vram_gb = self.vram_min
        if self.vram_min == 8.0 and self.inference.q4_vram_gb != 4.0:
            self.vram_min = self.inference.q4_vram_gb

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

    @property
    def active_parameters_billions(self) -> float:
        val = self.active_parameters.upper().replace("B", "").replace("M", "").strip()
        try:
            parsed = float(val)
            if "M" in self.active_parameters.upper():
                return parsed / 1000.0
            return parsed
        except ValueError:
            return self.parameters_billions

    @classmethod
    def from_yaml(cls, path: Path) -> "RegistryModel":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        arch = data.get("architecture", {})
        hw = data.get("hardware", {})
        inf_data = data.get("inference", {})
        train_data = data.get("training", {})
        multi_data = data.get("multi_gpu", {})
        cpu_data = data.get("cpu", {})
        src = data.get("source", {})
        lic = data.get("license", {})
        caps_data = data.get("capabilities", {})

        vram_min = float(hw.get("minimum_vram_gb", inf_data.get("q4_vram_gb", 8.0)))
        params_str = str(arch.get("parameters", "3B"))
        active_str = str(arch.get("active_parameters", params_str))

        # Build CPU Requirements
        cpu_req = CpuRequirements(
            min_cores=int(cpu_data.get("min_cores", hw.get("cpu_min_cores", 4))),
            rec_cores=int(cpu_data.get("rec_cores", 8)),
            architecture=cpu_data.get("architecture", "x86_64"),
            instruction_sets=cpu_data.get("instruction_sets", ["AVX2"]),
            est_cpu_tok_per_sec=float(cpu_data.get("est_cpu_tok_per_sec", 12.0)),
        )

        # Build Inference Requirements
        inf_req = InferenceRequirements(
            q4_vram_gb=float(inf_data.get("q4_vram_gb", hw.get("vram_q4_gb", vram_min))),
            q8_vram_gb=float(inf_data.get("q8_vram_gb", vram_min * 1.3)),
            fp16_vram_gb=float(inf_data.get("fp16_vram_gb", hw.get("vram_fp16_gb", vram_min * 2.0))),
            ram_gb=float(inf_data.get("ram_gb", hw.get("ram_min_gb", 8.0))),
            est_tokens_per_sec=float(inf_data.get("est_tokens_per_sec", 35.0)),
        )

        # Build Training Requirements
        train_req = TrainingRequirements(
            lora_vram_gb=float(train_data.get("lora_vram_gb", hw.get("finetune_vram_gb", vram_min * 1.5))),
            qlora_vram_gb=float(train_data.get("qlora_vram_gb", vram_min * 1.1)),
            layer_streaming_vram_gb=float(train_data.get("layer_streaming_vram_gb", 3.32)),
            training_ram_gb=float(train_data.get("training_ram_gb", hw.get("training_ram_gb", 16.0))),
            download_size_gb=float(train_data.get("download_size_gb", hw.get("storage_gb", 8.0))),
            runtime_storage_gb=float(train_data.get("runtime_storage_gb", hw.get("storage_gb", 8.0) * 1.2)),
            workspace_storage_gb=float(train_data.get("workspace_storage_gb", hw.get("storage_gb", 8.0) * 2.5)),
        )

        # Build Multi-GPU Topology
        multi_gpu = MultiGpuTopology(
            min_count=int(multi_data.get("min_count", 1)),
            rec_count=int(multi_data.get("rec_count", 1)),
            vram_per_gpu_gb=float(multi_data.get("vram_per_gpu_gb", vram_min)),
            total_vram_gb=float(multi_data.get("total_vram_gb", vram_min)),
            interconnect=multi_data.get("interconnect", "PCIe"),
            parallelism=multi_data.get("parallelism", "None"),
        )

        methods_list = train_data.get("methods", ["QLoRA", "LoRA", "layer_streaming"])
        if "methods" in hw and not train_data.get("methods"):
            methods_list = hw.get("methods")

        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            family=data.get("family", data.get("name", data["id"]).split()[0]),
            parameters=params_str,
            architecture=arch.get("type", "Dense"),
            active_parameters=active_str,
            num_experts=int(arch.get("num_experts", 1)),
            num_active_experts=int(arch.get("num_active_experts", 1)),
            modality=data.get("modality", "Text"),
            vram_min=vram_min,
            methods=methods_list,
            repository=src.get("repository", data["id"]),
            license=lic.get("name", "Apache 2.0"),
            hidden_size=int(arch.get("hidden_size", 4096)),
            num_layers=int(arch.get("num_layers", 32)),
            context_length=int(arch.get("context_length", 4096)),
            cpu=cpu_req,
            inference=inf_req,
            training=train_req,
            multi_gpu=multi_gpu,
            quantizations=data.get("quantizations", ["FP16", "FP8", "INT8", "INT4", "GGUF_Q4_K_M", "GGUF_Q8_0"]),
            training_compatibility=train_data.get("compatibility", {
                "sft": True, "lora": True, "qlora": True, "layer_streaming": "layer_streaming" in methods_list,
                "dpo": True, "orpo": True, "simpo": True, "kto": True, "grpo": False,
            }),
            tasks=data.get("tasks", ["chat", "code", "domain_qa", "reasoning", "classification", "summarization", "extraction"]),
            capabilities=caps_data if isinstance(caps_data, dict) else {},
            has_vision=bool(caps_data.get("vision", False)),
            has_audio=bool(caps_data.get("audio", False)),
            has_tools=bool(caps_data.get("tools", True)),
            has_reasoning=bool(caps_data.get("reasoning", True)),
            recommended_tier=data.get("recommended_tier", "T1"),
            confidence=float(data.get("confidence", 0.95)),
        )


ModelSpec = RegistryModel