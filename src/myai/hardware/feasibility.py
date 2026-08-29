"""MYAI Hardware Feasibility & Dual-Gate Validation (Report §11, §13).

Calculates exact VRAM requirements, activation footprints, and conducts
dual-gate feasibility checks (Hardware Fit × Data Fit).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainingConfig:
    quantization: str = "4bit"         # "4bit", "8bit", "fp16"
    lora_rank: int = 16
    lora_alpha: int = 32
    seq_len: int = 512
    batch_size: int = 1
    grad_accum: int = 8
    grad_checkpointing: bool = True
    stream_layers: bool = False        # Exact Layer Streaming (Soup feature)
    stream_source: str = "ram"         # "ram" or "disk"
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    def __post_init__(self):
        if not self.lora_alpha:
            self.lora_alpha = self.lora_rank * 2


def _get_model_attribute(model: Any, attr: str, default: Any) -> Any:
    """Helper to extract model properties from schema objects or namespaces."""
    if hasattr(model, attr):
        return getattr(model, attr)
    if isinstance(model, dict) and attr in model:
        return model[attr]
    if attr == "params_b":
        if hasattr(model, "parameters_billions"):
            return getattr(model, "parameters_billions")
        if hasattr(model, "parameters"):
            val = str(getattr(model, "parameters")).upper().replace("B", "").strip()
            try:
                return float(val)
            except ValueError:
                pass
    return default


def estimate_vram_gb(model: Any, cfg: TrainingConfig) -> float:
    """
    Computes an empirical estimate of peak VRAM during fine-tuning (GB).
    Includes:
      1. Base model weights (quantized, full precision, or layer-streamed buffer)
      2. KV cache and forward activation memory (with/without gradient checkpointing)
      3. LoRA adapter parameters and optimizer states (AdamW)
      4. PyTorch CUDA allocator buffer / workspace headroom (~15%)
    """
    params_b = _get_model_attribute(model, "params_b", 3.0)
    hidden_size = _get_model_attribute(model, "hidden_size", 4096)
    num_layers = _get_model_attribute(model, "num_layers", 32)

    # 1. Base weights in GB
    if getattr(cfg, "stream_layers", False):
        # Layer Streaming: only 2 active buffer layers kept in VRAM
        layer_size_gb = (params_b * 0.55) / max(1, num_layers)
        weight_gb = layer_size_gb * 2.0
    elif cfg.quantization == "4bit":
        weight_gb = params_b * 0.55
    elif cfg.quantization == "8bit":
        weight_gb = params_b * 1.05
    else:  # fp16 / bf16
        weight_gb = params_b * 2.05

    # 2. Activation memory (GB)
    base_act_bytes = num_layers * cfg.seq_len * hidden_size * cfg.batch_size * 2
    if cfg.grad_checkpointing:
        act_gb = (base_act_bytes * 0.25) / 1e9
    else:
        act_gb = (base_act_bytes * 1.2) / 1e9

    # 3. LoRA trainable parameters & AdamW optimizer states
    num_targets = len(cfg.target_modules)
    trainable_params = num_layers * num_targets * 2 * cfg.lora_rank * hidden_size
    lora_opt_gb = (trainable_params * 14) / 1e9

    # 4. Context & KV Cache memory
    kv_cache_gb = (2 * num_layers * hidden_size * cfg.seq_len * cfg.batch_size * 2) / 1e9

    # Sum components with 15% allocator overhead
    total_raw = weight_gb + act_gb + lora_opt_gb + kv_cache_gb + 0.35  # CUDA context overhead
    return round(total_raw * 1.15, 2)


@dataclass
class FeasibilityReport:
    is_feasible: bool
    hardware_fit: bool
    data_fit: bool
    estimated_vram_gb: float
    available_vram_gb: float
    recommended_config: TrainingConfig
    warnings: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


def check_feasibility(
    hw: Any,
    model: Any,
    data: Optional[Any] = None,
    goal: Optional[Any] = None,
    cfg: Optional[TrainingConfig] = None,
    allow_layer_streaming: bool = False,
) -> FeasibilityReport:
    """Explicit Dual Gate (Hardware Fit × Data Fit) validation."""
    if cfg is None:
        cfg = TrainingConfig()

    vram_gb = getattr(hw, "vram_gb", 0.0)
    has_gpu = getattr(hw, "has_gpu", vram_gb > 0)
    ram_gb = getattr(hw, "ram_gb", 16.0)

    est_vram = estimate_vram_gb(model, cfg)
    available_vram = vram_gb if has_gpu else ram_gb

    warnings: List[str] = []
    reasons: List[str] = []

    # Hardware Fit Check
    if has_gpu:
        hardware_fit = est_vram <= (vram_gb * 0.95)
        if not hardware_fit:
            if allow_layer_streaming or getattr(cfg, "stream_layers", False):
                stream_cfg = TrainingConfig(
                    quantization=cfg.quantization,
                    lora_rank=cfg.lora_rank,
                    lora_alpha=cfg.lora_alpha,
                    seq_len=cfg.seq_len,
                    batch_size=cfg.batch_size,
                    grad_accum=cfg.grad_accum,
                    grad_checkpointing=cfg.grad_checkpointing,
                    stream_layers=True,
                )
                stream_est_vram = estimate_vram_gb(model, stream_cfg)
                if stream_est_vram <= (vram_gb * 0.95):
                    cfg = stream_cfg
                    est_vram = stream_est_vram
                    hardware_fit = True
                    reasons.append(f"Layer Streaming activated: {est_vram} GB estimated / {vram_gb} GB available (8B model on {vram_gb:.1f}GB GPU).")
                else:
                    warnings.append(f"Estimated VRAM ({est_vram} GB) exceeds available VRAM ({vram_gb} GB).")
            else:
                warnings.append(f"Estimated VRAM ({est_vram} GB) exceeds available VRAM ({vram_gb} GB).")
        else:
            reasons.append(f"VRAM budget satisfied: {est_vram} GB estimated / {vram_gb} GB available.")
    else:
        hardware_fit = est_vram <= (ram_gb * 0.70)
        if not hardware_fit:
            warnings.append(f"Estimated CPU memory ({est_vram} GB) exceeds safe system RAM ({ram_gb} GB).")
        else:
            reasons.append(f"System RAM budget satisfied on CPU: {est_vram} GB / {ram_gb} GB available.")

    # Data Fit Check
    data_fit = True
    if data is not None:
        avg_tokens = getattr(data, "avg_tokens", getattr(data, "tokens_approx", 512))
        max_tokens = getattr(data, "max_tokens", 0)
        context_len = _get_model_attribute(model, "context_length", 4096)
        if avg_tokens > context_len:
            warnings.append(f"Average sample length ({avg_tokens} tokens) exceeds model context ({context_len}).")
            data_fit = False
        elif max_tokens > context_len:
            warnings.append(f"Maximum sample length ({max_tokens} tokens) exceeds model context ({context_len}) — will be truncated to fit.")
            reasons.append("Average dataset token lengths fit comfortably within model context.")
        else:
            reasons.append("Dataset token lengths fit comfortably within model context.")

    is_feasible = hardware_fit and data_fit
    return FeasibilityReport(
        is_feasible=is_feasible,
        hardware_fit=hardware_fit,
        data_fit=data_fit,
        estimated_vram_gb=est_vram,
        available_vram_gb=available_vram,
        recommended_config=cfg,
        warnings=warnings,
        reasons=reasons,
    )


@dataclass
class FeasibilityResult:
    overall: str                    # "PASS" | "FAIL"
    estimated_vram_gb: float
    reasoning: str
    report: FeasibilityReport


def run_feasibility(hw: Any, model: Any, data: Optional[Any] = None) -> FeasibilityResult:
    """Run dual-gate feasibility assessment, returning standardized FeasibilityResult."""
    report = check_feasibility(hw, model, data=data)
    overall = "PASS" if report.is_feasible else "FAIL"
    reasoning_text = "; ".join(report.reasons if report.is_feasible else report.warnings) or "Feasibility evaluated."
    return FeasibilityResult(
        overall=overall,
        estimated_vram_gb=report.estimated_vram_gb,
        reasoning=reasoning_text,
        report=report,
    )
