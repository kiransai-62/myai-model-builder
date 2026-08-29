from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union
from ..hardware.detector import detect_hardware, HardwareReport
from ..data.validator import validate_data, DataReport
from .registry import get_registry_models
from .schema import RegistryModel
from ..core.config import ProjectConfig
from ..core.goal import GoalProfile, TaskType


@dataclass
class ModelRecommendation:
    model: RegistryModel
    score: int
    method: str
    fits_vram: bool
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.95


def recommend_models(
    hardware: HardwareReport,
    data_report: DataReport,
    models: list[RegistryModel] | None = None,
    goal: GoalProfile | None = None,
) -> list[ModelRecommendation]:
    """Score and rank registered models based on hardware, dataset, and goal characteristics."""
    if models is None:
        models = get_registry_models()

    recommendations: list[ModelRecommendation] = []
    
    # Available effective VRAM (0 means CPU/integrated, or not detected)
    has_gpu = hardware.vram_gb > 0
    effective_vram = hardware.vram_gb if has_gpu else (hardware.ram_gb / 2.0)

    for m in models:
        score = 100
        reasons = []
        
        # Dual-gate check: Resident VRAM vs Layer Streaming vs CPU
        fits_resident = m.vram_min <= (hardware.vram_gb if has_gpu else hardware.ram_gb)
        fits_streaming = has_gpu and hardware.vram_gb >= 3.5 and m.params_b <= 8.0 and "layer_streaming" in m.methods
        fits_vram = fits_resident or fits_streaming
        
        # Determine training method recommendation
        if not has_gpu:
            method = "LoRA"
            if hardware.ram_gb >= m.vram_min:
                reasons.append(f"Runs on CPU with {hardware.ram_gb} GB System RAM")
            else:
                reasons.append(f"Requires ~{m.vram_min} GB (Current: {hardware.ram_gb} GB)")
        elif hardware.vram_gb >= m.vram_min:
            method = "QLoRA"
            reasons.append(f"Fits in {hardware.vram_gb} GB VRAM with QLoRA")
        elif fits_streaming:
            method = "Exact Layer Streaming"
            reasons.append(f"Fits in {hardware.vram_gb} GB VRAM with Layer Streaming (~3.32 GB peak)")
        else:
            method = "QLoRA"
            reasons.append(f"Requires ~{m.vram_min} GB (Current: {hardware.vram_gb} GB)")

        # Penalty for hardware resource mismatch
        if not fits_vram:
            score -= 50
            reasons.append("Hardware resources below minimum recommendation")
        elif fits_streaming and not fits_resident:
            # Slight modifier for streaming overhead vs resident
            score -= 5
            reasons.append("Layer streaming activated for low-VRAM GPU")

        # Size appropriateness based on tokens
        # For small datasets (< 100k tokens), smaller models (<= 3B) prevent overfitting and train faster
        if data_report.tokens_approx < 100_000:
            if m.params_b <= 1.5:
                score += 10
                reasons.append("Optimal parameter size for compact dataset")
            elif m.params_b >= 7.0:
                score -= 20
                reasons.append("Large model for small dataset size (risk of overfitting)")
        elif data_report.tokens_approx > 1_000_000:
            if m.params_b >= 3.0 and m.params_b <= 14.0:
                score += 10
                reasons.append("High parameter capacity suited for large dataset")

        # Hardware Tier boost
        if hardware.tier in ("T2", "T3") and m.params_b in (3.0, 7.0, 8.0, 14.0):
            score += 5
        elif hardware.tier in ("T0", "T1") and m.params_b <= 3.0:
            score += 5

        # Goal-driven adjustments
        if goal is not None:
            if goal.task == TaskType.CODE:
                if "coder" in m.id.lower() or "code" in m.name.lower() or "phi-4" in m.id.lower():
                    score += 25
                    reasons.append("Optimized for code synthesis and programming tasks")
                else:
                    score -= 10
                    reasons.append("General base model; code-specialized model preferred")
            elif goal.task in (TaskType.CHAT, TaskType.DOMAIN_QA):
                if "instruct" in m.id.lower() or "chat" in m.id.lower():
                    score += 15
                    reasons.append("Instruction-tuned architecture matches conversational/Q&A goal")
            elif goal.task == TaskType.REASONING:
                if m.has_reasoning or "distill" in m.id.lower() or "r1" in m.id.lower() or "phi" in m.id.lower():
                    score += 20
                    reasons.append("High-depth reasoning architecture matches analytical goal")
            
            if goal.target_deployment == "edge":
                if m.parameters_billions <= 1.5:
                    score += 15
                    reasons.append("Lightweight footprint ideal for edge deployment")
                elif m.parameters_billions >= 7.0:
                    score -= 25
                    reasons.append("Exceeds optimal parameters for edge deployment")
                    
            if goal.latency_priority == "fast":
                if m.parameters_billions <= 3.0:
                    score += 10
                    reasons.append("Low parameter latency matches fast response priority")

        # Normalize score
        score = max(0, min(100, score))
        recommendations.append(
            ModelRecommendation(
                model=m,
                score=score,
                method=method,
                fits_vram=fits_vram,
                reasons=reasons,
                confidence=m.confidence,
            )
        )

    recommendations.sort(key=lambda x: (x.fits_vram, x.score), reverse=True)
    return recommendations


from ..data.manager import resolve_dataset_source

CATALOG = get_registry_models()


@dataclass
class RecommendationResult:
    model: RegistryModel
    score: int
    reasoning: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def recommend_model(
    hardware: HardwareReport,
    goal: GoalProfile,
    data_summary: Any = None,
    models: list[RegistryModel] | None = None,
) -> RecommendationResult:
    global CATALOG
    if models is None:
        models = get_registry_models()
        CATALOG = models

    if data_summary is None:
        data_report = DataReport()
    elif isinstance(data_summary, DataReport):
        data_report = data_summary
    else:
        data_report = DataReport(
            examples=getattr(data_summary, "num_samples", 0),
            tokens_approx=getattr(data_summary, "tokens_approx", 50000),
            duplicates=getattr(data_summary, "exact_duplicates", 0),
        )

    recs = recommend_models(hardware, data_report, models, goal=goal)
    if not recs:
        default_m = models[0] if models else RegistryModel(
            "llama-3-8b-instruct", "Llama 3 8B Instruct", "8B", 8.0, ["QLoRA"],
            "meta-llama/Meta-Llama-3-8B-Instruct", "Llama-3"
        )
        return RecommendationResult(model=default_m, score=50, reasoning=["Default fallback model selected."])
    top = recs[0]
    return RecommendationResult(
        model=top.model,
        score=top.score,
        reasoning=top.reasons,
        warnings=[r for r in top.reasons if "below" in r.lower() or "exceeds" in r.lower()]
    )


def get_top_recommendation(root: Path, cfg: ProjectConfig) -> tuple[ModelRecommendation | None, HardwareReport, DataReport]:
    """Inspect project dataset, system hardware, and goal profile, returning top recommendation."""
    data_dir = resolve_dataset_source(root, cfg)
    report = validate_data(data_dir)
    hw = detect_hardware()
    models = get_registry_models()
    
    recs = recommend_models(hw, report, models, goal=cfg.goal)
    top_rec = recs[0] if recs else None
    return top_rec, hw, report
