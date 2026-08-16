from dataclasses import dataclass, field
from pathlib import Path
from ..hardware.detector import detect_hardware, HardwareReport
from ..data.validator import validate_data, DataReport
from .registry import get_registry_models
from .schema import RegistryModel
from ..core.config import ProjectConfig

@dataclass
class ModelRecommendation:
    model: RegistryModel
    score: int
    method: str
    fits_vram: bool
    reasons: list[str] = field(default_factory=list)

def recommend_models(
    hardware: HardwareReport,
    data_report: DataReport,
    models: list[RegistryModel] | None = None
) -> list[ModelRecommendation]:
    """Score and rank registered models based on hardware and dataset characteristics."""
    if models is None:
        models = get_registry_models()

    recommendations: list[ModelRecommendation] = []
    
    # Available effective VRAM (0 means CPU/integrated, or not detected)
    has_gpu = hardware.vram_gb > 0
    effective_vram = hardware.vram_gb if has_gpu else (hardware.ram_gb / 2.0)

    for m in models:
        score = 100
        reasons = []
        fits_vram = m.vram_min <= (hardware.vram_gb if has_gpu else hardware.ram_gb)
        
        # Determine training method recommendation
        if has_gpu and hardware.vram_gb >= m.vram_min:
            method = "QLoRA"
            reasons.append(f"Fits in {hardware.vram_gb} GB VRAM with QLoRA")
        elif not has_gpu and hardware.ram_gb >= m.vram_min:
            method = "LoRA"
            reasons.append(f"Runs on CPU with {hardware.ram_gb} GB System RAM")
        else:
            method = "QLoRA"
            reasons.append(f"Requires ~{m.vram_min} GB (Current: {hardware.vram_gb if has_gpu else hardware.ram_gb} GB)")

        # Penalty for VRAM mismatch
        if not fits_vram:
            score -= 50
            reasons.append("Hardware resources below minimum recommendation")

        # Size appropriateness based on tokens
        # For small datasets (< 100k tokens), smaller models (<= 3B) prevent overfitting and train faster
        if data_report.tokens_approx < 100_000:
            if "0.5B" in m.parameters or "1B" in m.parameters or "1.5B" in m.parameters:
                score += 10
                reasons.append("Optimal parameter size for compact dataset")
            elif "7B" in m.parameters or "8B" in m.parameters:
                score -= 20
                reasons.append("Large model for small dataset size (risk of overfitting)")
        elif data_report.tokens_approx > 1_000_000:
            if "3B" in m.parameters or "7B" in m.parameters or "8B" in m.parameters:
                score += 10
                reasons.append("High parameter capacity suited for large dataset")

        # Hardware Tier boost
        if hardware.tier in ("T2", "T3") and ("3B" in m.parameters or "7B" in m.parameters):
            score += 5
        elif hardware.tier in ("T0", "T1") and ("0.5B" in m.parameters or "1.5B" in m.parameters or "3B" in m.parameters):
            score += 5

        # Normalize score
        score = max(0, min(100, score))
        recommendations.append(
            ModelRecommendation(
                model=m,
                score=score,
                method=method,
                fits_vram=fits_vram,
                reasons=reasons
            )
        )

    recommendations.sort(key=lambda x: (x.fits_vram, x.score), reverse=True)
    return recommendations

from ..data.manager import resolve_dataset_source

def get_top_recommendation(root: Path, cfg: ProjectConfig) -> tuple[ModelRecommendation | None, HardwareReport, DataReport]:
    """Inspect project dataset and system hardware, returning top recommendation."""
    data_dir = resolve_dataset_source(root, cfg)
    report = validate_data(data_dir)
    hw = detect_hardware()
    models = get_registry_models()
    
    recs = recommend_models(hw, report, models)
    top_rec = recs[0] if recs else None
    return top_rec, hw, report
