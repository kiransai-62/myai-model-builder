"""
MYAI Hardware Intelligence & Model Recommendation Engine (15-Point Matrix).

Evaluates models against:
1. System Compatibility (8 factors: VRAM, RAM, GPU Compute, CPU, Storage, Throughput, Context, Runtime)
2. Dataset Fit (Token volume vs. parameter capacity)
3. Task Fit (Code, Reasoning, Chat, Extraction, Domain QA)
4. Training Fit (LoRA, QLoRA, Layer Streaming, DPO)
5. Deployment Fit (Edge, Fast latency, Cloud server)

Assigns 4-tier verdicts (RECOMMENDED, COMPATIBLE, POSSIBLE, UNSUPPORTED) with
explainable fit breakdowns, dynamic context profiles, and alternative model suggestions.
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
from ..hardware.detector import detect_hardware, HardwareReport
from ..hardware.memory_calc import (
    calculate_dynamic_memory_profile,
    evaluate_context_profiles,
    DynamicMemoryProfile,
)
from ..data.validator import validate_data, DataReport
from .registry import get_registry_models
from .schema import RegistryModel
from ..core.config import ProjectConfig
from ..core.goal import GoalProfile, TaskType
from ..registry.scorer import (
    SystemCompatibilityScorer,
    ModelRecommenderScorer,
    SystemCompatibilityBreakdown,
    RecommendationScoreResult,
)


class CompatibilityVerdict(str, Enum):
    RECOMMENDED = "RECOMMENDED"    # ⭐ Ample headroom, optimal fit
    COMPATIBLE = "COMPATIBLE"      # ✅ Meets minimum specs stably
    POSSIBLE = "POSSIBLE"          # ⚠️ Operable via layer streaming or low context
    UNSUPPORTED = "UNSUPPORTED"    # ❌ Hardware insufficient


# Backward compatibility aliases
HardwareFitBreakdown = SystemCompatibilityBreakdown


@dataclass
class ExplainableFitScores:
    hardware_fit: float = 100.0
    dataset_fit: float = 100.0
    task_fit: float = 100.0
    training_fit: float = 100.0
    deployment_fit: float = 100.0
    overall_composite: float = 100.0
    confidence: float = 0.95


@dataclass
class ModelRecommendation:
    model: RegistryModel
    score: int
    method: str
    fits_vram: bool
    verdict: CompatibilityVerdict = CompatibilityVerdict.COMPATIBLE
    reasons: List[str] = field(default_factory=list)
    confidence: float = 0.95
    predicted_tokens_per_sec: float = 30.0
    fit_breakdown: Optional[ExplainableFitScores] = None
    hw_breakdown: Optional[SystemCompatibilityBreakdown] = None
    dynamic_memory: Optional[DynamicMemoryProfile] = None
    recommended_context: int = 4096
    context_profiles: Dict[int, Tuple[float, str]] = field(default_factory=dict)
    why_this_model: List[str] = field(default_factory=list)
    alternative_model: Optional["ModelRecommendation"] = None

    @property
    def verdict_badge(self) -> str:
        if self.verdict == CompatibilityVerdict.RECOMMENDED:
            return "[bold green]⭐ RECOMMENDED[/bold green]"
        elif self.verdict == CompatibilityVerdict.COMPATIBLE:
            return "[green]✅ COMPATIBLE[/green]"
        elif self.verdict == CompatibilityVerdict.POSSIBLE:
            return "[yellow]⚠️ POSSIBLE[/yellow]"
        return "[red]❌ UNSUPPORTED[/red]"


def compute_8_factor_hardware_score(
    hardware: HardwareReport,
    model: RegistryModel,
    mem_profile: DynamicMemoryProfile,
) -> Tuple[float, SystemCompatibilityBreakdown]:
    """Computes weighted 8-factor system compatibility score."""
    return SystemCompatibilityScorer.score(hardware, model, mem_profile)


def recommend_models(
    hardware: HardwareReport,
    data_report: DataReport,
    models: Optional[List[RegistryModel]] = None,
    goal: Optional[GoalProfile] = None,
) -> List[ModelRecommendation]:
    """
    Score and rank registered models based on 8-factor system compatibility,
    dataset alignment, goal characteristics, and explainability breakdown.
    """
    if models is None:
        models = get_registry_models()

    recommendations: List[ModelRecommendation] = []
    has_gpu = hardware.vram_gb > 0

    for m in models:
        # 1. Training method determination
        fits_resident = m.vram_min <= (hardware.vram_gb if has_gpu else hardware.ram_gb)
        fits_streaming = has_gpu and hardware.vram_gb >= 3.5 and m.params_b <= 8.0 and "layer_streaming" in m.methods
        fits_vram = fits_resident or fits_streaming

        if not has_gpu:
            method = "LoRA"
        elif hardware.vram_gb >= m.vram_min:
            method = "QLoRA"
        elif fits_streaming:
            method = "Exact Layer Streaming"
        else:
            method = "QLoRA"

        # 2. Dynamic Memory & Multi-Tier Context Profile Calculation
        mem_profile = calculate_dynamic_memory_profile(
            params_total_b=m.parameters_billions,
            params_active_b=m.active_parameters_billions,
            num_layers=m.num_layers,
            hidden_size=m.hidden_size,
            quant_format="INT4",
            context_length=min(m.context_length, 4096),
            batch_size=2,
            is_training=True,
            training_method="layer_streaming" if method == "Exact Layer Streaming" else method.lower(),
            available_vram_gb=hardware.vram_gb,
            available_ram_gb=hardware.ram_gb,
            gpu_tier=hardware.tier,
        )

        # 3. Multi-Dimension Recommendation Scoring
        score_res: RecommendationScoreResult = ModelRecommenderScorer.score_model(
            hardware=hardware,
            data_report=data_report,
            model=m,
            mem_profile=mem_profile,
            goal=goal,
            fits_vram=fits_vram,
        )

        # 4. 4-Tier Verdict Assignment
        if not fits_vram or score_res.system_compatibility < 40.0:
            verdict = CompatibilityVerdict.UNSUPPORTED
        elif fits_streaming and not fits_resident:
            verdict = CompatibilityVerdict.POSSIBLE
        elif score_res.overall_recommendation >= 85.0 and mem_profile.headroom_gb >= 1.0:
            verdict = CompatibilityVerdict.RECOMMENDED
        elif score_res.overall_recommendation >= 60.0:
            verdict = CompatibilityVerdict.COMPATIBLE
        else:
            verdict = CompatibilityVerdict.POSSIBLE

        fit_breakdown = ExplainableFitScores(
            hardware_fit=score_res.system_compatibility,
            dataset_fit=score_res.dataset_fit,
            task_fit=score_res.task_fit,
            training_fit=score_res.training_fit,
            deployment_fit=score_res.deployment_fit,
            overall_composite=score_res.overall_recommendation,
            confidence=score_res.confidence,
        )

        # "Why this model?" explainability rationale
        why_bullets: List[str] = []
        if has_gpu and fits_resident:
            why_bullets.append(f"Fits in {hardware.vram_gb} GB VRAM comfortably with {method}")
        elif has_gpu and fits_streaming:
            why_bullets.append(f"Operates in 4GB VRAM via Exact Layer Streaming (~3.32 GB peak)")
        elif not has_gpu:
            why_bullets.append(f"Runs on CPU with {hardware.ram_gb} GB System RAM")

        why_bullets.extend(score_res.reasons[:3])
        if mem_profile.estimated_tokens_per_sec >= 20.0:
            why_bullets.append(f"Fast inference throughput (~{mem_profile.estimated_tokens_per_sec} tok/s)")

        recommendations.append(
            ModelRecommendation(
                model=m,
                score=int(score_res.overall_recommendation),
                method=method,
                fits_vram=fits_vram,
                verdict=verdict,
                reasons=score_res.reasons,
                confidence=score_res.confidence,
                predicted_tokens_per_sec=mem_profile.estimated_tokens_per_sec,
                fit_breakdown=fit_breakdown,
                hw_breakdown=score_res.system_breakdown,
                dynamic_memory=mem_profile,
                recommended_context=mem_profile.recommended_context,
                context_profiles=mem_profile.context_profiles,
                why_this_model=why_bullets,
            )
        )

    # Sort primarily by: verdict precedence (RECOMMENDED > COMPATIBLE > POSSIBLE > UNSUPPORTED) then composite score
    verdict_rank = {
        CompatibilityVerdict.RECOMMENDED: 4,
        CompatibilityVerdict.COMPATIBLE: 3,
        CompatibilityVerdict.POSSIBLE: 2,
        CompatibilityVerdict.UNSUPPORTED: 1,
    }
    recommendations.sort(key=lambda x: (x.fits_vram, verdict_rank.get(x.verdict, 0), x.score), reverse=True)

    # Attach alternative model suggestion to top recommendation
    if len(recommendations) > 1:
        top = recommendations[0]
        # Find first viable alternative (e.g. next size up or down)
        alt = next((r for r in recommendations[1:] if r.fits_vram), recommendations[1])
        top.alternative_model = alt

    return recommendations


from ..data.manager import resolve_dataset_source

CATALOG = get_registry_models()


@dataclass
class RecommendationResult:
    model: RegistryModel
    score: int
    reasoning: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    verdict: CompatibilityVerdict = CompatibilityVerdict.COMPATIBLE
    fit_breakdown: Optional[ExplainableFitScores] = None
    hw_breakdown: Optional[SystemCompatibilityBreakdown] = None
    predicted_tok_per_sec: float = 30.0
    recommended_context: int = 4096
    why_this_model: List[str] = field(default_factory=list)
    alternative_model: Optional[ModelRecommendation] = None


def recommend_model(
    hardware: HardwareReport,
    goal: GoalProfile,
    data_summary: Any = None,
    models: Optional[List[RegistryModel]] = None,
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
        return RecommendationResult(
            model=default_m, score=50, reasoning=["Default fallback model selected."],
            verdict=CompatibilityVerdict.COMPATIBLE
        )
    top = recs[0]
    return RecommendationResult(
        model=top.model,
        score=top.score,
        reasoning=top.reasons,
        warnings=[r for r in top.reasons if "below" in r.lower() or "exceeds" in r.lower()],
        verdict=top.verdict,
        fit_breakdown=top.fit_breakdown,
        hw_breakdown=top.hw_breakdown,
        predicted_tok_per_sec=top.predicted_tokens_per_sec,
        recommended_context=top.recommended_context,
        why_this_model=top.why_this_model,
        alternative_model=top.alternative_model,
    )


def get_top_recommendation(
    root: Path, cfg: ProjectConfig
) -> Tuple[Optional[ModelRecommendation], HardwareReport, DataReport]:
    """Inspect project dataset, system hardware, and goal profile, returning top recommendation."""
    data_dir = resolve_dataset_source(root, cfg)
    report = validate_data(data_dir)
    hw = detect_hardware()
    models = get_registry_models()
    
    recs = recommend_models(hw, report, models, goal=cfg.goal)
    top_rec = recs[0] if recs else None
    return top_rec, hw, report
