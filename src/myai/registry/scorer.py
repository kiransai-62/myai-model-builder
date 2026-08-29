"""
MYAI System Compatibility & Multi-Dimension Recommendation Scorer.

Calculates:
1. System Compatibility Score (8 factors: VRAM, RAM, GPU Compute, CPU, Storage, Throughput, Context, Runtime)
2. Dataset Fit Score (Token volume vs. Model parameter capacity)
3. Task & Capability Fit Score (Code, Reasoning, Chat, Extraction, Domain QA)
4. Training Fit Score (LoRA, QLoRA, Layer Streaming, DPO compatibility)
5. Deployment Fit Score (Edge, Fast latency, Cloud server)
6. Overall Recommendation Score (Goal-weighted composite)
"""
from dataclasses import dataclass
from typing import Tuple, List, Optional
from ..hardware.detector import HardwareReport
from ..hardware.memory_calc import DynamicMemoryProfile
from ..models.schema import RegistryModel
from ..data.validator import DataReport
from ..core.goal import GoalProfile, TaskType


@dataclass
class SystemCompatibilityBreakdown:
    vram_score: float = 100.0        # Weight: 30%
    ram_score: float = 100.0         # Weight: 15%
    gpu_compute_score: float = 100.0 # Weight: 15%
    cpu_score: float = 100.0         # Weight: 10%
    storage_score: float = 100.0     # Weight: 10%
    throughput_score: float = 100.0  # Weight: 10%
    context_score: float = 100.0     # Weight: 5%
    runtime_score: float = 100.0     # Weight: 5%
    composite_system_score: float = 100.0


@dataclass
class RecommendationScoreResult:
    system_compatibility: float
    dataset_fit: float
    task_fit: float
    training_fit: float
    deployment_fit: float
    overall_recommendation: float
    confidence: float
    reasons: List[str]
    system_breakdown: SystemCompatibilityBreakdown


class SystemCompatibilityScorer:
    """Computes the 8-factor system compatibility score."""

    @staticmethod
    def score(
        hardware: HardwareReport,
        model: RegistryModel,
        mem_profile: DynamicMemoryProfile,
    ) -> Tuple[float, SystemCompatibilityBreakdown]:
        has_gpu = hardware.vram_gb > 0

        # 1. VRAM Score (30%)
        if has_gpu:
            if mem_profile.headroom_gb >= 2.0:
                vram_s = 100.0
            elif mem_profile.headroom_gb >= 0.5:
                vram_s = 85.0
            elif mem_profile.headroom_gb >= 0.0:
                vram_s = 65.0
            else:
                vram_s = max(0.0, 50.0 + (mem_profile.headroom_gb * 10.0))
        else:
            vram_s = 70.0  # CPU simulation mode

        # 2. RAM Score (15%)
        ram_needed = model.training.training_ram_gb
        if hardware.ram_gb >= ram_needed * 1.5:
            ram_s = 100.0
        elif hardware.ram_gb >= ram_needed:
            ram_s = 85.0
        elif hardware.ram_gb >= ram_needed * 0.7:
            ram_s = 60.0
        else:
            ram_s = 30.0

        # 3. GPU Compute Score (15%)
        if has_gpu:
            tier_scores = {"T3": 100.0, "T2": 90.0, "T1": 75.0, "T0": 55.0}
            gpu_s = tier_scores.get(hardware.tier, 70.0)
        else:
            gpu_s = 40.0

        # 4. CPU Score (10%)
        cpu_cores = 8
        try:
            cpu_cores = int(hardware.cpu.split()[0])
        except Exception:
            pass
        if cpu_cores >= model.cpu.rec_cores:
            cpu_s = 100.0
        elif cpu_cores >= model.cpu.min_cores:
            cpu_s = 80.0
        else:
            cpu_s = 50.0

        # 5. Storage Score (10%)
        storage_needed = model.training.workspace_storage_gb
        if hardware.disk_gb >= storage_needed * 3.0:
            storage_s = 100.0
        elif hardware.disk_gb >= storage_needed:
            storage_s = 80.0
        else:
            storage_s = 20.0

        # 6. Throughput Score (10%)
        tok_s = mem_profile.estimated_tokens_per_sec
        if tok_s >= 40.0:
            thru_s = 100.0
        elif tok_s >= 20.0:
            thru_s = 85.0
        elif tok_s >= 10.0:
            thru_s = 70.0
        else:
            thru_s = 45.0

        # 7. Context Capacity Score (5%)
        if model.context_length >= 32768:
            ctx_s = 100.0
        elif model.context_length >= 8192:
            ctx_s = 85.0
        else:
            ctx_s = 70.0

        # 8. Runtime Support Score (5%)
        runtime_s = 95.0 if ("QLoRA" in model.methods or "LoRA" in model.methods) else 70.0

        composite_sys = (
            (vram_s * 0.30) +
            (ram_s * 0.15) +
            (gpu_s * 0.15) +
            (cpu_s * 0.10) +
            (storage_s * 0.10) +
            (thru_s * 0.10) +
            (ctx_s * 0.05) +
            (runtime_s * 0.05)
        )

        breakdown = SystemCompatibilityBreakdown(
            vram_score=round(vram_s, 1),
            ram_score=round(ram_s, 1),
            gpu_compute_score=round(gpu_s, 1),
            cpu_score=round(cpu_s, 1),
            storage_score=round(storage_s, 1),
            throughput_score=round(thru_s, 1),
            context_score=round(ctx_s, 1),
            runtime_score=round(runtime_s, 1),
            composite_system_score=round(composite_sys, 1),
        )
        return round(composite_sys, 1), breakdown


class ModelRecommenderScorer:
    """Calculates full multi-dimensional recommendation scoring."""

    @classmethod
    def score_model(
        cls,
        hardware: HardwareReport,
        data_report: DataReport,
        model: RegistryModel,
        mem_profile: DynamicMemoryProfile,
        goal: Optional[GoalProfile] = None,
        fits_vram: bool = True,
    ) -> RecommendationScoreResult:
        reasons: List[str] = []

        # 1. System Compatibility Score (8 factors)
        sys_score, sys_breakdown = SystemCompatibilityScorer.score(hardware, model, mem_profile)

        # 2. Dataset Size & Capacity Fit Score (0-100)
        # Deeply evaluates model parameter capacity vs. dataset tokens & example volume
        tokens = data_report.tokens_approx
        examples = data_report.examples
        params = model.params_b

        if tokens < 100_000 or examples < 1000:
            # Compact dataset: small models excel, large models risk severe overfitting
            if params <= 1.5:
                data_score = 95.0
                reasons.append("Optimal parameter capacity for compact dataset")
            elif params <= 4.0:
                data_score = 88.0
                reasons.append("Good capacity for small-to-medium dataset")
            elif params <= 8.0:
                data_score = 75.0
            else:
                data_score = 55.0
                reasons.append("Large model for compact dataset (risk of memorization/overfitting)")
        elif tokens > 1_000_000 or examples > 20_000:
            # Large dataset: larger models excel, tiny models risk underfitting
            if params >= 7.0:
                data_score = 98.0
                reasons.append("High parameter capacity suited for large dataset")
            elif params >= 3.0:
                data_score = 90.0
            else:
                data_score = 65.0
                reasons.append("Small model capacity may underfit large dataset")
        else:
            # Mid-sized dataset
            if 1.0 <= params <= 14.0:
                data_score = 92.0
                reasons.append("Balanced model capacity for dataset volume")
            else:
                data_score = 80.0

        # 3. Task & Domain Fit Score (0-100)
        task_score = 85.0
        if goal is not None:
            if goal.task == TaskType.CODE:
                if "coder" in model.id.lower() or "code" in model.name.lower() or "phi-4" in model.id.lower():
                    task_score = 98.0
                    reasons.append("Optimized for code synthesis and programming tasks")
                else:
                    task_score = 72.0
            elif goal.task in (TaskType.CHAT, TaskType.DOMAIN_QA):
                if "instruct" in model.id.lower() or "chat" in model.id.lower():
                    task_score = 95.0
                    reasons.append("Instruction-tuned architecture matches conversational/Q&A goal")
            elif goal.task == TaskType.REASONING:
                if model.has_reasoning or "distill" in model.id.lower() or "r1" in model.id.lower():
                    task_score = 98.0
                    reasons.append("Deep reasoning architecture matches analytical goal")
            elif goal.task == TaskType.EXTRACTION:
                if model.capabilities.get("structured_json", True):
                    task_score = 92.0

        # 4. Training Fit Score (0-100)
        training_score = 92.0 if fits_vram else 40.0
        if "layer_streaming" in model.methods:
            reasons.append("Supports Exact Layer Streaming for low-VRAM GPUs")

        # 5. Deployment Fit Score (0-100)
        deployment_score = 88.0
        if goal is not None:
            if goal.target_deployment == "edge":
                if model.parameters_billions <= 1.5:
                    deployment_score = 98.0
                    reasons.append("Lightweight footprint ideal for edge deployment")
                elif model.parameters_billions >= 7.0:
                    deployment_score = 50.0
                    reasons.append("Exceeds optimal parameters for edge deployment")
            if goal.latency_priority == "fast":
                if model.parameters_billions <= 3.0:
                    deployment_score = min(100.0, deployment_score + 5.0)

        # 6. Overall Recommendation Score (Composite)
        overall_rec = round(
            (sys_score * 0.35) +
            (data_score * 0.20) +
            (task_score * 0.20) +
            (training_score * 0.15) +
            (deployment_score * 0.10),
            1
        )
        overall_rec = max(0.0, min(100.0, overall_rec))
        confidence = round(min(0.99, (overall_rec / 100.0) * model.confidence), 2)

        return RecommendationScoreResult(
            system_compatibility=sys_score,
            dataset_fit=data_score,
            task_fit=task_score,
            training_fit=training_score,
            deployment_fit=deployment_score,
            overall_recommendation=overall_rec,
            confidence=confidence,
            reasons=reasons,
            system_breakdown=sys_breakdown,
        )
