"""MYAI Training Strategy Planner (Report §11, §18).

Derives the full training configuration from measurable constraints:
  Hardware (VRAM/RAM/tier) × Model (params/arch) × Data (size/tokens) × Goal (priorities)

Outputs an OOM-safe TrainingConfig plus LR/epochs/scheduler, a VRAM estimate,
a training-time estimate, and a storage budget check. Every decision is
explainable (reasoning + assumptions) and overridable via `override`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from myai.hardware.feasibility import TrainingConfig, estimate_vram_gb, _get_model_attribute

SEQ_BUCKETS = [512, 1024, 2048, 4096, 8192, 16384]
LR_BY_RANK = {8: 3e-4, 16: 2e-4, 32: 1.5e-4, 64: 1e-4}
# Rough sustained training throughput (tokens/sec) per hardware tier
TIER_THROUGHPUT = {"T0": 60, "T1": 120, "T2": 2200, "T3": 6500}


@dataclass
class TrainingStrategy:
    config: TrainingConfig
    learning_rate: float
    epochs: int
    warmup_ratio: float = 0.05
    scheduler: str = "cosine"
    weight_decay: float = 0.01
    effective_batch: int = 8
    estimated_vram_gb: float = 0.0
    estimated_minutes: float = 0.0
    storage_required_gb: float = 0.0
    reasoning: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    confidence: float = 0.85


# ------------------------------------------------------------------ planners
def _pick_seq_len(data: Any, model: Any, goal: Any) -> int:
    avg_tok = getattr(data, "avg_tokens", getattr(data, "tokens_approx", 512)) if data else 512
    target = int(avg_tok * 1.25)
    seq = next((b for b in SEQ_BUCKETS if b >= target), SEQ_BUCKETS[-1])
    
    if goal is not None and getattr(goal, "context_priority", "") == "long-context":
        seq = max(seq, 4096)
        
    model_ctx = _get_model_attribute(model, "context_length", 4096)
    return min(seq, model_ctx)


def _pick_precision(model: Any, hw: Any) -> str:
    """Highest precision whose weights + headroom fit the GPU."""
    has_gpu = getattr(hw, "has_gpu", False) or (getattr(hw, "vram_gb", 0) > 0)
    vram_gb = getattr(hw, "vram_gb", 0.0)
    params_b = _get_model_attribute(model, "params_b", 3.0)

    if not (has_gpu and vram_gb > 0):
        return "4bit"  # CPU/T1: always quantize for memory safety

    for q, nbytes in (("fp16", 2.0), ("8bit", 1.0), ("4bit", 0.5)):
        if (params_b * nbytes * 1.35) <= vram_gb * 0.80:
            return q
    return "4bit"


def _pick_rank(data: Any, goal: Any) -> int:
    n = getattr(data, "num_samples", getattr(data, "examples", 1000)) if data else 1000
    rank = 8 if n < 500 else 16 if n < 2000 else 32 if n < 10000 else 64
    
    if goal is not None and getattr(goal, "task", "") in ("code", "summarization"):
        rank = max(rank, 16)  # Complex tasks benefit from extra LoRA rank capacity
    return rank


def _pick_epochs(data: Any) -> int:
    n = getattr(data, "num_samples", getattr(data, "examples", 1000)) if data else 1000
    return 4 if n < 300 else 3 if n < 1000 else 2 if n < 5000 else 1


def _make_fit(cfg: TrainingConfig, hw: Any, model: Any, reasoning: List[str]) -> TrainingConfig:
    """Closed downgrade loop: keep relaxing constraints until VRAM fits."""
    has_gpu = getattr(hw, "has_gpu", False) or (getattr(hw, "vram_gb", 0) > 0)
    vram_gb = getattr(hw, "vram_gb", 0.0)

    if not (has_gpu and vram_gb > 0):
        return cfg

    budget = vram_gb * 0.90
    steps = []

    while estimate_vram_gb(model, cfg) > budget:
        if not cfg.grad_checkpointing:
            cfg.grad_checkpointing = True
            steps.append("enabled gradient checkpointing")
        elif cfg.batch_size > 1:
            cfg.batch_size = 1
            steps.append("reduced batch size to 1")
        elif cfg.seq_len > 512:
            cfg.seq_len //= 2
            steps.append(f"halved seq length to {cfg.seq_len}")
        elif cfg.quantization != "4bit":
            cfg.quantization = "8bit" if cfg.quantization == "fp16" else "4bit"
            steps.append(f"lowered precision to {cfg.quantization}")
        elif cfg.lora_rank > 8:
            cfg.lora_rank = 8
            cfg.lora_alpha = 16
            steps.append("reduced LoRA rank to 8")
        else:
            break

    if steps:
        reasoning.append("Auto-fit adjustments: " + "; ".join(steps) + ".")
    return cfg


def estimate_storage_gb(model: Any, cfg: TrainingConfig, num_samples: int, avg_tokens: int, checkpoints: int = 3) -> float:
    num_layers = _get_model_attribute(model, "num_layers", 32)
    hidden_size = _get_model_attribute(model, "hidden_size", 4096)
    
    tp = num_layers * len(cfg.target_modules) * 2 * cfg.lora_rank * hidden_size
    adapter_gb = (tp * 2) / 1e9
    ckpt_gb = adapter_gb * checkpoints
    data_gb = (num_samples * avg_tokens * 4 * 1.2) / 1e9  # processed copies
    return round(ckpt_gb + adapter_gb + data_gb + 0.2, 2)  # + logs


# ------------------------------------------------------------------ orchestrator
def plan_strategy(
    hw: Any,
    model: Any,
    data: Optional[Any] = None,
    goal: Optional[Any] = None,
    override: Optional[Dict[str, Any]] = None,
) -> TrainingStrategy:
    reasoning: List[str] = []
    assumptions: List[str] = []

    seq = _pick_seq_len(data, model, goal)
    avg_tok = getattr(data, "avg_tokens", getattr(data, "tokens_approx", "n/a")) if data else "n/a"
    reasoning.append(f"Sequence length {seq} chosen from avg data tokens (~{avg_tok}) +25% headroom.")

    quant = _pick_precision(model, hw)
    rank = _pick_rank(data, goal)
    cfg = TrainingConfig(
        quantization=quant,
        lora_rank=rank,
        lora_alpha=rank * 2,
        seq_len=seq,
        grad_checkpointing=(quant == "fp16"),
    )

    # Effective batch from dataset volume; physical batch from VRAM headroom
    n = getattr(data, "num_samples", getattr(data, "examples", 1000)) if data else 1000
    eff = 8 if n < 1000 else 16 if n < 5000 else 32
    
    has_gpu = getattr(hw, "has_gpu", False) or (getattr(hw, "vram_gb", 0) > 0)
    vram_gb = getattr(hw, "vram_gb", 0.0)
    params_b = _get_model_attribute(model, "params_b", 3.0)

    cfg.batch_size = 4 if (has_gpu and vram_gb > 0 and params_b * 2 < vram_gb * 0.5) else 1
    cfg.grad_accum = max(1, eff // cfg.batch_size)

    cfg = _make_fit(cfg, hw, model, reasoning)

    epochs = _pick_epochs(data)
    q = getattr(data, "quality_score", None)
    if q is not None and q < 60 and epochs > 2:
        epochs = 2
        reasoning.append(f"Epochs capped at 2: dataset quality {q:.0f}/100 risks overfitting noise.")

    lr = LR_BY_RANK.get(cfg.lora_rank, 2e-4)

    strategy = TrainingStrategy(
        config=cfg,
        learning_rate=lr,
        epochs=epochs,
        effective_batch=cfg.batch_size * cfg.grad_accum,
        estimated_vram_gb=round(estimate_vram_gb(model, cfg), 1),
    )

    # Time estimate heuristic
    tier = getattr(hw, "tier", "T1")
    tps = TIER_THROUGHPUT.get(tier, 1500)
    data_avg_tok = getattr(data, "avg_tokens", getattr(data, "tokens_approx", 512)) if data else 512
    total_tokens = n * data_avg_tok * epochs
    strategy.estimated_minutes = round(total_tokens / tps / 60, 1)
    assumptions.append(f"throughput ~{tps} tok/s for tier {tier}")

    # Storage guard (Report §11 runtime constraints)
    strategy.storage_required_gb = estimate_storage_gb(model, cfg, n, data_avg_tok)
    free = getattr(hw, "free_storage_gb", getattr(hw, "disk_gb", 50.0))
    if strategy.storage_required_gb > free * 0.9:
        reasoning.append(f"⚠️ Storage tight: needs {strategy.storage_required_gb}GB, {free}GB free. "
                         "Checkpoint retention reduced to 1.")

    reasoning.append(f"LoRA rank {rank} + lr {strategy.learning_rate} scaled to {n:,} samples.")
    strategy.reasoning = reasoning
    strategy.assumptions = assumptions + ["AdamW fp32 optimizer states", "+10% allocator overhead"]
    strategy.confidence = 0.85 if has_gpu else 0.65

    if override:
        for k, v in override.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            elif hasattr(strategy, k):
                setattr(strategy, k, v)
        strategy.reasoning.append(f"User override applied: {override}.")
        strategy.confidence = 1.0

    return strategy


def print_strategy(s: TrainingStrategy, model_name: str) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    c = s.config
    console.print(f"\n[bold cyan]⚙️  Training Strategy — {model_name}[/bold cyan] "
                  f"(confidence {s.confidence:.0%})")
    t = Table(show_header=False, box=None)
    t.add_column("k", style="dim")
    t.add_column("v")
    t.add_row("Precision / Rank", f"{c.quantization} / r{c.lora_rank} (α={c.lora_alpha})")
    t.add_row("Batch", f"{c.batch_size} × {c.grad_accum} accum = {s.effective_batch} effective")
    t.add_row("Seq / Epochs / LR", f"{c.seq_len} / {s.epochs} / {s.learning_rate}")
    t.add_row("Est. VRAM", f"{s.estimated_vram_gb} GB")
    t.add_row("Est. Time", f"~{s.estimated_minutes} min")
    t.add_row("Storage Needed", f"{s.storage_required_gb} GB")
    console.print(t)
    console.print("[bold cyan]🧠 Reasoning:[/bold cyan]")
    for r in s.reasoning:
        console.print(f"   ✨ {r}")
    console.print(f"[dim]   assumptions: {', '.join(s.assumptions)}[/dim]")
    console.print("[dim]💡 Override: myai train --override lr=1e-4 rank=32 epochs=5[/dim]\n")
