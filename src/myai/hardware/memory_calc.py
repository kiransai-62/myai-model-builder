"""
MYAI Dedicated Memory & Dynamic Context Calculation Engine.

Provides explicit, specialized memory calculation modes:
- inference()
- lora_training()
- qlora_training()
- dpo_training()
- grpo_training()
- layer_streaming()
- evaluate_context_profiles()
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any


class MemoryMode(str, Enum):
    INFERENCE = "inference"
    LORA = "lora_training"
    QLORA = "qlora_training"
    DPO = "dpo_training"
    GRPO = "grpo_training"
    LAYER_STREAMING = "layer_streaming"


@dataclass
class StorageBreakdown:
    download_size_gb: float
    runtime_storage_gb: float
    workspace_storage_gb: float


@dataclass
class DynamicMemoryProfile:
    weights_vram_gb: float
    kv_cache_vram_gb: float
    activation_vram_gb: float
    training_overhead_vram_gb: float
    total_peak_vram_gb: float
    system_ram_gb: float
    storage: StorageBreakdown
    estimated_tokens_per_sec: float
    is_safe: bool
    headroom_gb: float
    mode: MemoryMode = MemoryMode.INFERENCE
    recommended_context: int = 4096
    context_profiles: Dict[int, Tuple[float, str]] = field(default_factory=dict)


# Quantization bits mapping
QUANT_BITS_MAP: Dict[str, float] = {
    "FP32": 32.0,
    "FP16": 16.0,
    "BF16": 16.0,
    "FP8": 8.0,
    "INT8": 8.0,
    "GPTQ": 4.0,
    "AWQ": 4.0,
    "INT4": 4.0,
    "GGUF_Q4_K_M": 4.5,
    "GGUF_Q5_K_M": 5.5,
    "GGUF_Q6_K": 6.5,
    "GGUF_Q8_0": 8.5,
    "GGUF_F16": 16.0,
    "NF4": 4.0,
}


def calculate_weight_memory_gb(params_total_b: float, quant_format: str = "FP16") -> float:
    """Calculates model weight memory in GB with 5% non-quantized overhead."""
    bits = QUANT_BITS_MAP.get(quant_format.upper(), 16.0)
    bytes_per_param = bits / 8.0
    raw_gb = (params_total_b * 1e9 * bytes_per_param) / (1024 ** 3)
    return round(raw_gb * 1.05, 2)


def calculate_kv_cache_gb(
    num_layers: int,
    hidden_size: int,
    context_length: int,
    batch_size: int = 1,
    kv_precision_bits: float = 16.0,
) -> float:
    """Calculates KV Cache memory footprint in GB."""
    bytes_per_elem = kv_precision_bits / 8.0
    total_bytes = 2 * num_layers * hidden_size * bytes_per_elem * context_length * batch_size
    return round(total_bytes / (1024 ** 3), 3)


def calculate_activation_memory_gb(
    context_length: int,
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    is_training: bool = False,
    gradient_checkpointing: bool = True,
) -> float:
    """Calculates activation memory footprint in GB."""
    if not is_training:
        bytes_act = context_length * batch_size * hidden_size * 2 * 2
        return round(max(0.1, bytes_act / (1024 ** 3)), 2)

    if gradient_checkpointing:
        act_gb = (context_length * batch_size * hidden_size * 2 * num_layers * 0.1) / (1024 ** 3)
    else:
        act_gb = (context_length * batch_size * hidden_size * 2 * num_layers * 1.5) / (1024 ** 3)
    return round(max(0.2, act_gb), 2)


def calculate_storage_breakdown(
    params_total_b: float,
    quant_format: str = "FP16",
    is_training: bool = False,
) -> StorageBreakdown:
    """Calculates 3-tier storage requirements (Download, Runtime, Workspace)."""
    base_weight_gb = calculate_weight_memory_gb(params_total_b, quant_format)
    download_size = round(base_weight_gb * 1.02, 1)
    runtime_storage = round(download_size + 1.5, 1)

    if is_training:
        workspace_storage = round(runtime_storage * 2.5 + 4.0, 1)
    else:
        workspace_storage = runtime_storage

    return StorageBreakdown(
        download_size_gb=download_size,
        runtime_storage_gb=runtime_storage,
        workspace_storage_gb=workspace_storage,
    )


class MemoryCalculator:
    """
    Dedicated Memory Engine supporting explicit execution modes.
    """

    @staticmethod
    def inference(
        params_total_b: float,
        params_active_b: float,
        num_layers: int,
        hidden_size: int,
        quant_format: str = "GGUF_Q4_K_M",
        context_length: int = 4096,
        batch_size: int = 1,
        available_vram_gb: float = 8.0,
        available_ram_gb: float = 16.0,
        gpu_tier: str = "T2",
        benchmark_tok_per_sec: Optional[float] = None,
    ) -> DynamicMemoryProfile:
        """Inference mode: Weights + KV Cache + Minimal Activations + Headroom."""
        weights = calculate_weight_memory_gb(params_total_b, quant_format)
        kv = calculate_kv_cache_gb(num_layers, hidden_size, context_length, batch_size)
        act = calculate_activation_memory_gb(context_length, batch_size, hidden_size, num_layers, is_training=False)
        overhead = 0.0
        headroom = 0.3
        peak_vram = round(weights + kv + act + overhead + headroom, 2)
        system_ram = round(max(4.0, weights * 0.5 + 2.0), 1)

        storage = calculate_storage_breakdown(params_total_b, quant_format, is_training=False)
        base_speed = 250.0 if gpu_tier == "T3" else (120.0 if gpu_tier == "T2" else 45.0)
        effective_base = (benchmark_tok_per_sec * 8.0) if benchmark_tok_per_sec else base_speed
        est_tok_s = round(max(1.0, effective_base / max(0.5, params_active_b)), 1)

        vram_headroom = round(available_vram_gb - peak_vram, 2)
        is_safe = vram_headroom >= 0.2 and available_ram_gb >= system_ram

        return DynamicMemoryProfile(
            weights_vram_gb=weights,
            kv_cache_vram_gb=kv,
            activation_vram_gb=act,
            training_overhead_vram_gb=overhead,
            total_peak_vram_gb=peak_vram,
            system_ram_gb=system_ram,
            storage=storage,
            estimated_tokens_per_sec=est_tok_s,
            is_safe=is_safe,
            headroom_gb=vram_headroom,
            mode=MemoryMode.INFERENCE,
            recommended_context=context_length,
        )

    @staticmethod
    def lora_training(
        params_total_b: float,
        params_active_b: float,
        num_layers: int,
        hidden_size: int,
        quant_format: str = "FP16",
        context_length: int = 4096,
        batch_size: int = 2,
        available_vram_gb: float = 16.0,
        available_ram_gb: float = 32.0,
        gpu_tier: str = "T2",
        benchmark_tok_per_sec: Optional[float] = None,
    ) -> DynamicMemoryProfile:
        """LoRA (FP16/BF16 base): Weights + Activations + Gradients + Optimizer + LoRA Adapters."""
        weights = calculate_weight_memory_gb(params_total_b, quant_format)
        kv = calculate_kv_cache_gb(num_layers, hidden_size, context_length, batch_size)
        act = calculate_activation_memory_gb(context_length, batch_size, hidden_size, num_layers, is_training=True)
        # LoRA overhead: adapter weights + gradients + 8-bit AdamW optimizer states
        training_overhead = round(weights * 0.35 + 1.2, 2)
        headroom = 0.5
        peak_vram = round(weights + kv + act + training_overhead + headroom, 2)
        system_ram = round(max(8.0, weights * 1.5 + 4.0), 1)

        storage = calculate_storage_breakdown(params_total_b, quant_format, is_training=True)
        base_speed = 250.0 if gpu_tier == "T3" else (120.0 if gpu_tier == "T2" else 45.0)
        effective_base = (benchmark_tok_per_sec * 8.0) if benchmark_tok_per_sec else base_speed
        est_tok_s = round(max(0.5, (effective_base / max(0.5, params_active_b)) * 0.25), 1)

        vram_headroom = round(available_vram_gb - peak_vram, 2)
        is_safe = vram_headroom >= 0.3 and available_ram_gb >= system_ram

        return DynamicMemoryProfile(
            weights_vram_gb=weights,
            kv_cache_vram_gb=kv,
            activation_vram_gb=act,
            training_overhead_vram_gb=training_overhead,
            total_peak_vram_gb=peak_vram,
            system_ram_gb=system_ram,
            storage=storage,
            estimated_tokens_per_sec=est_tok_s,
            is_safe=is_safe,
            headroom_gb=vram_headroom,
            mode=MemoryMode.LORA,
            recommended_context=context_length,
        )

    @staticmethod
    def qlora_training(
        params_total_b: float,
        params_active_b: float,
        num_layers: int,
        hidden_size: int,
        quant_format: str = "NF4",
        context_length: int = 4096,
        batch_size: int = 2,
        available_vram_gb: float = 8.0,
        available_ram_gb: float = 16.0,
        gpu_tier: str = "T2",
        benchmark_tok_per_sec: Optional[float] = None,
    ) -> DynamicMemoryProfile:
        """QLoRA (4-bit NF4 base): 4-bit Base + FP16 Adapters + Paged AdamW."""
        weights = calculate_weight_memory_gb(params_total_b, quant_format)
        kv = calculate_kv_cache_gb(num_layers, hidden_size, context_length, batch_size)
        act = calculate_activation_memory_gb(context_length, batch_size, hidden_size, num_layers, is_training=True)
        training_overhead = round(weights * 0.20 + 0.8, 2)
        headroom = 0.4
        peak_vram = round(weights + kv + act + training_overhead + headroom, 2)
        system_ram = round(max(8.0, weights * 1.5 + 4.0), 1)

        storage = calculate_storage_breakdown(params_total_b, quant_format, is_training=True)
        base_speed = 250.0 if gpu_tier == "T3" else (120.0 if gpu_tier == "T2" else 45.0)
        effective_base = (benchmark_tok_per_sec * 8.0) if benchmark_tok_per_sec else base_speed
        est_tok_s = round(max(0.5, (effective_base / max(0.5, params_active_b)) * 0.25), 1)

        vram_headroom = round(available_vram_gb - peak_vram, 2)
        is_safe = vram_headroom >= 0.3 and available_ram_gb >= system_ram

        return DynamicMemoryProfile(
            weights_vram_gb=weights,
            kv_cache_vram_gb=kv,
            activation_vram_gb=act,
            training_overhead_vram_gb=training_overhead,
            total_peak_vram_gb=peak_vram,
            system_ram_gb=system_ram,
            storage=storage,
            estimated_tokens_per_sec=est_tok_s,
            is_safe=is_safe,
            headroom_gb=vram_headroom,
            mode=MemoryMode.QLORA,
            recommended_context=context_length,
        )

    @staticmethod
    def dpo_training(
        params_total_b: float,
        params_active_b: float,
        num_layers: int,
        hidden_size: int,
        context_length: int = 4096,
        batch_size: int = 1,
        available_vram_gb: float = 16.0,
        available_ram_gb: float = 32.0,
        gpu_tier: str = "T2",
    ) -> DynamicMemoryProfile:
        """DPO: Policy Model + Reference Model (frozen in 4-bit) + Preference Pairs."""
        policy_weights = calculate_weight_memory_gb(params_total_b, "NF4")
        ref_weights = calculate_weight_memory_gb(params_total_b, "NF4")
        kv = calculate_kv_cache_gb(num_layers, hidden_size, context_length, batch_size * 2)
        act = calculate_activation_memory_gb(context_length, batch_size * 2, hidden_size, num_layers, is_training=True)
        overhead = round(policy_weights * 0.25 + 1.0, 2)
        headroom = 0.5
        peak_vram = round(policy_weights + ref_weights + kv + act + overhead + headroom, 2)
        system_ram = round(max(16.0, policy_weights * 2.5 + 4.0), 1)

        storage = calculate_storage_breakdown(params_total_b, "NF4", is_training=True)
        est_tok_s = round(max(0.5, (100.0 / max(0.5, params_active_b)) * 0.18), 1)
        vram_headroom = round(available_vram_gb - peak_vram, 2)
        is_safe = vram_headroom >= 0.3 and available_ram_gb >= system_ram

        return DynamicMemoryProfile(
            weights_vram_gb=round(policy_weights + ref_weights, 2),
            kv_cache_vram_gb=kv,
            activation_vram_gb=act,
            training_overhead_vram_gb=overhead,
            total_peak_vram_gb=peak_vram,
            system_ram_gb=system_ram,
            storage=storage,
            estimated_tokens_per_sec=est_tok_s,
            is_safe=is_safe,
            headroom_gb=vram_headroom,
            mode=MemoryMode.DPO,
            recommended_context=context_length,
        )

    @staticmethod
    def grpo_training(
        params_total_b: float,
        params_active_b: float,
        num_layers: int,
        hidden_size: int,
        context_length: int = 4096,
        batch_size: int = 1,
        group_size: int = 4,
        available_vram_gb: float = 24.0,
        available_ram_gb: float = 64.0,
        gpu_tier: str = "T3",
    ) -> DynamicMemoryProfile:
        """GRPO: Policy Model + Group Rollout Activations (G samples) + Reward Head."""
        weights = calculate_weight_memory_gb(params_total_b, "NF4")
        kv = calculate_kv_cache_gb(num_layers, hidden_size, context_length, batch_size * group_size)
        act = calculate_activation_memory_gb(context_length, batch_size * group_size, hidden_size, num_layers, is_training=True)
        overhead = round(weights * 0.30 + 1.5, 2)
        headroom = 0.6
        peak_vram = round(weights + kv + act + overhead + headroom, 2)
        system_ram = round(max(24.0, weights * 3.0 + 8.0), 1)

        storage = calculate_storage_breakdown(params_total_b, "NF4", is_training=True)
        est_tok_s = round(max(0.5, (100.0 / max(0.5, params_active_b)) * 0.12), 1)
        vram_headroom = round(available_vram_gb - peak_vram, 2)
        is_safe = vram_headroom >= 0.3 and available_ram_gb >= system_ram

        return DynamicMemoryProfile(
            weights_vram_gb=weights,
            kv_cache_vram_gb=kv,
            activation_vram_gb=act,
            training_overhead_vram_gb=overhead,
            total_peak_vram_gb=peak_vram,
            system_ram_gb=system_ram,
            storage=storage,
            estimated_tokens_per_sec=est_tok_s,
            is_safe=is_safe,
            headroom_gb=vram_headroom,
            mode=MemoryMode.GRPO,
            recommended_context=context_length,
        )

    @staticmethod
    def layer_streaming(
        params_total_b: float,
        params_active_b: float,
        num_layers: int,
        hidden_size: int,
        context_length: int = 2048,
        batch_size: int = 1,
        available_vram_gb: float = 4.0,
        available_ram_gb: float = 32.0,
        gpu_tier: str = "T1",
    ) -> DynamicMemoryProfile:
        """Exact Layer Streaming: Base layers in host RAM; ~0.85GB + LoRA buffers in VRAM (~3.32 GB peak)."""
        stream_weights = round(0.85 + (params_total_b * 0.08), 2)
        # Layer streaming uses chunked attention buffers
        kv = calculate_kv_cache_gb(num_layers, hidden_size, min(context_length, 2048), batch_size)
        act = calculate_activation_memory_gb(min(context_length, 2048), batch_size, hidden_size, num_layers, is_training=True)
        training_overhead = 0.6
        headroom = 0.3
        peak_vram = round(stream_weights + kv + act + training_overhead + headroom, 2)
        # Weights reside in pinned host RAM
        system_ram = round(calculate_weight_memory_gb(params_total_b, "FP16") + 8.0, 1)

        storage = calculate_storage_breakdown(params_total_b, "FP16", is_training=True)
        est_tok_s = round(max(0.5, (45.0 / max(0.5, params_active_b)) * 0.15), 1)
        vram_headroom = round(available_vram_gb - peak_vram, 2)
        is_safe = vram_headroom >= 0.2 and available_ram_gb >= system_ram

        return DynamicMemoryProfile(
            weights_vram_gb=stream_weights,
            kv_cache_vram_gb=kv,
            activation_vram_gb=act,
            training_overhead_vram_gb=training_overhead,
            total_peak_vram_gb=peak_vram,
            system_ram_gb=system_ram,
            storage=storage,
            estimated_tokens_per_sec=est_tok_s,
            is_safe=is_safe,
            headroom_gb=vram_headroom,
            mode=MemoryMode.LAYER_STREAMING,
            recommended_context=context_length,
        )


def evaluate_context_profiles(
    params_total_b: float,
    params_active_b: float,
    num_layers: int,
    hidden_size: int,
    available_vram_gb: float,
    quant_format: str = "INT4",
    is_training: bool = False,
    method: str = "qlora",
) -> Tuple[int, Dict[int, Tuple[float, str]]]:
    """
    Evaluates multi-tier context profiles (2K, 4K, 8K, 16K, 32K, 64K, 128K)
    and returns (recommended_context, {tokens: (vram_gb, verdict)}).
    """
    profiles = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    results: Dict[int, Tuple[float, str]] = {}
    rec_ctx = 2048

    for ctx in profiles:
        if is_training:
            if method.lower() == "layer_streaming":
                prof = MemoryCalculator.layer_streaming(params_total_b, params_active_b, num_layers, hidden_size, context_length=ctx)
            elif method.lower() == "lora":
                prof = MemoryCalculator.lora_training(params_total_b, params_active_b, num_layers, hidden_size, context_length=ctx, available_vram_gb=available_vram_gb)
            else:
                prof = MemoryCalculator.qlora_training(params_total_b, params_active_b, num_layers, hidden_size, context_length=ctx, available_vram_gb=available_vram_gb)
        else:
            prof = MemoryCalculator.inference(params_total_b, params_active_b, num_layers, hidden_size, quant_format=quant_format, context_length=ctx, available_vram_gb=available_vram_gb)

        if available_vram_gb > 0:
            if prof.headroom_gb >= 1.5:
                verdict = "⭐ Recommended"
                rec_ctx = ctx
            elif prof.headroom_gb >= 0.2:
                verdict = "✅ Compatible"
                if rec_ctx < ctx and ctx <= 8192:
                    rec_ctx = ctx
            elif prof.headroom_gb >= -1.0 and method.lower() == "layer_streaming":
                verdict = "⚠️ Possible"
            else:
                verdict = "❌ Unsupported"
        else:
            verdict = "✅ Compatible" if ctx <= 8192 else "⚠️ Possible"
            if ctx <= 4096:
                rec_ctx = ctx

        results[ctx] = (prof.total_peak_vram_gb, verdict)

    return rec_ctx, results


def calculate_dynamic_memory_profile(
    params_total_b: float,
    params_active_b: float,
    num_layers: int,
    hidden_size: int,
    quant_format: str = "INT4",
    context_length: int = 4096,
    batch_size: int = 1,
    is_training: bool = False,
    training_method: str = "qlora",
    available_vram_gb: float = 8.0,
    available_ram_gb: float = 16.0,
    gpu_tier: str = "T2",
    benchmark_tok_per_sec: Optional[float] = None,
) -> DynamicMemoryProfile:
    """Unified entrypoint dispatching to specialized memory modes."""
    rec_ctx, ctx_profiles = evaluate_context_profiles(
        params_total_b, params_active_b, num_layers, hidden_size,
        available_vram_gb=available_vram_gb, quant_format=quant_format,
        is_training=is_training, method=training_method
    )

    if not is_training:
        prof = MemoryCalculator.inference(
            params_total_b, params_active_b, num_layers, hidden_size,
            quant_format=quant_format, context_length=context_length,
            batch_size=batch_size, available_vram_gb=available_vram_gb,
            available_ram_gb=available_ram_gb, gpu_tier=gpu_tier,
            benchmark_tok_per_sec=benchmark_tok_per_sec,
        )
    elif training_method.lower() == "layer_streaming":
        prof = MemoryCalculator.layer_streaming(
            params_total_b, params_active_b, num_layers, hidden_size,
            context_length=context_length, batch_size=batch_size,
            available_vram_gb=available_vram_gb, available_ram_gb=available_ram_gb,
            gpu_tier=gpu_tier,
        )
    elif training_method.lower() == "lora":
        prof = MemoryCalculator.lora_training(
            params_total_b, params_active_b, num_layers, hidden_size,
            context_length=context_length, batch_size=batch_size,
            available_vram_gb=available_vram_gb, available_ram_gb=available_ram_gb,
            gpu_tier=gpu_tier, benchmark_tok_per_sec=benchmark_tok_per_sec,
        )
    elif training_method.lower() == "dpo":
        prof = MemoryCalculator.dpo_training(
            params_total_b, params_active_b, num_layers, hidden_size,
            context_length=context_length, batch_size=batch_size,
            available_vram_gb=available_vram_gb, available_ram_gb=available_ram_gb,
            gpu_tier=gpu_tier,
        )
    elif training_method.lower() == "grpo":
        prof = MemoryCalculator.grpo_training(
            params_total_b, params_active_b, num_layers, hidden_size,
            context_length=context_length, batch_size=batch_size,
            available_vram_gb=available_vram_gb, available_ram_gb=available_ram_gb,
            gpu_tier=gpu_tier,
        )
    else:  # qlora
        prof = MemoryCalculator.qlora_training(
            params_total_b, params_active_b, num_layers, hidden_size,
            quant_format=quant_format, context_length=context_length,
            batch_size=batch_size, available_vram_gb=available_vram_gb,
            available_ram_gb=available_ram_gb, gpu_tier=gpu_tier,
            benchmark_tok_per_sec=benchmark_tok_per_sec,
        )

    prof.recommended_context = rec_ctx
    prof.context_profiles = ctx_profiles
    return prof
