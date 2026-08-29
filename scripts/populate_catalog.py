import yaml
from pathlib import Path

models = [
    # ── 0.1B–1B — Micro / Tiny ─────────────────────────────────────────
    {
        "id": "smollm2-135m-instruct",
        "name": "SmolLM2 135M Instruct",
        "family": "SmolLM2",
        "modality": "Text",
        "architecture": {"parameters": "135M", "type": "Dense", "hidden_size": 576, "num_layers": 30, "context_length": 2048},
        "hardware": {"cpu_min_cores": 2, "ram_min_gb": 4.0, "minimum_vram_gb": 2.0, "vram_q4_gb": 1.0, "vram_fp16_gb": 1.5, "storage_gb": 1.0, "finetune_vram_gb": 2.0, "training_ram_gb": 4.0},
        "capabilities": {"vision": False, "audio": False, "tools": True, "reasoning": False},
        "training": {"methods": ["LoRA", "QLoRA"]},
        "source": {"repository": "HuggingFaceTB/SmolLM2-135M-Instruct"},
        "license": {"name": "Apache 2.0"},
        "recommended_tier": "T0",
        "confidence": 0.98
    },
    {
        "id": "smollm2-360m-instruct",
        "name": "SmolLM2 360M Instruct",
        "family": "SmolLM2",
        "modality": "Text",
        "architecture": {"parameters": "360M", "type": "Dense", "hidden_size": 960, "num_layers": 32, "context_length": 2048},
        "hardware": {"cpu_min_cores": 2, "ram_min_gb": 4.0, "minimum_vram_gb": 2.0, "vram_q4_gb": 1.0, "vram_fp16_gb": 2.0, "storage_gb": 1.5, "finetune_vram_gb": 2.5, "training_ram_gb": 6.0},
        "capabilities": {"vision": False, "audio": False, "tools": True, "reasoning": False},
        "training": {"methods": ["LoRA", "QLoRA"]},
        "source": {"repository": "HuggingFaceTB/SmolLM2-360M-Instruct"},
        "license": {"name": "Apache 2.0"},
        "recommended_tier": "T0",
        "confidence": 0.98
    },
    {
        "id": "gemma3-270m-instruct",
        "name": "Gemma 3 270M Instruct",
        "family": "Gemma 3",
        "modality": "Text",
        "architecture": {"parameters": "270M", "type": "Dense", "hidden_size": 768, "num_layers": 18, "context_length": 4096},
        "hardware": {"cpu_min_cores": 2, "ram_min_gb": 4.0, "minimum_vram_gb": 2.0, "vram_q4_gb": 1.0, "vram_fp16_gb": 2.0, "storage_gb": 2.0, "finetune_vram_gb": 2.5, "training_ram_gb": 6.0},
        "capabilities": {"vision": False, "audio": False, "tools": True, "reasoning": False},
        "training": {"methods": ["LoRA", "QLoRA"]},
        "source": {"repository": "google/gemma-3-270m-it"},
        "license": {"name": "Gemma Community"},
        "recommended_tier": "T0",
        "confidence": 0.95
    },
    {
        "id": "qwen3-0.6b-instruct",
        "name": "Qwen3 0.6B Instruct",
        "family": "Qwen3",
        "modality": "Text",
        "architecture": {"parameters": "0.6B", "type": "Dense", "hidden_size": 1024, "num_layers": 24, "context_length": 32768},
        "hardware": {"cpu_min_cores": 2, "ram_min_gb": 4.0, "minimum_vram_gb": 2.0, "vram_q4_gb": 1.5, "vram_fp16_gb": 3.0, "storage_gb": 2.0, "finetune_vram_gb": 3.0, "training_ram_gb": 8.0},
        "capabilities": {"vision": False, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA"]},
        "source": {"repository": "Qwen/Qwen3-0.6B-Instruct"},
        "license": {"name": "Apache 2.0"},
        "recommended_tier": "T0",
        "confidence": 0.96
    },
    {
        "id": "qwen3.5-0.8b-instruct",
        "name": "Qwen3.5 0.8B Instruct",
        "family": "Qwen3.5",
        "modality": "Text",
        "architecture": {"parameters": "0.8B", "type": "Dense", "hidden_size": 1024, "num_layers": 24, "context_length": 32768},
        "hardware": {"cpu_min_cores": 2, "ram_min_gb": 6.0, "minimum_vram_gb": 2.5, "vram_q4_gb": 2.0, "vram_fp16_gb": 3.5, "storage_gb": 2.0, "finetune_vram_gb": 3.5, "training_ram_gb": 8.0},
        "capabilities": {"vision": False, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA"]},
        "source": {"repository": "Qwen/Qwen3.5-0.8B-Instruct"},
        "license": {"name": "Apache 2.0"},
        "recommended_tier": "T0",
        "confidence": 0.96
    },
    {
        "id": "gemma3-1b-instruct",
        "name": "Gemma 3 1B Instruct",
        "family": "Gemma 3",
        "modality": "Text + Vision",
        "architecture": {"parameters": "1B", "type": "Dense", "hidden_size": 1536, "num_layers": 26, "context_length": 8192},
        "hardware": {"cpu_min_cores": 2, "ram_min_gb": 6.0, "minimum_vram_gb": 3.0, "vram_q4_gb": 2.5, "vram_fp16_gb": 4.0, "storage_gb": 3.0, "finetune_vram_gb": 4.0, "training_ram_gb": 8.0},
        "capabilities": {"vision": True, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA"]},
        "source": {"repository": "google/gemma-3-1b-it"},
        "license": {"name": "Gemma Community"},
        "recommended_tier": "T0",
        "confidence": 0.95
    },
    {
        "id": "smollm2-1.7b-instruct",
        "name": "SmolLM2 1.7B Instruct",
        "family": "SmolLM2",
        "modality": "Text",
        "architecture": {"parameters": "1.7B", "type": "Dense", "hidden_size": 2048, "num_layers": 24, "context_length": 8192},
        "hardware": {"cpu_min_cores": 4, "ram_min_gb": 8.0, "minimum_vram_gb": 4.0, "vram_q4_gb": 3.0, "vram_fp16_gb": 6.0, "storage_gb": 4.0, "finetune_vram_gb": 5.0, "training_ram_gb": 12.0},
        "capabilities": {"vision": False, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA"]},
        "source": {"repository": "HuggingFaceTB/SmolLM2-1.7B-Instruct"},
        "license": {"name": "Apache 2.0"},
        "recommended_tier": "T1",
        "confidence": 0.96
    },

    # ── 1B–4B — Small / Compact ────────────────────────────────────────
    {
        "id": "qwen3-4b-instruct",
        "name": "Qwen3 4B Instruct",
        "family": "Qwen3",
        "modality": "Text",
        "architecture": {"parameters": "4B", "type": "Dense", "hidden_size": 2560, "num_layers": 36, "context_length": 32768},
        "hardware": {"cpu_min_cores": 4, "ram_min_gb": 12.0, "minimum_vram_gb": 6.0, "vram_q4_gb": 5.0, "vram_fp16_gb": 10.0, "storage_gb": 6.0, "finetune_vram_gb": 8.0, "training_ram_gb": 16.0},
        "capabilities": {"vision": False, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA"]},
        "source": {"repository": "Qwen/Qwen3-4B-Instruct"},
        "license": {"name": "Apache 2.0"},
        "recommended_tier": "T1",
        "confidence": 0.97
    },
    {
        "id": "phi-4-mini-instruct",
        "name": "Phi-4 Mini Instruct",
        "family": "Phi-4",
        "modality": "Text",
        "architecture": {"parameters": "3.8B", "type": "Dense", "hidden_size": 3072, "num_layers": 32, "context_length": 131072},
        "hardware": {"cpu_min_cores": 4, "ram_min_gb": 12.0, "minimum_vram_gb": 6.0, "vram_q4_gb": 5.0, "vram_fp16_gb": 9.0, "storage_gb": 6.0, "finetune_vram_gb": 8.0, "training_ram_gb": 16.0},
        "capabilities": {"vision": False, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA", "layer_streaming"]},
        "source": {"repository": "microsoft/Phi-4-mini-instruct"},
        "license": {"name": "MIT"},
        "recommended_tier": "T1",
        "confidence": 0.98
    },
    {
        "id": "ministral-3-3b-instruct",
        "name": "Ministral 3 3B Instruct",
        "family": "Ministral 3",
        "modality": "Text + Vision",
        "architecture": {"parameters": "3B", "type": "Dense", "hidden_size": 3072, "num_layers": 28, "context_length": 131072},
        "hardware": {"cpu_min_cores": 4, "ram_min_gb": 12.0, "minimum_vram_gb": 6.0, "vram_q4_gb": 5.0, "vram_fp16_gb": 8.0, "storage_gb": 6.0, "finetune_vram_gb": 7.5, "training_ram_gb": 16.0},
        "capabilities": {"vision": True, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA", "layer_streaming"]},
        "source": {"repository": "mistralai/Ministral-3b-instruct"},
        "license": {"name": "Mistral Research"},
        "recommended_tier": "T1",
        "confidence": 0.96
    },
    {
        "id": "gemma3-4b-instruct",
        "name": "Gemma 3 4B Instruct",
        "family": "Gemma 3",
        "modality": "Text + Vision",
        "architecture": {"parameters": "4B", "type": "Dense", "hidden_size": 2560, "num_layers": 32, "context_length": 131072},
        "hardware": {"cpu_min_cores": 4, "ram_min_gb": 12.0, "minimum_vram_gb": 6.0, "vram_q4_gb": 5.0, "vram_fp16_gb": 10.0, "storage_gb": 6.0, "finetune_vram_gb": 8.0, "training_ram_gb": 16.0},
        "capabilities": {"vision": True, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA", "layer_streaming"]},
        "source": {"repository": "google/gemma-3-4b-it"},
        "license": {"name": "Gemma Community"},
        "recommended_tier": "T1",
        "confidence": 0.96
    },

    # ── 7B–15B — Mid / Medium ──────────────────────────────────────────
    {
        "id": "qwen3-8b-instruct",
        "name": "Qwen3 8B Instruct",
        "family": "Qwen3",
        "modality": "Text",
        "architecture": {"parameters": "8B", "type": "Dense", "hidden_size": 4096, "num_layers": 32, "context_length": 131072},
        "hardware": {"cpu_min_cores": 6, "ram_min_gb": 16.0, "minimum_vram_gb": 8.0, "vram_q4_gb": 7.0, "vram_fp16_gb": 18.0, "storage_gb": 10.0, "finetune_vram_gb": 12.0, "training_ram_gb": 32.0},
        "capabilities": {"vision": False, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA", "layer_streaming", "dpo", "simpo"]},
        "source": {"repository": "Qwen/Qwen3-8B-Instruct"},
        "license": {"name": "Apache 2.0"},
        "recommended_tier": "T2",
        "confidence": 0.98
    },
    {
        "id": "phi-4-14b-instruct",
        "name": "Phi-4 14B Instruct",
        "family": "Phi-4",
        "modality": "Text",
        "architecture": {"parameters": "14B", "type": "Dense", "hidden_size": 5120, "num_layers": 40, "context_length": 16384},
        "hardware": {"cpu_min_cores": 8, "ram_min_gb": 24.0, "minimum_vram_gb": 14.0, "vram_q4_gb": 11.0, "vram_fp16_gb": 30.0, "storage_gb": 16.0, "finetune_vram_gb": 20.0, "training_ram_gb": 48.0},
        "capabilities": {"vision": False, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA", "layer_streaming"]},
        "source": {"repository": "microsoft/phi-4"},
        "license": {"name": "MIT"},
        "recommended_tier": "T2",
        "confidence": 0.97
    },
    {
        "id": "ministral-3-8b-instruct",
        "name": "Ministral 3 8B Instruct",
        "family": "Ministral 3",
        "modality": "Text + Vision",
        "architecture": {"parameters": "8B", "type": "Dense", "hidden_size": 4096, "num_layers": 32, "context_length": 131072},
        "hardware": {"cpu_min_cores": 6, "ram_min_gb": 16.0, "minimum_vram_gb": 8.0, "vram_q4_gb": 7.0, "vram_fp16_gb": 18.0, "storage_gb": 10.0, "finetune_vram_gb": 12.0, "training_ram_gb": 32.0},
        "capabilities": {"vision": True, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA", "layer_streaming"]},
        "source": {"repository": "mistralai/Ministral-8b-instruct"},
        "license": {"name": "Mistral Research"},
        "recommended_tier": "T2",
        "confidence": 0.97
    },
    {
        "id": "ministral-3-14b-instruct",
        "name": "Ministral 3 14B Instruct",
        "family": "Ministral 3",
        "modality": "Text + Vision",
        "architecture": {"parameters": "14B", "type": "Dense", "hidden_size": 5120, "num_layers": 40, "context_length": 131072},
        "hardware": {"cpu_min_cores": 8, "ram_min_gb": 24.0, "minimum_vram_gb": 14.0, "vram_q4_gb": 12.0, "vram_fp16_gb": 30.0, "storage_gb": 16.0, "finetune_vram_gb": 20.0, "training_ram_gb": 48.0},
        "capabilities": {"vision": True, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA", "layer_streaming"]},
        "source": {"repository": "mistralai/Ministral-14b-instruct"},
        "license": {"name": "Mistral Research"},
        "recommended_tier": "T2",
        "confidence": 0.96
    },
    {
        "id": "deepseek-distill-7b-instruct",
        "name": "DeepSeek R1 Distill Qwen 7B",
        "family": "DeepSeek Distill",
        "modality": "Text",
        "architecture": {"parameters": "7B", "type": "Dense", "hidden_size": 3584, "num_layers": 28, "context_length": 131072},
        "hardware": {"cpu_min_cores": 6, "ram_min_gb": 16.0, "minimum_vram_gb": 8.0, "vram_q4_gb": 6.5, "vram_fp16_gb": 16.0, "storage_gb": 10.0, "finetune_vram_gb": 12.0, "training_ram_gb": 32.0},
        "capabilities": {"vision": False, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA", "layer_streaming", "dpo", "simpo"]},
        "source": {"repository": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"},
        "license": {"name": "MIT"},
        "recommended_tier": "T2",
        "confidence": 0.98
    },

    # ── 20B–40B — Large Local ──────────────────────────────────────────
    {
        "id": "mistral-small-3.1-24b-instruct",
        "name": "Mistral Small 3.1 24B Instruct",
        "family": "Mistral Small",
        "modality": "Text + Vision",
        "architecture": {"parameters": "24B", "type": "Dense", "hidden_size": 6144, "num_layers": 48, "context_length": 32768},
        "hardware": {"cpu_min_cores": 8, "ram_min_gb": 32.0, "minimum_vram_gb": 24.0, "vram_q4_gb": 18.0, "vram_fp16_gb": 48.0, "storage_gb": 25.0, "finetune_vram_gb": 28.0, "training_ram_gb": 64.0},
        "capabilities": {"vision": True, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA", "layer_streaming"]},
        "source": {"repository": "mistralai/Mistral-Small-24B-Instruct-2501"},
        "license": {"name": "Apache 2.0"},
        "recommended_tier": "T3",
        "confidence": 0.98
    },
    {
        "id": "deepseek-r1-distill-32b-instruct",
        "name": "DeepSeek R1 Distill Qwen 32B",
        "family": "DeepSeek Distill",
        "modality": "Text",
        "architecture": {"parameters": "32B", "type": "Dense", "hidden_size": 5120, "num_layers": 64, "context_length": 131072},
        "hardware": {"cpu_min_cores": 12, "ram_min_gb": 48.0, "minimum_vram_gb": 24.0, "vram_q4_gb": 22.0, "vram_fp16_gb": 65.0, "storage_gb": 35.0, "finetune_vram_gb": 36.0, "training_ram_gb": 64.0},
        "capabilities": {"vision": False, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA", "layer_streaming", "dpo"]},
        "source": {"repository": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"},
        "license": {"name": "MIT"},
        "recommended_tier": "T3",
        "confidence": 0.98
    },
    {
        "id": "qwen3-30b-a3b-instruct",
        "name": "Qwen3 30B-A3B Instruct",
        "family": "Qwen3 MoE",
        "modality": "Text",
        "architecture": {"parameters": "30B", "type": "MoE", "active_parameters": "3B", "hidden_size": 2048, "num_layers": 48, "context_length": 32768},
        "hardware": {"cpu_min_cores": 8, "ram_min_gb": 32.0, "minimum_vram_gb": 18.0, "vram_q4_gb": 18.0, "vram_fp16_gb": 60.0, "storage_gb": 25.0, "finetune_vram_gb": 24.0, "training_ram_gb": 48.0},
        "capabilities": {"vision": False, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA"]},
        "source": {"repository": "Qwen/Qwen3-30B-A3B-Instruct"},
        "license": {"name": "Apache 2.0"},
        "recommended_tier": "T3",
        "confidence": 0.97
    },

    # ── 60B–90B — Workstation / Multi-GPU ──────────────────────────────
    {
        "id": "llama-3.1-70b-instruct",
        "name": "Llama 3.1 70B Instruct",
        "family": "Llama 3.1",
        "modality": "Text",
        "architecture": {"parameters": "70B", "type": "Dense", "hidden_size": 8192, "num_layers": 80, "context_length": 131072},
        "hardware": {"cpu_min_cores": 16, "ram_min_gb": 64.0, "minimum_vram_gb": 48.0, "vram_q4_gb": 42.0, "vram_fp16_gb": 140.0, "storage_gb": 55.0, "finetune_vram_gb": 64.0, "training_ram_gb": 96.0},
        "capabilities": {"vision": False, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA", "layer_streaming"]},
        "source": {"repository": "meta-llama/Meta-Llama-3.1-70B-Instruct"},
        "license": {"name": "Llama-3.1-Community"},
        "recommended_tier": "T3",
        "confidence": 0.98
    },

    # ── Server MoE & Extreme (100B - 675B+) ────────────────────────────
    {
        "id": "glm-4.5-air-106b-a12b",
        "name": "GLM-4.5 Air 106B-A12B",
        "family": "GLM-4.5",
        "modality": "Text",
        "architecture": {"parameters": "106B", "type": "MoE", "active_parameters": "12B", "hidden_size": 4096, "num_layers": 46, "context_length": 131072},
        "hardware": {"cpu_min_cores": 24, "ram_min_gb": 128.0, "minimum_vram_gb": 64.0, "vram_q4_gb": 65.0, "vram_fp16_gb": 212.0, "storage_gb": 90.0, "finetune_vram_gb": 96.0, "training_ram_gb": 192.0},
        "capabilities": {"vision": False, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA"]},
        "source": {"repository": "THUDM/glm-4.5-air"},
        "license": {"name": "Open Model License"},
        "recommended_tier": "T3",
        "confidence": 0.97
    },
    {
        "id": "mistral-large-3-675b",
        "name": "Mistral Large 3 675B MoE",
        "family": "Mistral Large",
        "modality": "Text + Vision",
        "architecture": {"parameters": "675B", "type": "MoE", "active_parameters": "45B", "hidden_size": 8192, "num_layers": 88, "context_length": 131072},
        "hardware": {"cpu_min_cores": 64, "ram_min_gb": 512.0, "minimum_vram_gb": 340.0, "vram_q4_gb": 340.0, "vram_fp16_gb": 1350.0, "storage_gb": 400.0, "finetune_vram_gb": 512.0, "training_ram_gb": 768.0},
        "capabilities": {"vision": True, "audio": False, "tools": True, "reasoning": True},
        "training": {"methods": ["LoRA", "QLoRA"]},
        "source": {"repository": "mistralai/Mistral-Large-3"},
        "license": {"name": "Mistral Research"},
        "recommended_tier": "T3",
        "confidence": 0.99
    }
]

out_dir = Path("src/myai/registry/models")
out_dir.mkdir(parents=True, exist_ok=True)
for m in models:
    fpath = out_dir / f"{m['id']}.yaml"
    fpath.write_text(yaml.dump(m, sort_keys=False), encoding="utf-8")
print(f"Successfully wrote {len(models)} models to {out_dir}")
