# 🖥️ MYAI Model Hardware Requirements Catalog

This catalog documents the official memory, compute, and storage requirements for the model families supported by **MYAI**, from **0.1B Micro/Tiny** models up to **675B+ Extreme MoE Server** models.

---

## ⚙️ Hardware Assumptions

* **CPU**: Modern x86-64 CPU (preferably 4+ physical cores) for local tokenization, preprocessing, and inference.
* **RAM**: Includes operating system and runtime overhead in the recommendation, not merely raw model weights.
* **GPU**: NVIDIA CUDA GPU (e.g. RTX 3050 Laptop / 3060 / 4070 / 4090 / A100 / H100).
* **VRAM**: Primarily for GPU-resident training and inference. When VRAM is constrained on 4GB GPUs, MYAI's **Exact Layer Streaming** streams frozen base decoder layers from host RAM/NVMe.
* **Storage**: Includes headroom for base checkpoints, tokenizers, optimizer states, adapter checkpoints, and temporary export artifacts.
* **Quantization**: 4-bit estimates (`NF4`, `Q4_K_M`) are for practical local execution; full precision fine-tuning requires higher memory budgets.

---

## 📦 Size Band Specifications

### 1. 0.1B–1B — Micro / Tiny
* **Primary Tiers**: Tier T0 (CPU) & Tier T1 (Low VRAM)
* **Typical MYAI Use**: On-device edge classification, lightweight intent extraction, fast embedded agents.

| Model Family | Size | CPU | System RAM | GPU | VRAM (Q4 Inf) | Storage | Typical MYAI Use |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SmolLM2** | 135M | 2+ cores | 4 GB | Optional | ~1–2 GB | < 1 GB | Tiny on-device agent |
| **SmolLM2** | 360M | 2+ cores | 4 GB | Optional | ~1–2 GB | ~1 GB | Tiny classification / agent |
| **Gemma 3** | 270M | 2+ cores | 4 GB | Optional | 2 GB | 1–2 GB | Micro assistant, routing |
| **Qwen3** | 0.6B | 2+ cores | 4 GB | Optional | 2 GB | 1–2 GB | Lightweight assistant |
| **Qwen3.5** | 0.8B | 2+ cores | 4–6 GB | Optional | 2–3 GB | 2 GB | Fast extraction, simple chat |
| **Llama 3.2** | 1B | 2+ cores | 6 GB | Optional | 2–3 GB | 2–3 GB | Small general assistant |
| **Gemma 3** | 1B | 2+ cores | 6 GB | Optional | 3 GB | 2–3 GB | Small multimodal / text |
| **SmolLM2** | 1.7B | 2–4 cores | 6–8 GB | Optional | 3–4 GB | 3–4 GB | Compact local assistant |

---

### 2. 1B–4B — Small / Compact
* **Primary Tiers**: Tier T1 (Low VRAM - 4GB Laptop GPU) & Tier T2
* **Typical MYAI Use**: Fast conversational assistants, local coding copilots, domain Q&A, and vision reasoning.

| Model Family | Size | CPU | System RAM | GPU | VRAM (Q4 Inf) | Storage | Typical MYAI Use |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama 3.2** | 3B | 4 cores | 8 GB | Optional / 4 GB | 4 GB | 3–5 GB | Chat, extraction, domain Q&A |
| **Qwen3** | 4B | 4 cores | 8 GB | 4–6 GB | 5–6 GB | 5–6 GB | High-speed general AI |
| **Qwen3.5** | 2B | 4 cores | 8 GB | 4 GB | 4 GB | 3–4 GB | Fast general assistant |
| **Qwen3.5** | 4B | 4 cores | 8–12 GB | 6 GB | 5–6 GB | 5–6 GB | Chat, coding, reasoning |
| **Gemma 3** | 4B | 4 cores | 8–12 GB | 6 GB | 5–6 GB | 5–6 GB | Text + vision multimodal |
| **Phi-4-mini** | 3.8B | 4 cores | 8–12 GB | 6 GB | 5–6 GB | 5–6 GB | 128k context reasoning / code |
| **Ministral 3** | 3B class | 4 cores | 8–12 GB | 6 GB | 5–6 GB | 5–7 GB | Edge vision + text |

---

### 3. 7B–15B — Mid / Medium
* **Primary Tiers**: Tier T1 (via Layer Streaming) & Tier T2 (Resident QLoRA / DPO)
* **Typical MYAI Use**: Production-grade local AI, complex multi-step reasoning, tool calling, and preference alignment.

| Model Family | Size | CPU | System RAM | GPU | VRAM (4-bit Inf) | Storage | Fine-Tuning Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen3** | 8B | 6+ cores | 16 GB | NVIDIA 8 GB+ | 6–8 GB | 8–10 GB | 10–12 GB+ *(~3.32 GB via Layer Streaming)* |
| **Llama 3.1 / 3.x** | 8B | 6+ cores | 16 GB | NVIDIA 8 GB+ | 6–8 GB | 8–10 GB | 10–14 GB+ *(~3.32 GB via Layer Streaming)* |
| **Qwen3.5** | 9B | 6+ cores | 16 GB | NVIDIA 8 GB+ | 7–9 GB | 9–11 GB | 12–16 GB+ |
| **Gemma 3** | 12B | 8 cores | 16–24 GB | NVIDIA 12 GB+ | 9–11 GB | 10–14 GB | 16–24 GB+ |
| **Phi-4** | 14B | 8 cores | 24 GB | NVIDIA 12–16 GB | 10–13 GB | 12–16 GB | 18–24 GB+ |
| **Ministral 3** | 8B | 6+ cores | 16 GB | NVIDIA 8 GB+ | 6–8 GB | 8–10 GB | 12–16 GB+ |
| **Ministral 3** | 14B | 8 cores | 24 GB | NVIDIA 16 GB+ | 10–13 GB | 13–16 GB | 18–24 GB+ |
| **DeepSeek Distill** | 7B/8B/14B | 6–8 cores | 16–24 GB | NVIDIA 8–16 GB | 6–13 GB | 8–16 GB | 12–24 GB+ |

---

### 4. 20B–40B — Large Local
* **Primary Tiers**: Tier T3 (Single High-End GPU / RTX 4090 / 32GB–48GB Workstations)
* **Typical MYAI Use**: Advanced deep reasoning, mathematical synthesis, and high-accuracy autonomous agents.

| Model Family | Size | CPU | System RAM | GPU | VRAM (4-bit Inf) | Storage | Fine-Tuning Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen3** | 30B-A3B (MoE) | 8+ cores | 32 GB | NVIDIA 16 GB+ | 18–22 GB | 20–25 GB | 24–32 GB+ |
| **Qwen3** | 32B | 8–12 cores | 32–48 GB | NVIDIA 24 GB+ | 20–24 GB | 22–30 GB | 32–48 GB+ |
| **Qwen3.5** | 27B | 8–12 cores | 32 GB | NVIDIA 20–24 GB | 18–22 GB | 20–25 GB | 28–40 GB+ |
| **Qwen3.5** | 35B-A3B (MoE) | 8–12 cores | 32–48 GB | NVIDIA 24 GB+ | 20–25 GB | 25–30 GB | 32–48 GB+ |
| **Gemma 3** | 27B | 8–12 cores | 32–48 GB | NVIDIA 24 GB+ | 18–22 GB | 20–25 GB | 32–48 GB+ |
| **Mistral Small 3.1** | 24B | 8+ cores | 32 GB | NVIDIA 24 GB / RTX 4090 | ~18–22 GB | ~25 GB | 28–40 GB+ |
| **DeepSeek-R1 Distill** | 32B | 8–12 cores | 32–48 GB | NVIDIA 24 GB+ | ~22–26 GB | 25–35 GB | 32–48 GB+ |

---

### 5. 60B–90B — High-End Workstation / Multi-GPU
* **Primary Tiers**: Dual GPU / 48GB–96GB VRAM High-End Workstations

| Model Family | Size | CPU | System RAM | GPU Configuration | Approx. 4-bit VRAM | Storage | Fine-Tuning |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama 3.1** | 70B | 16+ cores | 64–96 GB | 48 GB+ / Multi-GPU | ~40–45 GB | 45–55 GB | 64–96 GB+ |
| **Llama 3.3** | 70B | 16+ cores | 64–96 GB | 48 GB+ / Multi-GPU | ~40–45 GB | 45–55 GB | 64–96 GB+ |
| **DeepSeek Distill** | 70B | 16+ cores | 64–96 GB | 48 GB+ / Multi-GPU | ~40–45 GB | 45–55 GB | 64–96 GB+ |
| **Qwen 2.5** | 72B | 16+ cores | 64–96 GB | 48 GB+ / Multi-GPU | ~42–48 GB | 50–60 GB | 64–96+ GB |

---

### 6. 100B–150B — Server / Multi-GPU MoE
* **Primary Tiers**: Multi-GPU Node Clusters (2–4 × 48GB/80GB GPUs)

| Model Family | Total Params | Active Params | CPU | System RAM | GPU Setup | Approx. 4-bit VRAM | Storage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen3.5** | 122B-A10B | 10B | 24+ cores | 128 GB+ | 2–4 × 48 GB+ | ~70–80 GB+ | 80–100 GB |
| **GLM-4.5-Air** | 106B-A12B | 12B | 24+ cores | 128 GB+ | Multi-GPU | ~60–70 GB+ | ~75–100 GB |
| **Mistral Medium 3.5** | 128B | Dense | 24+ cores | 128 GB+ | Multi-GPU | ~70–85 GB+ | ~85–110 GB |

---

### 7. 200B–250B — Large MoE
| Model Family | Total Params | Active Params | CPU | System RAM | GPU Setup | 4-bit Memory Planning | Storage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen3** | 235B-A22B | 22B | 32+ cores | 256 GB+ | Multi-GPU | ~130–150 GB+ | 140–170 GB |
| **Large MoE Cluster** | 200–250B | 15–30B | 32+ cores | 256 GB+ | Multi-GPU | 120–160 GB+ | 140–180 GB |

> **Note**: Active parameters determine computational latency; total parameters dictate model weight storage and VRAM capacity.

---

### 8. 250B–500B+ — Extreme Server MoE
| Model Family | Total Params | Active Params | CPU | System RAM | GPU Setup | Approx. 4-bit Memory | Storage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen3.5** | ~397B | ~17B | 48+ cores | 512 GB+ | Multi-GPU | ~200 GB+ | 230 GB+ |
| **GLM-4.5** | 355B | 32B | 48+ cores | 512 GB+ | Multi-GPU | ~180–200 GB+ | 210–250 GB |
| **Llama 4 Maverick** | ~402B | 17B | 48+ cores | 512 GB+ | Multi-GPU | ~200 GB+ | 230 GB+ |
| **Mistral Large 3** | 675B | MoE | 64+ cores | 512 GB–1 TB+ | Large Multi-GPU | ~340 GB+ | 400 GB+ |

---

## 🧩 Better MYAI Schema Specification

The `RegistryModel` entity in `src/myai/models/schema.py` exposes complete hardware, architectural, and capability attributes:

```yaml
id: qwen3-8b-instruct
name: Qwen3 8B Instruct
family: Qwen3
modality: Text
architecture:
  parameters: 8B
  type: Dense
  hidden_size: 4096
  num_layers: 32
  context_length: 131072
hardware:
  cpu_min_cores: 6
  ram_min_gb: 16.0
  minimum_vram_gb: 8.0
  vram_q4_gb: 7.0
  vram_fp16_gb: 18.0
  storage_gb: 10.0
  finetune_vram_gb: 12.0
  training_ram_gb: 32.0
capabilities:
  vision: false
  audio: false
  tools: true
  reasoning: true
training:
  methods: [LoRA, QLoRA, layer_streaming, dpo, simpo]
source:
  repository: Qwen/Qwen3-8B-Instruct
license:
  name: Apache 2.0
recommended_tier: T2
confidence: 0.98
```
