# 🖥️ MYAI Model Hardware Intelligence & Requirements Catalog

This document defines the comprehensive **15-Point Hardware Intelligence Architecture**, dynamic memory formulas, multi-quantization specifications, and requirements catalog across all size tiers (from **0.1B Micro/Tiny** models up to **675B+ Extreme MoE Server** models).

---

## ⚙️ 1. Hardware Assumptions & Architecture

* **CPU**: Modern x86-64 / ARM64 CPU with instruction set acceleration (**AVX2**, **AVX-512**, **ARM NEON**) for tokenization, preprocessing, and CPU simulation inference.
* **Host System RAM**: Accounts for OS kernel, Python runtime, active dataset caching, tokenizer vocab buffers, and pinned host memory (crucial for **Exact Layer Streaming**).
* **GPU & CUDA Compute**: Evaluates CUDA compute capability, Tensor Cores, **BF16 / FP16 / FP8** precision support, PCIe generation bandwidth, and NVLink inter-GPU topologies.
* **Separation of Inference vs. Training**: Fine-tuning demands substantial memory for gradients, AdamW 8-bit optimizer states, and LoRA adapter buffers, while inference depends on quantized weights ($Q4/Q8$) and dynamic KV caching.
* **3-Tier Storage Model**:
  1. `download_size_gb`: Compressed/raw model weights to fetch.
  2. `runtime_storage_gb`: Model checkpoint + tokenizer vocab + embedding cache.
  3. `workspace_storage_gb`: Checkpoints (x3) + optimizer states + temporary export artifacts.

---

## 📐 2. Dynamic Memory & VRAM Calculation Engine

MYAI calculates peak memory dynamically rather than relying on static estimates:

$$\text{VRAM}_{\text{total}} = W(\text{Quant}) + \text{KV}(\text{Ctx}, B, L, H, D) + A(\text{Ctx}, B, H) + \text{LoRA}(r, \alpha) + \text{AllocHeadroom}$$

### Memory Subsystem Breakdown:
1. **Weights Memory ($W$)**:
   $$\text{Weights (GB)} = \frac{\text{Params}_{\text{total}} \times 10^9 \times (\text{Bits} / 8)}{1024^3} \times 1.05$$
2. **KV Cache Memory ($\text{KV}$)**:
   $$\text{KV Cache (GB)} = \frac{2 \times \text{Layers} \times \text{HiddenSize} \times (\text{PrecisionBits} / 8) \times \text{ContextLength} \times \text{BatchSize}}{1024^3}$$
3. **Activation Memory ($A$)**:
   $$\text{Activations (GB)} = \frac{\text{ContextLength} \times \text{BatchSize} \times \text{HiddenSize} \times 2 \times \text{Layers} \times \text{CheckpointFactor}}{1024^3}$$
4. **Training Overhead**:
   * **LoRA (FP16)**: $\sim 35\%$ of weight memory $+ 1.2\text{ GB}$ adapter states.
   * **QLoRA (4-bit NF4)**: $\sim 20\%$ of weight memory $+ 0.8\text{ GB}$ adapter states.
   * **Exact Layer Streaming**: Caps peak weights in VRAM to $\sim 0.85\text{ GB} + 1.2\text{ GB}$ LoRA buffers ($\sim 3.32\text{ GB}$ peak VRAM for 8B models).

---

## 🧮 3. 8-Factor Hardware Fit Scoring & 4-Tier Verdict System

### 8-Factor Weighted Hardware Score:

| Hardware Factor | Weight | Evaluation Criteria |
| :--- | :---: | :--- |
| **1. VRAM Headroom** | **30%** | Peak VRAM vs. Available GPU VRAM ($> 2\text{ GB}$ headroom = $100\%$) |
| **2. System RAM** | **15%** | Host RAM vs. Training/Streaming requirements |
| **3. GPU Compute Tier** | **15%** | Architecture tier ($T3=100\%$, $T2=90\%$, $T1=75\%$, $T0=55\%$) |
| **4. CPU Capacity** | **10%** | Physical core count vs. Model minimum/recommended cores |
| **5. Storage Budget** | **10%** | Free disk vs. 3-tier workspace storage requirements |
| **6. Throughput (tok/s)** | **10%** | Predicted/measured tokens per second based on active parameters |
| **7. Context Capacity** | **5%** | Maximum context length fit without truncation |
| **8. Runtime Support** | **5%** | Native driver, QLoRA, LoRA, and GGUF runtime availability |

### 4-Tier Verdict Classification:
* ⭐ **`RECOMMENDED`**: Overall fit score $\ge 85\%$, ample VRAM headroom ($\ge 1.0\text{ GB}$), and optimal dataset/task alignment.
* ✅ **`COMPATIBLE`**: Overall fit score $\ge 60\%$, meets minimum specifications stably.
* ⚠️ **`POSSIBLE`**: Operable via **Exact Layer Streaming** or reduced context length, but headroom is tight.
* ❌ **`UNSUPPORTED`**: Insufficient hardware memory (VRAM/RAM/CPU) for safe execution.

---

## 📦 4. Comprehensive Model Requirements Catalog (0.1B to 675B+)

### 1. 0.1B–1B — Micro / Tiny
* **Primary Targets**: Tier T0 (CPU) & Edge / Embedded
* **Key Use Cases**: Ultra-fast classification, intent extraction, lightweight on-device agents.

| Model ID | Family | Params | CPU (Min/Rec) | Inf VRAM (Q4/FP16) | Train VRAM (LoRA) | Storage (Worksp) | Est. Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `smollm2-135m-instruct` | SmolLM2 | 135M | 2c / 4c (AVX2) | ~1.0 GB / ~1.5 GB | ~2.0 GB | 2.0 GB | ~120 tok/s |
| `smollm2-360m-instruct` | SmolLM2 | 360M | 2c / 4c (AVX2) | ~1.2 GB / ~2.0 GB | ~2.5 GB | 3.0 GB | ~95 tok/s |
| `gemma3-270m-instruct` | Gemma 3 | 270M | 2c / 4c (AVX2) | ~1.2 GB / ~2.0 GB | ~2.5 GB | 3.0 GB | ~85 tok/s |
| `qwen3-0.6b-instruct` | Qwen3 | 0.6B | 2c / 4c (AVX2) | ~1.5 GB / ~2.5 GB | ~3.5 GB | 4.0 GB | ~75 tok/s |
| `qwen3.5-0.8b-instruct` | Qwen3.5 | 0.8B | 2c / 4c (AVX2) | ~1.8 GB / ~3.0 GB | ~4.0 GB | 5.0 GB | ~68 tok/s |
| `llama-3.2-1b-instruct` | Llama 3.2 | 1B | 2c / 4c (AVX2) | ~2.0 GB / ~3.5 GB | ~4.5 GB | 6.0 GB | ~60 tok/s |
| `gemma3-1b-instruct` | Gemma 3 | 1B (V) | 2c / 4c (AVX2) | ~2.5 GB / ~4.0 GB | ~5.0 GB | 6.5 GB | ~55 tok/s |
| `smollm2-1.7b-instruct` | SmolLM2 | 1.7B | 4c / 6c (AVX2) | ~3.0 GB / ~5.5 GB | ~6.0 GB | 8.0 GB | ~48 tok/s |

---

### 2. 1B–4B — Small / Compact
* **Primary Targets**: Tier T1 (4GB Laptop GPU / GTX 1650) & Tier T2
* **Key Use Cases**: Fast conversational AI, domain Q&A, local coding assistance, edge vision.

| Model ID | Family | Params | CPU (Min/Rec) | Inf VRAM (Q4/FP16) | Train VRAM (QLoRA) | Storage (Worksp) | Est. Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `llama-3.2-3b-instruct` | Llama 3.2 | 3B | 4c / 8c (AVX-512) | ~4.0 GB / ~8.0 GB | ~5.5 GB | 14.0 GB | ~42 tok/s |
| `qwen3-4b-instruct` | Qwen3 | 4B | 4c / 8c (AVX2) | ~5.0 GB / ~10.0 GB | ~6.5 GB | 18.0 GB | ~38 tok/s |
| `phi-4-mini-instruct` | Phi-4 | 3.8B | 4c / 8c (AVX-512) | ~5.0 GB / ~9.5 GB | ~6.0 GB | 17.0 GB | ~38 tok/s |
| `gemma3-4b-instruct` | Gemma 3 | 4B (V) | 4c / 8c (AVX2) | ~5.0 GB / ~10.0 GB | ~6.5 GB | 19.0 GB | ~36 tok/s |
| `ministral-3-3b-instruct`| Ministral 3 | 3B (V) | 4c / 8c (AVX2) | ~4.5 GB / ~8.5 GB | ~5.8 GB | 15.0 GB | ~40 tok/s |

---

### 3. 7B–15B — Mid / Medium
* **Primary Targets**: Tier T1 (*Exact Layer Streaming*) & Tier T2 (Resident RTX 3060 / 4070)
* **Key Use Cases**: Production local AI, complex reasoning, tool calling, preference alignment (DPO/SimPO).

| Model ID | Family | Params | CPU (Min/Rec) | Inf VRAM (Q4/FP16) | Train VRAM (Resident) | Train VRAM (Streaming) | Est. Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `llama-3.1-8b-instruct` | Llama 3.1 | 8B | 6c / 12c (AVX-512) | ~7.0 GB / ~17.5 GB | ~9.5 GB (QLoRA) | **~3.32 GB** | ~28 tok/s |
| `qwen3-8b-instruct` | Qwen3 | 8B | 6c / 12c (AVX2) | ~7.0 GB / ~18.0 GB | ~9.8 GB (QLoRA) | **~3.32 GB** | ~28 tok/s |
| `gemma3-12b-instruct` | Gemma 3 | 12B (V) | 8c / 16c (AVX2) | ~10.0 GB / ~25.0 GB | ~13.0 GB (QLoRA) | **~3.80 GB** | ~22 tok/s |
| `phi-4-14b-instruct` | Phi-4 | 14B | 8c / 16c (AVX-512) | ~11.5 GB / ~29.0 GB | ~14.5 GB (QLoRA) | **~4.00 GB** | ~20 tok/s |
| `deepseek-distill-7b-instruct` | DeepSeek | 7B | 6c / 12c (AVX2) | ~6.5 GB / ~16.0 GB | ~9.0 GB (QLoRA) | **~3.20 GB** | ~30 tok/s |

---

### 4. 20B–40B — Large Local
* **Primary Targets**: Tier T3 (RTX 3090 / 4090 / 24GB–48GB Workstations)
* **Key Use Cases**: Advanced deep reasoning, mathematical synthesis, high-accuracy autonomous agents.

| Model ID | Family | Type | Params (Act/Tot) | Inf VRAM (Q4) | Train VRAM (QLoRA) | Storage (Worksp) | Est. Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `mistral-small-3.1-24b` | Mistral Small | Dense (V)| 24B / 24B | ~18.0 GB | ~22.0 GB | 110.0 GB | ~14 tok/s |
| `qwen3-30b-a3b-instruct` | Qwen3 MoE | MoE | **3B / 30B** | ~18.0 GB | ~24.0 GB | 130.0 GB | **~32 tok/s** |
| `deepseek-r1-distill-32b`| DeepSeek | Dense | 32B / 32B | ~22.0 GB | ~28.0 GB | 150.0 GB | ~10 tok/s |

---

### 5. 60B–90B — High-End Workstation / Multi-GPU
* **Primary Targets**: Dual GPU / 48GB–96GB VRAM Workstations

| Model ID | Family | Params | Topology | Inf VRAM (Q4) | Train VRAM (LoRA) | Storage (Worksp) | Est. Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `llama-3.1-70b-instruct` | Llama 3.1 | 70B | 2–4 GPUs (NVLink) | ~42.0 GB | ~55.0 GB (QLoRA) | 320.0 GB | ~5 tok/s |

---

### 6. 100B–675B+ — Server & Extreme MoE Clusters
* **Primary Targets**: Distributed Multi-GPU Cluster Nodes (4–16 GPUs, 80GB VRAM, NVLink / InfiniBand, FSDP / Tensor Parallelism)

| Model ID | Family | Type | Params (Act/Tot) | Multi-GPU Topology | Inf VRAM (Q4) | Train VRAM (FSDP) | Storage (Worksp) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `glm-4.5-air-106b-a12b` | GLM-4.5 | MoE | **12B / 106B** | 2–4 × 48 GB (NVLink) | ~65.0 GB | ~85.0 GB (QLoRA) | 480.0 GB |
| `qwen3-235b-a22b` | Qwen3 MoE | MoE | **22B / 235B** | 4–8 × 80 GB (NVLink) | ~140.0 GB | ~180.0 GB (QLoRA) | 1,100.0 GB |
| `mistral-large-3-675b` | Mistral Large | MoE (V)| **45B / 675B** | 8–16 × 80 GB (InfiniBand)| ~340.0 GB | ~450.0 GB (QLoRA) | 3,000.0 GB |

---

## 🔍 5. Explainable Recommendation Example

When running `myai recommend`, MYAI computes and displays multi-factor explainability:

```text
1. Llama 3.1 8B Instruct (llama-3.1-8b-instruct) (Active)
   Verdict      : ⭐ RECOMMENDED
   Fit Score    : 94/100 (Confidence: 94%)
   Method       : QLoRA
   Estimated Spd: ~28.0 tok/s
   VRAM (Q4/FP) : ~7.0 GB / ~17.5 GB (Training: ~16.0 GB)
   Storage      : Download 15.0 GB · Workspace 38.0 GB
   Fit Matrix   : HW 95.0% · Data 95.0% · Task 95.0% · Deploy 85.0%
   Rationale    : Fits in 12.0 GB VRAM with QLoRA; Instruction-tuned architecture matches conversational/Q&A goal
```
