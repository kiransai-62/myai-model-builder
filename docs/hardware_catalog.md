# 🖥️ MYAI Hardware Intelligence & Requirements Catalog

This document provides a reference for **MYAI's Hardware Intelligence Matrix**, compute tier classifications, and memory requirements across supported open-weight model families.

---

## ⚙️ 1. Hardware Profiling & Compute Dimensions

MYAI profiles your local execution environment across five key hardware dimensions to verify training and inference feasibility:

* **CPU & Instruction Sets**: Multi-core x86-64 / ARM64 architectures with AVX2, AVX-512, or ARM NEON vector extensions for tokenization, preprocessing, and CPU-offloaded computation.
* **System Host RAM**: Evaluates available system memory for dataset caching, tokenizer buffers, and pinned host memory used during memory-optimized streaming.
* **GPU & Acceleration**: Inspects CUDA compute capability, Tensor Core availability, BF16 / FP16 precision support, and memory bandwidth.
* **Memory Subsystem**: Proactively calculates runtime memory demands across model weights, KV cache, activation memory, and adapter buffers.
* **Workspace Storage**: Evaluates available disk space for base weights, runtime checkpoints, and exported artifacts.

---

## 📊 2. Compute Tiers & Supported Architectures

MYAI categorizes hardware environments into four compute tiers:

| Hardware Tier | Memory Profile | Typical Workstation Specs | Supported Model Capabilities |
| :--- | :--- | :--- | :--- |
| **Tier T0 (CPU Only)** | 8GB–32GB Host RAM | Modern Multi-core CPU (Intel / AMD / Apple Silicon) | SmolLM2 (135M–1.7B), Qwen 2.5 (0.5B–1.5B) CPU inference & lightweight tuning |
| **Tier T1 (Low VRAM)** | **4GB–6GB VRAM** | Entry-level GPUs, Laptop RTX 3050/4050, GTX 1650 | SmolLM2, Qwen 2.5 (1.5B/3B), Gemma 3 (1B/4B), 8B models (via Layer Streaming) |
| **Tier T2 (Mid VRAM)** | **8GB–16GB VRAM** | RTX 3060, RTX 4070, Apple Silicon (16GB–36GB unified) | Llama 3.1 (8B), Qwen 2.5 (7B/14B), Phi-4 (14B), Ministral (8B/14B) |
| **Tier T3 (High VRAM)** | **24GB+ VRAM** | RTX 3090, RTX 4090, Pro/Workstation GPUs (A100, H100) | Mistral Small (24B), Qwen 2.5 (32B), Llama 3.1 (70B), MoE architectures |

---

## 💾 3. Model Memory Profiles & Quantization Formats

Memory requirements vary by parameter count and target precision:

| Model Size Tier | Parameter Range | FP16 / BF16 Memory | 4-bit Quantized (Q4_K_M / NF4) | Recommended Minimum Hardware |
| :--- | :---: | :---: | :---: | :--- |
| **Micro / Edge** | 100M – 500M | $\sim 0.3\text{ – }1.0\text{ GB}$ | $< 0.5\text{ GB}$ | Tier T0 (CPU) or any GPU |
| **Tiny / Light** | 1B – 3B | $\sim 2.0\text{ – }6.0\text{ GB}$ | $\sim 1.0\text{ – }2.5\text{ GB}$ | Tier T0/T1 (4GB+ RAM or 4GB VRAM) |
| **Mid / Standard** | 7B – 9B | $\sim 14.0\text{ – }18.0\text{ GB}$ | $\sim 4.5\text{ – }6.5\text{ GB}$ | Tier T1 (Streaming) or Tier T2 (8GB+ VRAM) |
| **Heavy / Server** | 14B – 32B | $\sim 28.0\text{ – }64.0\text{ GB}$ | $\sim 9.0\text{ – }20.0\text{ GB}$ | Tier T2/T3 (16GB–24GB+ VRAM) |
| **Extreme / MoE** | 70B+ / MoE | $> 140\text{ GB}$ | $\sim 40.0\text{ – }50.0\text{ GB}$ | Tier T3 (Multi-GPU / High-Memory Workstations) |

---

## 🎯 4. Decision Verdict Classifications

When running `myai recommend` or `myai system check`, MYAI assigns one of four compatibility verdicts to each candidate model:

* ⭐ **`RECOMMENDED`**: Optimal hardware fit with ample memory headroom and strong task alignment.
* ✅ **`COMPATIBLE`**: Stable execution within standard operational parameters.
* ⚠️ **`POSSIBLE`**: Supported via memory-optimized layer streaming or reduced context lengths.
* ❌ **`UNSUPPORTED`**: Insufficient hardware memory for safe execution.
