<p align="center">
  <img src="assets/myai-logo.png" alt="MYAI Logo - Build · Train · Evolve" width="340">
</p>

<h1 align="center">MYAI</h1>

<p align="center">
  <strong>The Local-First Autonomous AI Model Builder & Zero-Dependency Packager.</strong><br>
  <em>BUILD · TRAIN · EVOLVE</em>
</p>

<p align="center">
  <a href="#-quickstart-in-3-commands">Quickstart</a> &middot;
  <a href="#-why-myai">Why MYAI?</a> &middot;
  <a href="#-key-capabilities">Capabilities</a> &middot;
  <a href="#-hardware-intelligence--tiers">Hardware Intelligence</a> &middot;
  <a href="#-supported-model-families">Models & Hardware</a> &middot;
  <a href="#-complete-cli-reference">CLI Reference</a> &middot;
  <a href="#-export-containment-gate">Security Gate</a> &middot;
  <a href="#-running-your-exported-model">Web Chat UI</a>
</p>

<p align="center">
  <a href="https://github.com/kiransai-62/myai"><img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License"></a>
  <a href="docs/hardware_catalog.md"><img src="https://img.shields.io/badge/hardware%20intelligence-multi--tier%20matrix-blueviolet" alt="Hardware Intelligence"></a>
  <a href="#-post-training-alignment-zoo"><img src="https://img.shields.io/badge/alignment-DPO%20%7C%20ORPO%20%7C%20SimPO%20%7C%20KTO-orange" alt="Alignment"></a>
  <a href="#-export-containment-gate"><img src="https://img.shields.io/badge/export%20gate-verified%20containment-brightgreen" alt="Security Gate"></a>
  <img src="https://img.shields.io/badge/privacy-local--first%20%26%20private-success" alt="Local First Privacy">
</p>

---

Tell MYAI what AI you want. MYAI analyzes your **goal**, your **hardware**, and your **data** — then automatically plans, cleans, fine-tunes, evaluates, optimizes, and packages your custom model into a standalone, portable ZIP with its own Luminous Web & CLI Chat UI, or exports it to GGUF format for Ollama and `llama.cpp`.

> **No cloud GPUs required. Local data privacy. No framework lock-in.**

---

## ⚡ Quickstart in 3 Commands

```bash
# 1. Initialize in current directory (or create a subfolder with 'myai init <name>')
myai init .

# 2. Add your dataset in-place (JSONL, JSON, CSV, TXT, Parquet)
myai data add <path/to-data>      ex - ./workouts.jsonl

# 3. Autopilot: Goal → Hardware → Data → Train → Eval → Optimize → Export
myai auto --export
```

### 🖥️ Autopilot Lifecycle in Action:
```text
[1]  🎯 Goal Understanding:   domain-qa / fitness (Goal-relative eval weights locked)
[2]  🖥️ Hardware Benchmark:   NVIDIA GeForce RTX 3060 · 12.0 GB VRAM · Tier T2
[3]  📊 Dataset Intelligence: 1,420 samples · Quality 92/100 · 0.0% Duplication
[4]  🧠 Model Selection:      Qwen 2.5 (1.5B) Instruct · Recommended
[5]  ⚖️ Dual-Gate Check:      PASS · Estimated 3.8 GB VRAM (8.2 GB Headroom)
[6]  ⚙️ Strategy Planner:     QLoRA 4-bit r16 · 3 Epochs · ~4.2 min
[7]  🏗️ Training Engine:      run-001 · Converged in 4.1 min · Loss 0.412
[8]  🏆 Goal Leaderboard:     run-001 → 87.4/100 Composite Score (Release Candidate)
[9]  🔧 Auto-Optimizer:       Prescribed LR adjustment → run-002 (89.6/100 Composite)
[10] 🛡️ Ship Gate:            Offline Regression Suites Passed → VERDICT: SHIP
[11] 📦 Security Export Gate: Containment Checks Passed → fitness-coach.myai.zip
🎉 YOUR AI IS READY TO DEPLOY
```

---

## 💡 Why MYAI?

Building, evaluating, and packaging fine-tuned LLMs has traditionally been fragmented across ad-hoc scripts, CUDA out-of-memory errors, disconnected evaluation tools, and bulky serving setups. MYAI unifies the entire stack into an **autonomous, local-first platform**:

| Feature | Description |
| :--- | :--- |
| 🔒 **Local-First & Private** | All compute, tokenization, training, and evaluation runs on your local machine. No external cloud telemetry or mandatory cloud connections. |
| 🌊 **Memory-Efficient Training** | Adaptive layer streaming and memory management enables fine-tuning on budget and laptop GPUs. |
| 🖥️ **Hardware Intelligence** | Proactive hardware profiling, multi-tier context profiles, and intelligent resource matching. |
| 🎯 **Post-Training Alignment Zoo** | Direct preference optimization supporting **DPO, ORPO, SimPO, and KTO**. |
| 🧠 **Deterministic Reward Synthesis** | Ingests reference datasets and synthesizes calibrated verifiers (`numeric`, `json_schema`, `regex`, `tool_call`). |
| 🛡️ **Regression Ship Gate** | Bundled offline test suites (JSON validity, tool calling, arithmetic, safety) issuing automated SHIP / DON'T-SHIP verdicts. |
| 📦 **Zero-Dependency Exports** | Self-contained ZIP containing its own built-in `http.server` Luminous Web Chat UI, GGUF format for Ollama, and merged weight checkpoints. |

---

## 🌟 Key Capabilities

```mermaid
graph TD
    A["🎯 Goal Profile (Task, Domain, Latency, Context)"] --> B["🖥️ Hardware Profiler & Benchmark"]
    B --> C["🧹 Dataset Intelligence (Reference Mode & Tokenizer Analysis)"]
    C --> D["⚖️ Feasibility & Memory Verification"]
    D --> E["⚙️ Adaptive Training Strategy Planner"]
    E --> F["🏗️ Fine-Tuning & Alignment Zoo (LoRA / QLoRA / DPO / SimPO)"]
    F --> G["🏆 Goal-Weighted Metric Leaderboard"]
    G --> H["🔧 Autonomous Mutation Optimizer Loop"]
    H --> I["🛡️ Offline Regression Ship Gate"]
    I --> J["📦 Containment Export Gate"]
    J --> K["🚀 Standalone Luminous Web Chat ZIP & GGUF (Ollama)"]
```

### 1. 🎯 Goal Understanding & Composite Metric Weighting
Define your AI's task and domain (`chat`, `code`, `domain-qa`, `summarization`, `extraction`, `reasoning`). MYAI automatically computes goal-weighted evaluation matrices so performance is measured against your actual goal.

### 2. 🧹 Dataset Intelligence & Strict Reference Mode
* **Strict Reference Mode**: Original data files are **never modified in-place**.
* **PII & Secret Scrubbing**: Automatically redacts emails, phone numbers, and API tokens (`sk-`, `hf_`, `ghp_`, `AKIA`).
* **Exact & Fuzzy Deduplication**: Removes duplicate and near-duplicate samples.
* **Leakage Detection**: Isolates train/validation contamination before training starts.

### 3. 🔬 Tokenizer Analysis & Context Distribution
* Exact token counts across modern model families (Llama, Qwen, SmolLM).
* Percentile distributions and context overflow warnings before allocating memory.

### 4. 🌊 Memory-Optimized Layer Streaming
Enables local training on resource-constrained GPUs by dynamically managing transformer block residency in GPU memory.

### 5. 🧬 Post-Training Alignment Zoo
Fine-tune beyond basic SFT with modern preference alignment algorithms:
* **DPO** (Direct Preference Optimization)
* **ORPO** (Odds Ratio Preference Optimization — reference-model-free)
* **SimPO** (Simple Preference Optimization — length-normalized margin)
* **KTO** (Kahneman-Tversky Optimization — binary feedback)

### 6. 🛡️ Regression Ship Gate & Export Gate
* Validates fine-tuned checkpoints against offline regression suites (`myai ship`).
* Enforces strict containment (no internal source code, no `.git`, no `.env`, no raw datasets, no path traversals) before packaging.

---

## 🖥️ Hardware Intelligence & Tiers

MYAI profiles your local machine across CPU, system RAM, GPU compute capability, and VRAM:

| Dimension | Mechanism | Purpose |
| :--- | :--- | :--- |
| **Dynamic Memory Profiling** | Dedicated modes (`inference`, `lora`, `qlora`, `dpo`, `streaming`) | Proactive calculation of weights, KV cache, activations, and runtime buffers |
| **Context Profiles** | Multi-tier evaluation ($2\text{K}$ to $128\text{K}$) | Verifies headroom at realistic operating context lengths |
| **System Compatibility Scoring** | Multi-factor hardware & throughput modeling | Capacity-matched model selection and execution feasibility |
| **Decision Verdicts** | `RECOMMENDED`, `COMPATIBLE`, `POSSIBLE`, `UNSUPPORTED` | Clear decision verdicts with explainable rationale |
| **Storage Budget Management** | Multi-tier storage accounting | Prevents workspace disk exhaustion during training |

---

## 📊 Supported Model Families

MYAI provides a modular model catalog spanning **0.1B to 70B+** parameters across leading open-weight model families:

| Model Family | Representative Sizes | Architecture | Quantization Formats | Primary Strengths |
| :--- | :--- | :---: | :---: | :--- |
| **SmolLM2** | 135M, 360M, 1.7B | Dense | FP16, INT8, Q4_K_M | Ultra-lightweight on-device assistants, edge devices |
| **Qwen 2.5 / 3** | 0.5B, 1.5B, 3B, 7B, 14B, 32B | Dense & MoE | FP16, FP8, AWQ, GPTQ, Q4_K_M | Multilingual instruction following, coding, reasoning |
| **Llama 3.1 / 3.2** | 1B, 3B, 8B, 70B | Dense | FP16, BF16, Q4_K_M, INT8 | General reasoning, tool calling, instruction tuning |
| **Gemma 3** | 270M, 1B, 4B, 12B | Dense | FP16, BF16, Q4_K_M | Mathematical reasoning, high-efficiency generation |
| **Mistral / Ministral** | 3B, 8B, 14B, 24B | Dense & MoE | FP16, FP8, AWQ, Q4_K_M | Code generation, reasoning, efficient MoE |
| **Phi-4** | 3.8B (Mini), 14B | Dense | FP16, BF16, Q4_K_M | Advanced STEM reasoning, logic, and extraction |
| **DeepSeek (R1 Distill)** | 7B, 32B | Dense | FP16, Q4_K_M, AWQ | Deep analytical reasoning, math, and code synthesis |

### Compute Tiers Overview

| Hardware Tier | Memory Profile | Example Hardware | Supported Models |
| :--- | :--- | :--- | :--- |
| **Tier T0 (CPU Only)** | 8GB–32GB Host RAM | Intel Core / AMD Ryzen / Apple Silicon | SmolLM2 (135M–1.7B), Qwen 2.5 (0.5B–1.5B) |
| **Tier T1 (Low VRAM)** | **4GB–6GB VRAM** | GTX 1650, RTX 3050 Laptop | SmolLM2, Qwen 2.5 (1.5B/3B), Gemma 3 (1B/4B), 8B (Streaming) |
| **Tier T2 (Mid VRAM)** | **8GB–16GB VRAM** | RTX 3060, RTX 4070, Apple M-Series | Llama 3.1 (8B), Qwen 2.5 (7B/14B), Phi-4 (14B), Ministral (8B) |
| **Tier T3 (High VRAM)** | **24GB+ VRAM** | RTX 3090, RTX 4090, A100, H100 | Mistral Small (24B), Qwen 2.5 (32B), Llama 3.1 (70B) |

---

## 🧭 Step-by-Step Workflow Guide

### 1. Initialize Project & Goal
```bash
myai init fitness-coach --task domain-qa --domain fitness --context balanced
cd fitness-coach
```

### 2. Register & Clean Data
```bash
# Register dataset source and perform automatic tokenizer analysis
myai data add ./coaching_data.jsonl

# Clean dataset, redact PII, deduplicate, and split 10% for holdout evaluation
myai data clean --fuzzy --val-split 0.1
```

### 3. Model Recommendation & System Check
```bash
# Hardware capability check
myai system check

# Goal- and hardware-aware recommendation
myai recommend
```

### 4. Fine-Tuning & Alignment
```bash
# Standard QLoRA fine-tuning
myai train --epochs 3 --lr 2e-4

# Fine-tune with memory streaming on low-VRAM hardware
myai train --stream-layers

# Direct preference alignment (SimPO, DPO, ORPO, or KTO)
myai train --task simpo
```

### 5. Synthesize Verifiers & Run Ship Gate
```bash
# Synthesize deterministic Python verifiers from references
myai reward synth ./data/references.jsonl -o reward.py

# Execute offline regression gate
myai ship
```

### 6. Export Standalone Package
```bash
# Export zero-dependency Web Chat ZIP
myai export

# Export GGUF format with 4-bit quantization for Ollama
myai export --format gguf --quant q4_k_m

# Merge adapter weights into base model checkpoint
myai merge
```

---

## 🚀 Running Your Exported Model

The exported `.zip` contains a self-contained runtime that requires **zero MYAI framework dependencies**:

```text
fitness-coach.myai.zip
├── model/                  # LoRA adapter weights (safetensors / adapter_model.bin)
├── tokenizer/              # Tokenizer config, vocab, and special tokens
├── metadata.json           # Model provenance, base repo, and goal profile
├── evaluation.json         # Goal-weighted metric scores & gate verification
├── loader.py               # Pure-Python inference loader
└── chat/
    ├── app.py              # Zero-dependency web server (built-in http.server)
    ├── ui.py               # Terminal fallback chat interface
    ├── config.json         # UI configuration & branding
    └── web/
        └── index.html      # Luminous Web Chat interface
```

### Launch Luminous Web Chat UI
```bash
python chat/app.py
```
* Runs on Python's standard library `http.server` (zero external framework dependencies).
* Opens a dark-mode, responsive interface in your browser.
* Includes real-time streaming, parameter controls, token counters, and message history.

---

## 🛡️ Export Containment Gate

Every artifact packaged by `myai export` is validated against automated security and isolation checks:

| Category | Policy / Check | Description |
| :--- | :--- | :--- |
| **Integrity** | Archive & Artifact Structure | Validates non-corrupt ZIP header, model weights, and tokenizer files |
| **Provenance** | Metadata & Evaluation Audit | Manifest records base model origin, training config, and holdout metric scores |
| **Runtime** | Portable Loader & Chat UI | Includes standalone `loader.py` and zero-dependency Web Chat runtime |
| **Containment** | Source & Version Control Isolation | Excludes internal framework source code and `.git/` history |
| **Privacy** | Secret Scrubbing & Data Isolation | Verifies zero API keys (`sk-`, `ghp_`, `hf_`, `AKIA`) and excludes raw training datasets |
| **Security** | Path & Traversal Protection | Prevents absolute host paths and path-traversal entries (`../`) |

---

## 🛠️ Complete CLI Reference

| Command Group | Command | Description |
| :--- | :--- | :--- |
| **Project** | `myai init [name]` | Initialize a project workspace with interactive Goal Profile interview |
| | `myai status` | Inspect project lifecycle state and recommended next steps |
| | `myai system check` | Probe local CPU, RAM, GPU, VRAM, and compute tier |
| | `myai system benchmark` | Run live hardware compute and memory throughput benchmark |
| **Autopilot** | `myai auto [--export]` | **Autonomous Pipeline**: Goal → Hardware → Data → Train → Eval → Optimize → Export |
| **Data** | `myai data add <path>` | Register local datasets in Reference Mode & run tokenizer analysis |
| | `myai data tokenize` | **Tokenizer Analysis**: Compute token counts, distributions & context fit |
| | `myai data clean` | Clean, deduplicate, scrub PII/secrets, and split train/val |
| | `myai data list` / `info` | View registered datasets, sample counts, and quality metrics |
| **Models** | `myai model list` | Browse hierarchical model catalog (Dense, MoE, CPU cores, VRAM) |
| | `myai model use <id>` | Explicitly select active base model for the project |
| | `myai recommend` | Hardware- and goal-aware recommendation with multi-factor scoring |
| **Training** | `myai train` | Train model with live loss curves and progress telemetry |
| | `myai train --stream-layers` | **Layer Streaming**: Fine-tune on resource-constrained GPUs |
| | `myai train --task <method>` | Train preference alignment (**DPO, ORPO, SimPO, KTO**) |
| **Alignment** | `myai reward synth` | Synthesize calibrated deterministic Python verifiers from references |
| | `myai ship` | **Ship Gate**: Offline regression gate for SHIP verdict |
| | `myai merge` | Merge LoRA adapter weights directly into base model weights |
| **Tracking** | `myai runs list` / `info <id>` | List historical training runs and metric breakdowns |
| | `myai runs best` | View experiment leaderboard and current Release Candidate |
| | `myai optimize` | Autonomous retrain/compare hyperparameter optimization loop |
| **Export** | `myai export [--format]` | Package as standalone Web App ZIP, **GGUF (Ollama)**, or Merged weights |
| **Serving** | `myai serve` / `myai ask` | Serve local model with Knowledge Gate RAG protection |

---

## 📜 License

Distributed under the **Apache 2.0** License. See [LICENSE](LICENSE) for details.
