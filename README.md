# MYAI 🧠

**The Local-First Autonomous AI Model Builder & Zero-Dependency Packager.**

Tell MYAI what AI you want. MYAI understands your **goal**, your **hardware**, and your **data** — then automatically plans, cleans, fine-tunes, evaluates, optimizes, and packages your custom model into a standalone, portable ZIP with its own Luminous Web & CLI Chat UI.

No cloud GPUs required. Zero data leakage. No framework lock-in.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Tests](https://img.shields.io/badge/tests-194%2F194%20passing-brightgreen)
![Security Gate](https://img.shields.io/badge/export%20gate-18%2F18%20verified-brightgreen)
![Privacy](https://img.shields.io/badge/privacy-100%25%20local%20%26%20air--gapped-success)
![Autopilot](https://img.shields.io/badge/autopilot-goal--to--deployment-blueviolet)
![Tokenizer](https://img.shields.io/badge/tokenizer-streaming%20%26%20model--aware-blue)

---

## ⚡ Quickstart in 3 Commands

```bash
# 1. Initialize your project with an interactive Goal Profile
myai init fitness-coach

# 2. Add your data in-place (JSONL, JSON, CSV)
cd fitness-coach
myai data add ./workouts.jsonl

# 3. Autopilot: Goal → Hardware → Data → Train → Eval → Optimize → Export
myai auto --export
```

```text
[1] 🎯 Goal: domain-qa / fitness
[2] 🖥️ Hardware: NVIDIA GeForce RTX 3060 · tier T2
[3] 📊 Data: 1,420 samples · quality 92/100 · dup 0.0%
[4] 🧠 Model: Qwen 2.5 (1.5B) · LoRA / QLoRA
[5] ⚖️ Feasibility: PASS · est 3.8 GB VRAM
[6] ⚙️ Strategy: 4bit r16 · 3 ep · ~4.2 min · 0.8 GB storage
[7] 🏗️ Training: run-001 · 4.1 min
[8] 🏆 Leaderboard: run-001 → 87.4/100 composite
[9] 🔧 Optimizer: improved → run-002 (89.6/100)
[10] 📦 Export: 18/18 security gate passed → fitness-coach.myai.zip
🎉 YOUR AI IS READY
```

---

## 🌟 Key Capabilities

```mermaid
graph LR
    A[🎯 Goal Understanding] --> B[🖥️ Hardware Benchmark]
    B --> C[🧹 Dataset Intelligence]
    C --> D[⚖️ Dual-Gate Feasibility]
    D --> E[⚙️ Training Strategy Planner]
    E --> F[🏗️ Training Engine]
    F --> G[🏆 Goal-Weighted Leaderboard]
    G --> H[🔧 Autonomous Optimizer]
    H --> I[📦 18-Point Security Gate]
    I --> J[🚀 Standalone Portable ZIP]
```

### 1. 🎯 Goal Understanding & Composite Metric Weighting (Stage A)

Define your AI's task and domain (`chat`, `code`, `domain-qa`, `summarization`, `extraction`, `reasoning`). MYAI automatically computes goal-weighted evaluation matrices (BLEU, ROUGE, readability, domain accuracy, exact match) so "best" is measured against your actual goal, not generic averages.

### 2. 🧹 Dataset Intelligence, Cleaning & Contamination Filter (Stage B)

* **Strict Reference Mode**: Original data files are **never modified**.
* **PII & Secret Scrubbing**: Redacts emails, phone numbers, OpenAI (`sk-`), HuggingFace (`hf_`), GitHub (`ghp_`), and AWS credentials.
* **Exact & Fuzzy Deduplication**: SequenceMatcher and MD5 hashing eliminate sample redundancy.
* **Leakage Detection**: Automatically isolates train/validation contamination before training starts.

### 3. ⚖️ Dual-Gate Feasibility & OOM-Safe Strategy (Stage C)

* **Hardware Fit**: Empirical VRAM modeling computes base model weights, KV cache, activation memory, LoRA adapter states, and CUDA allocator headroom.
* **Closed-Loop Auto-Downgrade**: If VRAM is tight, automatically relaxes precision (4-bit/8-bit), activates gradient checkpointing, adjusts batch size, and caps sequence length.
* **Storage Guard**: Verifies available disk space against checkpoint and export budgets.

### 4. 🏆 Experiment Leaderboard & Release Candidate Designation (Stage D)

* **Matrix Ranking**: Ranks every historical run by its goal-weighted composite score.
* **Regression Gate**: Regressed runs have their score halved and are strictly forbidden from becoming release candidates.

### 5. 🔧 Autonomous Optimizer Loop (Stage E)

* Diagnoses the weakest goal-weighted metrics in the release candidate.
* Prescribes minimal strategy mutations (e.g. adjust LR, LoRA rank, sequence length, epochs).
* Retrains and promotes the new run **only if** `Δ >= min_delta` (Improvement Justified Gate) and regression stability holds.
* Bounded iterations and full `--dry-run` preview support.

### 6. 📦 18-Point Export Security Gate & Luminous Chat UI

Every exported model ZIP is validated against an **18-point verification gate**:

* ✅ Includes LoRA adapter weights, tokenizer, metadata, and evaluation report.
* ✅ Includes zero-dependency Standalone Web & CLI Chat UI (built with native Python `http.server`).
* 🚫 **No MYAI framework code**, no `.git/`, no `.env` files, no API keys, no raw datasets, and no path traversal entries.

---

## 💻 Installation

### Requirements

* **Python**: 3.10+
* **OS**: Windows, macOS, or Linux
* **Hardware**: CPU (Tier T1) or NVIDIA GPU with CUDA (Tiers T2/T3)
* **Privacy**: 100% Local. No telemetry. No external API keys required.

### Install from Source / Virtual Environment

```bash
git clone https://github.com/your-username/myai.git
cd myai
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -e .
```

### Install with Optional Dependencies

```bash
pip install -e ".[train,serving,eval]"
```

---

## 🧭 Step-by-Step Workflow

### 1. Initialize Project & Set Goal

```bash
myai init my-assistant
cd my-assistant
```

This sets up `myai.yaml` with your project goal and evaluation weights.

### 2. Check Project Status Anytime

```bash
myai status
```

```text
┌────────────────────────────────────────────────────────┐
│ 📊 Project: my-assistant                               │
│ State: DATA_READY (2/5 stages complete)                │
│ Goal: chat / general                                   │
│ Dataset: 500 samples (train.jsonl)                     │
│ Next Step: Run 'myai train' or 'myai auto'             │
└────────────────────────────────────────────────────────┘
```

### 3. Add and Clean Data (Reference Mode)

```bash
# Register dataset source and run Tokenizer Analysis
myai data add ./data.jsonl

# Clean, deduplicate, scrub PII, and split train/val
myai data clean --fuzzy --val-split 0.1
```

### 4. Benchmark Hardware & Recommend Model

```bash
# Run live forward/training throughput probe
myai system check

# Get hardware-aware recommendation with transparent reasoning
myai recommend
```

### 5. Train & Track Experiments

```bash
# Manual training with live progress and auto-checkpointing
myai train --epochs 3 --lr 2e-4

# View experiment leaderboard and current release candidate
myai runs best
```

### 6. Run Autonomous Optimizer Loop

```bash
# Dry-run preview
myai optimize --dry-run

# Run up to 3 bounded optimization iterations
myai optimize --max-iters 3 --min-delta 2.0
```

### 7. Export Standalone Package

```bash
# Export the top release candidate (or specify --run <id>)
myai export
```

---

## 🚀 Running Your Exported Model

The exported `.zip` contains a self-contained runtime that requires **zero MYAI code**:

```text
my-assistant.myai.zip
├── model/                  # LoRA adapter weights (adapter_model.bin / safetensors)
├── tokenizer/              # Tokenizer configuration & vocab
├── metadata.json           # Model provenance, base repo, and goal details
├── evaluation.json         # Goal-weighted evaluation scores
├── loader.py               # Zero-framework inference helper
└── chat/
    ├── app.py              # Zero-dependency local web server entry point
    ├── ui.py               # Terminal fallback UI
    ├── config.json         # UI configuration & branding
    └── web/
        └── index.html      # Luminous Web Chat interface
```

### 1. Launch Luminous Web Chat UI (Default)

```bash
python chat/app.py
```

* Runs on built-in `http.server` (requires no `pip install flask` or `fastapi`).
* Opens a modern, dark-mode ambient interface in your browser.
* Includes animated thinking state, parameter controls, and message history.

### 2. Interactive Terminal Mode

```bash
python chat/app.py --cli
```

### 3. Direct One-Shot Query

```bash
python chat/app.py "What are the recommended rest intervals for hypertrophy?"
```

### 4. Programmatic Python Import

```python
from loader import ask

response = ask("Explain progressive overload in simple terms.")
print(response)
```

---

## 🛡️ 18-Point Security Gate

Before any export is written to disk, `myai` enforces strict containment:

| # | Security Check | Description |
| --- | --- | --- |
| **1** | Archive Integrity | Valid non-corrupt ZIP structure |
| **2** | Model Weights | `model/` directory with valid adapter binaries |
| **3** | Tokenizer | `tokenizer/` configuration present |
| **4** | Metadata | `metadata.json` present with base repo provenance |
| **5** | Evaluation | `evaluation.json` with verified metric scores |
| **6** | Documentation | Standalone `README.md` included |
| **7** | Portable Loader | Self-contained `loader.py` inference script |
| **8** | Standalone Chat | Full `chat/app.py`, `chat/ui.py`, `chat/web/index.html` runtime |
| **9** | **Source Isolation** | 🚫 Zero MYAI source code (`src/`, `myai/`) |
| **10** | **Version Control** | 🚫 Zero `.git/` directories or history |
| **11** | **Environment Files** | 🚫 Zero `.env` or environment configuration files |
| **12** | **Secret Scrubbing** | 🚫 Zero API keys (`sk-`, `ghp_`, `hf_`, `AKIA`) in metadata |
| **13** | **Dataset Privacy** | 🚫 Zero raw training datasets (`.jsonl`, `.csv`, `.parquet`) |
| **14** | **Model Isolation** | 🚫 Zero unrelated model weights |
| **15** | **Cache Cleanliness** | 🚫 Zero `__pycache__`, `.pyc`, `.DS_Store`, or temp files |
| **16** | **Host Path Privacy** | 🚫 Zero raw absolute host filesystem paths |
| **17** | **Traversal Protection** | 🚫 Zero path-traversal entries (`../`, absolute paths) |
| **18** | **Atomic Packaging** | Package generated and validated atomically |

---

## 🛠️ CLI Command Reference

| Command | Description |
| --- | --- |
| `myai init [name]` | Initialize a new project with interactive Goal Profile interview |
| `myai status` | Inspect project lifecycle status and recommended next steps |
| `myai system check` | Probe local CPU, RAM, GPU, VRAM, and throughput benchmark |
| `myai auto [--export] [--dry-run]` | **Autopilot**: Goal-to-Deployment autonomous pipeline |
| `myai data add <path> [--model]` | Register local dataset sources in Reference Mode & run tokenizer analysis |
| `myai data tokenize [--dataset] [--model] [--path]` | **Tokenizer Analysis**: Compute exact tokens, distributions & context fit |
| `myai data clean [--fuzzy] [--val-split]` | Clean, deduplicate, scrub PII, and split datasets |
| `myai data list` / `info` | Inspect registered datasets, sample counts, and quality scores |
| `myai recommend` | Hardware- and goal-aware base model recommendation |
| `myai train [--auto] [--epochs] [--lr]` | Train model with live metrics and auto-checkpointing |
| `myai runs list` / `info <id>` | List historical training runs and metric breakdowns |
| `myai runs best` / `leaderboard` | View experiment leaderboard and release candidate |
| `myai optimize [--dry-run] [--max-iters]` | Autonomous retrain/compare optimization loop |
| `myai export [--run <id>]` | Package release candidate through the 18-point Security Gate |
| `myai serve` / `myai ask` | Serve local model with Knowledge Gate RAG protection |

---

## 🧪 Comprehensive Test Suite

MYAI is verified by **194 automated tests** covering unit, integration, tokenizer streaming, security, adversarial, and end-to-end scenarios:

```bash
# Run all tests
pytest -v

# Run the dedicated 42-point security and reliability audit
pytest tests/test_security_audit.py -v

# Run the tokenizer analysis test suite
pytest tests/test_tokenizer_analysis.py -v
```

```text
============================= test session starts =============================
tests/test_auto_pipeline.py ...........                                 [  6%]
tests/test_cleaner.py .......                                           [  9%]
tests/test_data_cleaner.py ..........                                   [ 14%]
tests/test_dataset_scorer.py .......                                    [ 18%]
tests/test_export.py ................                                   [ 26%]
tests/test_feasibility.py .........                                     [ 31%]
tests/test_goal.py .........                                            [ 36%]
tests/test_hardware_benchmark.py ...                                    [ 37%]
tests/test_leaderboard.py ........                                      [ 41%]
tests/test_model_recommender.py ......                                  [ 44%]
tests/test_optimizer.py .......                                         [ 48%]
tests/test_post_training_ui.py .........                                [ 53%]
tests/test_scorer.py ......                                             [ 56%]
tests/test_security_audit.py .......................................... [ 77%]
tests/test_state.py ...                                                 [ 79%]
tests/test_strategy.py .....                                            [ 81%]
tests/test_tokenizer_analysis.py ..........................             [ 95%]
tests/test_training_engine.py ..........                                [100%]

============================ 194 passed in 16.99s =============================
```

---

## 📜 License

Distributed under the **Apache 2.0** License. See [LICENSE](file:///e:/CMD%20GitHub/myai/LICENSE) for details.
