# 💻 MYAI CLI Reference Manual

Welcome to the comprehensive command-line reference for **MYAI** — the local-first autonomous AI model builder and packager.

---

## 📑 Table of Contents

- [Global Options & Syntax](#-global-options--syntax)
- [1. Project Initialization & Lifecycle](#1-project-initialization--lifecycle)
  - [`myai init`](#myai-init)
  - [`myai status`](#myai-status)
- [2. System & Hardware Benchmarking](#2-system--hardware-benchmarking)
  - [`myai system check`](#myai-system-check)
- [3. Dataset Intelligence & Tokenizer Analysis](#3-dataset-intelligence--tokenizer-analysis)
  - [`myai data add`](#myai-data-add)
  - [`myai data tokenize`](#myai-data-tokenize)
  - [`myai data clean`](#myai-data-clean)
  - [`myai data validate`](#myai-data-validate)
  - [`myai data list`](#myai-data-list)
  - [`myai data info`](#myai-data-info)
- [4. Model Recommendation & Feasibility](#4-model-recommendation--feasibility)
  - [`myai recommend`](#myai-recommend)
- [5. Model Training & Experiment Tracking](#5-model-training--experiment-tracking)
  - [`myai train`](#myai-train)
  - [`myai runs list`](#myai-runs-list)
  - [`myai runs info`](#myai-runs-info)
  - [`myai runs best` / `leaderboard`](#myai-runs-best--leaderboard)
- [6. Autonomous Optimizer Loop](#6-autonomous-optimizer-loop)
  - [`myai optimize`](#myai-optimize)
- [7. Autopilot (End-to-End Autonomous Pipeline)](#7-autopilot-end-to-end-autonomous-pipeline)
  - [`myai auto`](#myai-auto)
- [8. Security Gate & Standalone Export](#8-security-gate--standalone-export)
  - [`myai export`](#myai-export)
- [9. Serving & Knowledge Gate (RAG)](#9-serving--knowledge-gate-rag)
  - [`myai serve`](#myai-serve)
  - [`myai ask`](#myai-ask)
  - [`myai index`](#myai-index)

---

## 🌐 Global Options & Syntax

```bash
myai [OPTIONS] COMMAND [ARGS]...
```

| Option | Description |
| --- | --- |
| `--help` | Show top-level help message and exit. |
| `--install-completion` | Install shell completion for bash, zsh, fish, or powershell. |
| `--show-completion` | Show shell completion script. |

---

## 1. Project Initialization & Lifecycle

### `myai init`

Initializes a new MYAI project directory and conducts an interactive **Goal Profile** interview to capture your task, domain, and priority weights.

```bash
myai init [PROJECT_NAME]
```

#### Arguments & Options

* `PROJECT_NAME` *(Optional)*: Name of the project directory to create. If omitted, defaults to `my-model`.

#### Example

```bash
myai init fittrack-coach
```

#### Interactive Interview Prompts

```text
1. What is the primary task of this AI?
   [1] instruction-tuning  [2] chat  [3] domain-qa  [4] code  [5] summarization

2. What domain will this AI operate in?
   [1] general  [2] medical  [3] finance  [4] fitness  [5] legal  [6] customer-support

3. Architectural Priorities:
   Context length priority: [short / balanced / long-context]
   Latency vs Quality:      [fast / balanced / high-quality]
   Target deployment:       [edge / local-cpu-gpu / server]
```

---

### `myai status`

Inspects the current project's lifecycle state, attached dataset, trained model status, leaderboard rankings, and outputs the next recommended command.

```bash
myai status
```

#### Lifecycle States

* `INITIALIZED`: Project created, awaiting dataset registration.
* `DATA_READY`: Dataset validated, cleaned, and attached.
* `TRAINED`: At least one training run completed.
* `OPTIMIZED`: Optimizer loop executed and evaluated.
* `EXPORTED`: Release candidate validated and packaged into standalone ZIP.

---

## 2. System & Hardware Benchmarking

### `myai system check`

Scans local hardware (CPU cores, RAM, NVIDIA GPU, VRAM, disk space) and runs a live throughput benchmark to classify your compute tier (`T0` to `T3`).

```bash
myai system check
```

#### Compute Tiers

* `T0`: Pure CPU / Low RAM (< 8 GB) — ultra-compact model execution.
* `T1`: Standard CPU (8–16+ GB RAM) — CPU-quantized fine-tuning.
* `T2`: Mid-tier GPU (4–8 GB VRAM) — 4-bit QLoRA fine-tuning.
* `T3`: High-tier GPU (12–24+ GB VRAM) — 4-bit/8-bit/16-bit LoRA fine-tuning.

---

## 3. Dataset Intelligence & Tokenizer Analysis

### `myai data add`

Registers a local dataset in **Strict Reference Mode** (the original source file is never modified) and runs an automatic Tokenizer Analysis report.

```bash
myai data add <PATH> [OPTIONS]
```

#### Options

* `--model, -m TEXT`: Target base model for model-aware tokenization (default: `Qwen/Qwen2.5-1.5B-Instruct`).
* `--yes, -y`: Automatically accept confirmation prompts.

#### Example

```bash
myai data add "./data/fitness_conversations.jsonl" --model "meta-llama/Llama-3.2-3B-Instruct"
```

---

### `myai data tokenize`

Runs a deep **Tokenizer Analysis** on a file, folder, or registered dataset. Computes exact token lengths, character/word counts, sequence length distribution histograms, and context-window fit analysis.

```bash
myai data tokenize [OPTIONS]
```

#### Options

* `--path, -p PATH`: Direct path to a file or folder (`.json`, `.jsonl`, `.csv`, `.txt`).
* `--dataset, -d TEXT`: Dataset ID registered in the project.
* `--model, -m TEXT`: Target model repo ID or shorthand name.
* `--refresh, -r`: Invalidate cache and force a fresh tokenization run.

#### Example

```bash
myai data tokenize --path "./dataset.jsonl" --model "Qwen/Qwen2.5-1.5B-Instruct"
```

---

### `myai data clean`

Cleans and prepares data in **Strict Reference Mode**. Redacts PII/secrets, deduplicates records, detects train/val leakage, and creates isolated cleaned splits in `.myai/`.

```bash
myai data clean [OPTIONS]
```

#### Options

* `--fuzzy / --no-fuzzy`: Enable fuzzy deduplication using string similarity metrics (default: `True`).
* `--val-split FLOAT`: Fraction of dataset reserved for validation (default: `0.1`).
* `--seed INTEGER`: Random seed for deterministic reproducibility (default: `42`).

---

### `myai data validate`

Validates dataset syntax, supported schemas (`instruction`, `prompt_response`, `chat`, `text`), duplicate sample counts, and approximate token size.

```bash
myai data validate [PATH]
```

---

### `myai data list`

Lists all datasets registered across MYAI workspaces.

```bash
myai data list
```

---

### `myai data info`

Displays provenance metadata, file location, total sample count, and quality score for a specific dataset ID.

```bash
myai data info <DATASET_ID>
```

---

## 4. Model Recommendation & Feasibility

### `myai recommend`

Cross-references your project's **Goal Profile**, available **Hardware Tier**, and **Dataset Size** to recommend the optimal base model with transparent reasoning.

```bash
myai recommend [OPTIONS]
```

#### Options

* `--json`: Output raw recommendation metadata in JSON format.

---

## 5. Model Training & Experiment Tracking

### `myai train`

Launches the interactive 5-step training wizard or runs scripted fine-tuning with live terminal progress, real-time loss tracking, and automatic checkpointing.

```bash
myai train [OPTIONS]
```

#### Options

* `--data, -d PATH`: Skip Step 1 by providing the data path directly.
* `--model, -m TEXT`: Explicitly specify the base model ID.
* `--epochs, -e INTEGER`: Number of training epochs (default: `3`).
* `--lr FLOAT`: Learning rate (e.g. `2e-4`).
* `--batch-size, -b INTEGER`: Per-device batch size.
* `--method TEXT`: Fine-tuning method (`lora` or `qlora`).
* `--auto, -a`: Launch full autonomous build (Autopilot mode).
* `--dry-run`: Preview training parameters and feasibility without executing.
* `--yes, -y`: Accept confirmation prompts automatically.

#### Example

```bash
myai train --epochs 3 --lr 0.0002 --method qlora
```

---

### `myai runs list`

Lists historical training runs in the current project, showing run IDs, base models, final training loss, and elapsed time.

```bash
myai runs list
```

---

### `myai runs info`

Displays detailed hyperparameters, strategy configuration, and evaluation metrics for a specific run.

```bash
myai runs info <RUN_ID>
```

---

### `myai runs best` / `leaderboard`

Displays the goal-weighted experiment leaderboard. Ranks runs by composite score, applies regression penalties, and marks the current **Release Candidate**.

```bash
myai runs best
# or
myai leaderboard
```

---

## 6. Autonomous Optimizer Loop

### `myai optimize`

Analyzes metric deficiencies in the current Release Candidate, prescribes minimal strategy mutations (adjusting learning rate, LoRA rank, sequence length, or epochs), and executes a retrain/compare loop.

```bash
myai optimize [OPTIONS]
```

#### Options

* `--max-iters INTEGER`: Maximum optimization rounds (default: `2`).
* `--min-delta FLOAT`: Minimum composite score improvement required to promote a run (default: `1.0`).
* `--dry-run`: Preview diagnostic mutations without running training.

#### Example

```bash
myai optimize --max-iters 3 --min-delta 2.0
```

---

## 7. Autopilot (End-to-End Autonomous Pipeline)

### `myai auto`

Executes the complete autonomous **Goal-to-Deployment** pipeline in a single command:

$$\text{Goal} \rightarrow \text{Hardware} \rightarrow \text{Data} \rightarrow \text{Model} \rightarrow \text{Feasibility} \rightarrow \text{Train} \rightarrow \text{Eval} \rightarrow \text{Optimize} \rightarrow \text{Export}$$

```bash
myai auto [OPTIONS]
```

#### Options

* `--export / --no-export`: Automatically package the release candidate through the 18-point Security Gate upon completion (default: `True`).
* `--dry-run`: Validate the end-to-end plan and print stages without modifying state or training.
* `--model, -m TEXT`: Override the recommended base model.
* `--opt-iters INTEGER`: Maximum optimizer iterations (default: `2`).
* `--override, -o KEY=VALUE`: Strategy overrides (e.g. `-o epochs=5 -o lr=1e-4`).

#### Example

```bash
myai auto --export
```

---

## 8. Security Gate & Standalone Export

### `myai export`

Packages the active Release Candidate through the **18-Point Containment & Security Gate** into a standalone `.myai.zip` archive containing a zero-dependency Luminous Web & CLI Chat UI.

```bash
myai export [OPTIONS]
```

#### Options

* `--run, -r TEXT`: Specify a run ID to export (default: active Release Candidate).
* `--output, -o PATH`: Output directory or filename for the `.zip` package.
* `--yes, -y`: Automatically confirm export.

#### Example

```bash
myai export --output ./dist/fitness-coach.myai.zip
```

---

## 9. Serving & Knowledge Gate (RAG)

### `myai serve`

Launches a local HTTP inference server for your trained model with **Knowledge Gate** hallucination and retrieval guardrails.

```bash
myai serve [OPTIONS]
```

#### Options

* `--port, -p INTEGER`: Server port (default: `8000`).
* `--host TEXT`: Host binding address (default: `127.0.0.1`).
* `--gate / --no-gate`: Enable Knowledge Gate RAG validation (default: `True`).

---

### `myai ask`

Executes a single prompt query against the trained model in the current project.

```bash
myai ask "What is the recommended protein intake for endurance athletes?"
```

---

### `myai index`

Manages local document indexes for retrieval-augmented generation (RAG) with the Knowledge Gate.

```bash
# Add documents to knowledge index
myai index add ./docs/nutrition_guide.pdf

# List active knowledge indexes
myai index list
```

---

## 📖 Best Practices

1. **Always verify status**: Run `myai status` whenever you re-enter a project directory to see completed stages and next recommendations.
2. **Use Reference Mode**: Keep raw datasets in their original directories; `myai data add` registers them without duplicating or altering original files.
3. **Preview with `--dry-run`**: Use `--dry-run` with `myai auto` or `myai optimize` to inspect planned mutations and VRAM feasibility before starting compute-heavy training.
