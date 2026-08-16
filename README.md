# myai 🧠

**The local-first AI model builder with instant standalone chat export.**

Bring your data. `myai` analyzes your hardware, recommends the optimal model, fine-tunes it locally, runs comprehensive evaluation, and packages the model into a standalone, portable ZIP with its own Web & CLI Chat UI.

No cloud GPUs required. No framework lock-in.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![E2E](https://img.shields.io/badge/E2E-verified-brightgreen)
![Package Validation](https://img.shields.io/badge/package%20checks-18%2F18%20passing-brightgreen)

---

## 💻 Installation & Setup

### Requirements

Before installing MYAI, make sure your system has:

* **Python**: 3.11+
* **OS**: Windows, Linux, or macOS
* **Hardware**: Sufficient RAM and disk storage for your selected model
* **GPU (Optional)**: NVIDIA GPU with CUDA for accelerated training (CPU training supported on Tier T1)
* **Privacy**: MYAI does not upload your training dataset to a MYAI cloud service. Training and dataset processing run locally.

### Install from PyPI

```bash
python -m pip install myai
```

### Quick Verification

Verify the CLI and inspect your local hardware:

```bash
myai --help
myai system check
```

```text
Installation ➔ CLI detected ✓ ➔ Hardware detected ✓ ➔ Ready to create a project
```

> [!TIP]
> **Windows PowerShell PATH note:** If PowerShell reports `The term 'myai' is not recognized`, your Python `Scripts` directory may not be in your `PATH`.
> Locate the scripts directory with:
>
> ```powershell
> python -c "import sysconfig; print(sysconfig.get_path('scripts'))"
> ```
>
> Add that path to your User Environment Variables and restart PowerShell.

---

## 🚀 What You Get

`myai` cleanly separates **model creation** from **model usage**:

| Phase | Responsibility | Scope |
| :--- | :--- | :--- |
| **MYAI Framework** | • Prepare & validate local datasets<br>• Analyze hardware (CPU, RAM, GPU, VRAM)<br>• Recommend optimal base model<br>• Train, fine-tune & auto-evaluate<br>• Export standalone model bundle | Development & Training machine |
| **Exported ZIP** | • Trained LoRA adapter weights<br>• Tokenizer configuration<br>• Provenance & evaluation report<br>• Standalone Web & CLI Chat UI<br>• Self-contained loader script | Any compatible machine with Python & required runtime |

```text
Your Trained Model  +  Tokenizer  +  Metadata & Evaluation  +  Standalone Chat UI
                                     ═
                      Portable AI Application (.zip)
```

---

## ✨ The Complete Workflow

```bash
# 1. Initialize your project
myai init my-assistant && cd my-assistant

# 2. Check local hardware compatibility
myai system check

# 3. Add your domain data in-place (JSON, CSV, JSONL)
myai data add ./data.jsonl

# 4. Get hardware-aware model recommendation
myai recommend

# 5. Train with live metrics, storage budget & auto-evaluation
myai train

# 6. Post-training: MODEL READY UI automatically guides you to export
# Or export anytime via:
myai export
```

```text
Training ➔ Evaluation ➔ MODEL READY ✓ ➔ Export Package ➔ My-Custom-Model.zip
```

---

## 📦 Exported Package Structure

When you export your trained model, `myai` packages **only the model artifacts and standalone runtime** into a clean ZIP file:

```text
My-Custom-Model.zip
├── model/
│   ├── adapter_config.json
│   └── adapter_model.bin
├── tokenizer/
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── metadata.json              # Model provenance, base repo, and run details
├── evaluation.json            # Full evaluation scores & validation status
├── README.md                  # Quick-start documentation
├── loader.py                  # Standalone loader and inference helper
└── chat/
    ├── app.py                 # Standalone Web & CLI server entry point
    ├── ui.py                  # Minimal terminal interface fallback
    ├── config.json            # Chat application configuration
    └── web/
        └── index.html         # Luminous Web Chat UI (Vanilla HTML/CSS/JS)
```

---

## 💬 Standalone Chat UI Experience

The exported model package is completely decoupled from the MYAI framework. Extract it on another compatible machine and run it independently of MYAI:

### 1. Luminous Web Chat UI (Default)

```bash
python chat/app.py
```

* Starts a local HTTP server using Python's built-in `http.server` (**requires no additional web-server packages**).
* Opens the browser with a modern **Luminous ambient-glow interface**.
* Features live inference, animated thinking state (`✦ Thinking...`), model metadata badge, and clean message history.

### 2. Terminal Interactive Mode

```bash
python chat/app.py --cli
```

### 3. One-Shot Script Query

```bash
python chat/app.py "What is our company's refund policy?"
```

### 4. Programmatic Python Import

```python
from loader import ask

response = ask("How do I track my order?")
print(response)
```

---

## 🛡️ Strict Product Boundaries & Safety

Every exported model ZIP is validated against an **18-point verification gate** before it is marked as ready for the user:

* ✅ **Standalone Package**: Model adapter, tokenizer, metadata, evaluation report, loader, and customized Web & CLI Chat UI are included.
* 🚫 **No MYAI Framework Source**: Internal MYAI code (`myai/cli/`, `myai/core/`, `myai/training/`, `myai/evaluation/`, `myai/export/`, `src/`) is excluded.
* 🚫 **No Secrets or Credentials**: `.env` files, API keys, tokens, passwords, private keys, and absolute machine paths are detected and rejected.
* 🚫 **No Raw Datasets**: Original training datasets remain private on the training machine and are not packaged.
* 🚫 **No Development Files**: `.git/`, `.pyc`, `__pycache__/`, temporary files, and unrelated development artifacts are excluded.
* 🚫 **No Unsafe Paths**: Path-traversal entries such as `../` are detected and rejected.
* 🔒 **Validated Before Release**: The ZIP is only reported as successfully exported after package creation and validation complete successfully.

---

## 🧪 Quality & Verification

The framework is tested across three tiers of verification:

* **18-point Package Validation**: Automatically inspects every exported ZIP for integrity, cleanliness, and security boundaries.
* **34-test Unit & Integration Suite**: Automated test coverage for data scanning, training engine, evaluation metrics, and export packaging.
* **55-test Master E2E Suite**: Full lifecycle end-to-end verification (55/55 passed) covering project creation, hardware detection, training, evaluation, export, and isolated standalone runtime testing.

---

## ⚙️ Core CLI Commands

Commands verified against the current MYAI CLI implementation:

| Command | Description |
| :--- | :--- |
| `myai init <project>` | Initialize a new local AI model project |
| `myai system check` | Analyze local CPU, RAM, GPU, VRAM, and storage tier |
| `myai data add <path>` | Scan and register local datasets in-place |
| `myai data validate` | Validate training example count and token estimates |
| `myai data list` | List all registered local datasets |
| `myai data info [id]` | Show detailed dataset metadata and validation report |
| `myai recommend` | Hardware-aware model selection (e.g. Qwen 2.5, Llama 3, SmolLM) |
| `myai model list` | View available base models in registry |
| `myai train` | Train model with live progress, resume capability, and evaluation |
| `myai evaluate` | Run evaluation suite against held-out test cases |
| `myai export [id]` | Export model into standalone portable ZIP with Chat UI |
| `myai runs list` | Inspect training run history and checkpoints |
| `myai runs info <id>` | Inspect detailed provenance for a training run |

---

## 📜 License

Apache 2.0
