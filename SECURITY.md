# 🛡️ MYAI Security Policy

## 🔒 Security Principles

MYAI is designed from the ground up to be **local-first, privacy-preserving, and air-gapped**. We operate on four core security invariants:

1. **Zero External Data Leakage**: All data processing, tokenization, training, evaluation, and inference happen 100% locally on your machine. No telemetry or external cloud logging is performed.
2. **Strict Reference Mode**: MYAI treats user raw datasets as strictly immutable. Source files are read in-place and **never modified, deleted, or relocated**.
3. **Automated Secret & PII Sanitization**: The dataset cleaner continuously scans and redacts API keys, credentials, emails, and phone numbers before training tokens are processed.
4. **18-Point Containment Gate**: Every model export is validated through an 18-point verification gate before any ZIP archive is written to disk.

---

## 🚪 The 18-Point Containment & Security Gate

Before packaging any model into a standalone archive, MYAI runs 18 automated security checks:

| # | Security Check | Policy / Requirement |
| --- | --- | --- |
| **1** | **Archive Integrity** | Verified non-corrupt ZIP header and table of contents. |
| **2** | **Model Weights** | Valid adapter binaries and configuration present in `model/`. |
| **3** | **Tokenizer Vocab** | Tokenizer configuration and vocabulary files verified. |
| **4** | **Provenance Metadata** | `metadata.json` records base model repo and goal intent. |
| **5** | **Evaluation Report** | `evaluation.json` contains verified goal-weighted metrics. |
| **6** | **Standalone Documentation** | Embedded `README.md` with instructions for zero-dependency execution. |
| **7** | **Zero-Framework Loader** | Self-contained `loader.py` Python inference script. |
| **8** | **Embedded Chat UI** | Built-in Luminous Web & CLI Chat runtime (`chat/app.py`, `chat/web/`). |
| **9** | **Source Code Isolation** | 🚫 **Zero MYAI framework source code** (`src/`, `myai/`) allowed in package. |
| **10** | **Version Control Isolation** | 🚫 **Zero `.git/` directories** or commit histories included. |
| **11** | **Environment Isolation** | 🚫 **Zero `.env` or system environment files** packaged. |
| **12** | **Secret Scrubbing** | 🚫 **Zero API keys** (`sk-`, `ghp_`, `hf_`, `AKIA`) in metadata or configs. |
| **13** | **Dataset Privacy** | 🚫 **Zero raw training datasets** (`.jsonl`, `.csv`, `.parquet`, `.txt`) included. |
| **14** | **Model Isolation** | 🚫 **Zero unrelated model checkpoints** or foreign weights. |
| **15** | **Cache Cleanliness** | 🚫 **Zero `__pycache__`**, `.pyc`, `.DS_Store`, or temporary files. |
| **16** | **Host Path Privacy** | 🚫 **Zero absolute host filesystem paths** exposed in metadata. |
| **17** | **Path Traversal Protection** | 🚫 **Zero path traversal entries** (`../` or leading slashes) in archive members. |
| **18** | **Atomic Generation** | Package assembled in isolated staging directory and validated before release. |

---

## 🔍 Reporting a Vulnerability

If you discover a security issue or vulnerability in MYAI, please report it responsibly:

* **Email**: `security@myai.local` (or open a private security advisory on GitHub).
* Please include:
  * Description of the vulnerability.
  * Steps to reproduce or proof-of-concept.
  * Potential impact on local data or model exports.

We aim to acknowledge reports within 48 hours and provide patches promptly.
