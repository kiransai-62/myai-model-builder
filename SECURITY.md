# 🛡️ MYAI Security Policy

## Supported Versions

Security updates and patches are provided for the latest active release line. We recommend keeping your MYAI installation updated to the newest version.

| Version | Supported          | Status |
| ------- | ------------------ | ------ |
| 0.1.x   | :white_check_mark: | Active production release line |
| < 0.1   | :x:                | Unsupported |

---

## 🔒 Core Security Invariants

MYAI is engineered from the ground up to be **100% local-first, privacy-preserving, and air-gapped**. We enforce four immutable security invariants across every subsystem:

1. **Zero External Data Leakage**: All data processing, tokenization, training, evaluation, and inference execute entirely on your local machine. No telemetry, crash reporting, or external API calls are made without explicit user action.
2. **Strict Reference Mode**: MYAI treats user raw datasets as strictly immutable. Source files are read in-place and **never modified, deleted, or relocated** ($MD5_{\text{before}} = MD5_{\text{after}}$).
3. **Automated Secret & PII Sanitization**: The dataset cleaner continuously scans and redacts API keys (`sk-`, `ghp_`, `hf_`, `AKIA`), credentials, emails, and phone numbers before training tokens are processed.
4. **18-Point Containment Gate**: Every model export is validated through an automated 18-point verification gate before any ZIP archive is written to disk.

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

## 🎯 Threat Model & Scope

MYAI is an air-gapped, local-first CLI and runtime. The threat model assumes the operator executes MYAI on their own machine with their own private datasets.

### In-Scope Security Vulnerabilities
- **Path Traversal & Arbitrary File Access**: Vulnerabilities allowing unintended reads/writes from user-supplied dataset paths, configuration YAMLs, or exported archive extractions.
- **Secret & PII Leakage**: Accidental exposure of API tokens, SSH keys, credentials, or private data in logs, training checkpoints, provenance receipts, or export packages.
- **Injection Attacks**: Command injection, Ollama Modelfile injection, Jinja chat-template injection, or shell metacharacter manipulation via CLI flags or config fields.
- **Safe Code Execution / Reward Sandbox Escape**: Any arbitrary code execution in reward verifier synthesis (`myai reward synth`), ensuring all verifiers remain purely deterministic without unsafe dynamic evaluation.
- **Resource Exhaustion & Memory Safety**: Predictable CUDA OOM crashes or uncontrolled disk allocation circumventing feasibility and storage budget guards.

### Out-of-Scope
- Vulnerabilities in third-party model weights or untrusted datasets explicitly downloaded by the operator from upstream external sources.
- Attacks requiring physical access to an already-compromised host system.
- Standard denial of service resulting from hardware failure or underlying OS kernel bugs.

---

## 🔍 Reporting a Vulnerability

Please report security issues **privately** — do not open a public GitHub issue or discuss security-sensitive matters in public channels.

- **Preferred**: Open a private report via **GitHub Security Advisories**.
- **Email**: Reach out to the security team at **security@myai.local**.

### What to Include
To help us triage and patch the issue promptly, please provide:
1. The affected MYAI version and operating system (Windows, Linux, macOS).
2. A minimal reproducible example or proof-of-concept.
3. The observed impact and potential attack vector.

---

## 🤝 Coordinated Disclosure & Credit

We practice coordinated vulnerability disclosure. Once a fix has been validated and released, we will publish a security advisory and credit the reporter by name in the release notes (unless anonymity is requested).
