# 🛡️ MYAI Security Policy

## Supported Versions

Security updates and patches are provided for the latest active release line. We recommend keeping your MYAI installation updated to the newest version.

| Version | Supported          | Status |
| ------- | ------------------ | ------ |
| 0.1.x   | :white_check_mark: | Active release line |
| < 0.1   | :x:                | Unsupported |

---

## 🔒 Core Security Invariants

MYAI is engineered from the ground up to be **local-first and privacy-preserving**. We design around four core security invariants across every subsystem:

1. **Local Compute Execution**: Data processing, tokenization, training, evaluation, and inference execute on your local machine without mandatory cloud connections or remote telemetry.
2. **Strict Reference Mode**: MYAI treats user raw datasets as strictly immutable. Source files are read in-place and never altered or deleted.
3. **Automated Secret & PII Sanitization**: The dataset cleaner scans and redacts API keys (`sk-`, `ghp_`, `hf_`, `AKIA`), credentials, emails, and phone numbers before training tokens are processed.
4. **Export Containment Gate**: Model exports are validated through automated verification checks before any archive is written to disk.

---

## 🚪 Export Containment & Security Gate

Before packaging any model into a standalone archive, MYAI runs automated containment checks:

| Category | Security / Policy Check | Requirement |
| --- | --- | --- |
| **Integrity** | Archive Integrity | Verified non-corrupt ZIP header and structure |
| **Artifacts** | Model Weights & Tokenizer | Valid adapter weights, configuration, and tokenizer vocabulary files |
| **Provenance** | Provenance Manifest & Audit | `metadata.json` and `evaluation.json` record training configuration and evaluation scores |
| **Runtime** | Portable Loader & Chat UI | Self-contained Python loader and zero-dependency Luminous Web Chat runtime |
| **Containment** | Source & Version Control Isolation | Excludes internal framework source code (`src/`, `myai/`) and `.git/` histories |
| **Security** | Environment & Secret Isolation | Excludes `.env` files and redacts API keys (`sk-`, `ghp_`, `hf_`, `AKIA`) |
| **Privacy** | Dataset Privacy | Excludes raw training datasets (`.jsonl`, `.csv`, `.parquet`, `.txt`) from deployment packages |
| **Hygiene** | Cache & Path Privacy | Cleans temporary cache files (`__pycache__`) and prevents absolute host paths or traversal entries |

---

## 🎯 Threat Model & Scope

MYAI is designed for local-first developer execution. The threat model assumes the operator executes MYAI on their own machine with their own private datasets.

### In-Scope Security Vulnerabilities
- **Path Traversal & Arbitrary File Access**: Vulnerabilities allowing unintended reads/writes from user-supplied dataset paths, configuration YAMLs, or exported archive extractions.
- **Secret & PII Exposure**: Accidental exposure of API tokens, credentials, or private data in logs, training checkpoints, provenance receipts, or export packages.
- **Injection Attacks**: Command injection, Ollama Modelfile injection, Jinja chat-template injection, or shell metacharacter manipulation via CLI flags or config fields.
- **Safe Execution Sandbox**: Ensuring reward verifier synthesis and execution remain strictly deterministic and safe.
- **Resource Exhaustion & Memory Safety**: Predictable memory allocation avoiding uncontrolled disk or memory exhaustion.

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
