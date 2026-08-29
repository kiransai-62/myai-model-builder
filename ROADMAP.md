# 🗺️ MYAI Public Roadmap

This roadmap outlines the planned capabilities and milestone releases for **MYAI**.

---

## 📍 Release Milestones

### 🚀 v0.1.0 — Foundation & Autopilot (Current)

- [x] **Goal Understanding**: Interactive Goal Profile interview & goal-weighted evaluation formulas.
- [x] **Hardware Feasibility**: Live throughput probing, compute tiers (`T0`–`T3`), empirical VRAM estimation.
- [x] **Exact Layer Streaming**: Stream frozen base layers from RAM to VRAM buffer (8B on 4GB laptop GPUs, peak VRAM ~3.32 GB).
- [x] **Dataset Intelligence**: Strict Reference Mode, PII & credential redaction, exact/fuzzy deduplication.
- [x] **Tokenizer Engine**: Model-aware streaming tokenization, sequence histograms, context-fit analysis.
- [x] **Training & Alignment Engine**: LoRA/QLoRA, live progress monitoring, plus **DPO, ORPO, SimPO, and KTO** preference optimization.
- [x] **Deterministic Reward & Verifier Synthesis**: `myai reward synth` for automatic calibrated verifier generation (`numeric`, `json_schema`, `regex`, `tool_call`).
- [x] **Leg-2 Regression Shipping Gate**: `myai ship` with 4 bundled offline test suites and cryptographic evidence receipts.
- [x] **Experiment Leaderboard**: Goal-weighted matrix scoring, regression stability gating, Release Candidate designation.
- [x] **Autonomous Optimizer**: Deficiency diagnosis, targeted parameter mutation, bounded retrain/compare loops.
- [x] **Autopilot Pipeline**: Full autonomous `myai auto --export` execution.
- [x] **18-Point Export Gate**: Containment verification & standalone ZIP package with zero-dependency Luminous Web Chat UI.
- [x] **Multi-Target LoRA Merge & GGUF / Ollama Export**: Standalone merged weights & Ollama `Modelfile` generation.
- [x] **RAG Knowledge Gate**: Local document indexing and retrieval guardrails.

---

### 🌟 v0.2.0 — Expanded Quantization & Architectures (Q4 2026)

- [ ] **Multi-Turn Chat Evaluator**: Dynamic multi-turn evaluation personas for interactive conversational testing.
- [ ] **Extended Context Packing**: Advanced sample packing algorithms for long-context datasets (up to 32k tokens).
- [ ] **Automated Synthetic Data Expansion**: Offline few-shot data generation to augment small datasets (< 100 samples).

---

### ⚡ v0.3.0 — Distributed & Multi-GPU Acceleration (Q1 2027)

- [ ] **Multi-GPU DDP / FSDP**: Fully Sharded Data Parallel training support across multi-GPU setups.
- [ ] **Vision-Language Model (VLM) Fine-Tuning**: Support for multimodal image + text fine-tuning.

---

### 🏆 v1.0.0 — Enterprise Grade Local AI Studio (Q2 2027)

- [ ] **Desktop GUI Wrapper**: Optional native desktop application wrapping the CLI engine.
- [ ] **Hardware Orchestration**: Automated mesh clustering across heterogeneous local workstations.
- [ ] **Air-Gapped Compliance Verification**: Automated audit reports for ISO/SOC2 air-gapped environments.
