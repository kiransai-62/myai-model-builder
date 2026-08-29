# 📜 Changelog

All notable changes to **MYAI** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-08-30

### Added
- **Goal Understanding (Stage A)**: Interactive Goal Profile interview & domain-weighted composite evaluation formulas.
- **Dataset Intelligence & Cleaning (Stage B)**: Strict Reference Mode reader, PII & credential scrubbing, exact/fuzzy deduplication, train/val contamination isolation.
- **Model-Aware Tokenizer Engine**: Streaming record tokenizer, calibrated offline fallback, 7 distribution buckets, and context-window fit vs. overflow analyzer.
- **Dual-Gate Feasibility (Stage C)**: Live forward/training throughput prober, compute tier categorization (`T0`–`T3`), empirical VRAM estimation, and closed-loop auto-downgrade.
- **Training Engine (Stage E)**: LoRA and QLoRA fine-tuning with live terminal loss curves, progress bars, and resumable checkpoints.
- **Goal-Weighted Leaderboard (Stage F)**: Composite score calculation, regression penalty stability gate, and Release Candidate promotion.
- **Autonomous Optimizer Loop (Stage G)**: Metric deficiency diagnosis, hyperparameter mutation, and bounded retrain/compare loops.
- **Autopilot Capstone**: End-to-end autonomous `myai auto --export` execution pipeline.
- **18-Point Containment Gate (Stage H)**: Verification gate ensuring zero MYAI framework code, `.git/`, `.env`, or credentials in exported archives.
- **Zero-Dependency Luminous Web & CLI Chat UI**: Standalone browser and terminal inference interface running on native Python `http.server`.
- **Knowledge Gate (RAG)**: Local document indexing, embedding, chunking, and retrieval-augmented generation.
- **Comprehensive Verification Suite**: 194 automated unit, integration, streaming, adversarial, and security audit tests.
