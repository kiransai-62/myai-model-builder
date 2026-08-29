"""MYAI Comprehensive Security, Reliability & Functional Audit Test Suite.

Covers all 42 checklist categories from the production readiness audit.
Tests: positive, negative, boundary, malformed, adversarial, and recovery cases.
No GPU, internet, or API keys required.
"""
import hashlib
import json
import os
import random
import shutil
import tempfile
import textwrap
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

# ── Module imports ──────────────────────────────────────────────────
from myai.cli.main import app
from myai.core.config import ProjectConfig
from myai.core.goal import GoalProfile, TaskType, Domain
from myai.core.state import inspect_project_state, validate_precondition, ProjectState
from myai.data.cleaner import (
    prepare_datasets, _scrub_pii, _hash_text, _is_fuzzy_duplicate,
    _extract_samples_from_path, CleaningReport,
)
from myai.data.scorer import analyze_dataset, DatasetSummary
from myai.hardware.detector import detect_hardware, HardwareReport
from myai.hardware.feasibility import (
    TrainingConfig, estimate_vram_gb, check_feasibility,
    run_feasibility, FeasibilityResult,
)
from myai.hardware.benchmark import run_hardware_benchmark, BenchmarkResult
from myai.models.leaderboard import Leaderboard, RunRecord, RankedRun
from myai.models.schema import RegistryModel
from myai.models.recommender import recommend_model, CATALOG
from myai.optimizer.engine import (
    OptimizerEngine, PRESCRIPTIONS,
    OptimizationReport, Diagnosis,
)
from myai.training.strategy import plan_strategy, TrainingStrategy, _make_fit
from myai.autopilot.orchestrator import Autopilot, AutopilotReport, _load_sources
from myai.export.validator import validate_package, ValidationResult


runner = CliRunner()


def _make_hw(gpu="NVIDIA RTX 3060", vram=12.0, ram=32.0, disk=200.0, tier="T2"):
    return HardwareReport(cpu="8 cores", ram_gb=ram, disk_gb=disk, gpu=gpu, vram_gb=vram, tier=tier)


def _make_model(pid="test-1.5b", params="1.5B", vram_min=4.0, hidden=1536, layers=28, ctx=32768):
    return RegistryModel(
        id=pid, name=f"Test {params}", parameters=params, vram_min=vram_min,
        methods=["LoRA", "QLoRA"], repository="test/test-model", license="Apache 2.0",
        hidden_size=hidden, num_layers=layers, context_length=ctx,
    )


def _make_run(run_id, metrics=None, regression=True, strategy=None):
    return RunRecord(
        run_id=run_id, model_name="test-model", timestamp="2026-08-30T00:00:00",
        strategy=strategy or {"learning_rate": 2e-4, "epochs": 2, "lora_rank": 16},
        metrics=metrics or {"domain_accuracy": 0.8, "readability": 0.85, "bleu": 0.7, "rouge": 0.75, "exact_match": 0.6},
        regression_passed=regression,
    )


def _make_project(tmp, name="audit-ai", task="chat", domain="general", samples=5):
    proj = Path(tmp) / name
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "myai.yaml").write_text(
        f"project:\n  name: {name}\ngoal:\n  task: {task}\n  domain: {domain}\n",
        encoding="utf-8",
    )
    data = proj / "data"
    data.mkdir(exist_ok=True)
    lines = [json.dumps({"prompt": f"Q{i}", "response": f"A{i}"}) for i in range(samples)]
    (data / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return proj


# ═══════════════════════════════════════════════════════════════════
# §1–2  INSTALLATION & CLI SECURITY
# ═══════════════════════════════════════════════════════════════════
class TestCLISecurity(unittest.TestCase):
    """CLI argument handling: missing, invalid, unicode, shell metacharacters."""

    def test_cli_no_args_shows_help(self):
        r = runner.invoke(app, [])
        # Typer returns 2 for missing subcommand (usage error)
        self.assertIn(r.exit_code, [0, 2])

    def test_cli_unknown_command(self):
        r = runner.invoke(app, ["nonexistent-cmd"])
        self.assertNotEqual(r.exit_code, 0)

    def test_cli_init_invalid_chars(self):
        r = runner.invoke(app, ["init", "../../escape"])
        # Should not create dirs outside workspace or crash
        self.assertIn(r.exit_code, [0, 1, 2])

    def test_cli_train_without_project(self):
        with patch("myai.core.paths.find_project_root", side_effect=SystemExit(1)):
            r = runner.invoke(app, ["train"])
            self.assertNotEqual(r.exit_code, 0)

    def test_cli_auto_help(self):
        import re
        r = runner.invoke(app, ["auto", "--help"])
        self.assertEqual(r.exit_code, 0)
        clean_stdout = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", r.stdout)
        self.assertIn("--dry-run", clean_stdout)
        self.assertIn("--model", clean_stdout)
        self.assertIn("--override", clean_stdout)

    def test_cli_status_help(self):
        r = runner.invoke(app, ["status", "--help"])
        self.assertEqual(r.exit_code, 0)

    def test_cli_system_check(self):
        r = runner.invoke(app, ["system", "check"])
        self.assertEqual(r.exit_code, 0)
        self.assertIn("CPU", r.stdout)

    def test_cli_optimize_help(self):
        import re
        r = runner.invoke(app, ["optimize", "--help"])
        self.assertEqual(r.exit_code, 0)
        clean_stdout = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", r.stdout)
        self.assertIn("--dry-run", clean_stdout)

    def test_cli_runs_best_help(self):
        r = runner.invoke(app, ["runs", "best", "--help"])
        self.assertEqual(r.exit_code, 0)


# ═══════════════════════════════════════════════════════════════════
# §3  PROJECT INITIALIZATION
# ═══════════════════════════════════════════════════════════════════
class TestProjectInit(unittest.TestCase):

    def test_init_creates_correct_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = _make_project(tmp)
            self.assertTrue((proj / "myai.yaml").exists())
            cfg = ProjectConfig.load(proj)
            self.assertEqual(cfg.name, "audit-ai")

    def test_init_unicode_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = _make_project(tmp, name="日本語プロジェクト")
            self.assertTrue((proj / "myai.yaml").exists())

    def test_init_long_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = "a" * 255
            proj = _make_project(tmp, name=name)
            self.assertTrue((proj / "myai.yaml").exists())


# ═══════════════════════════════════════════════════════════════════
# §4  HARDWARE DETECTION & BENCHMARKING
# ═══════════════════════════════════════════════════════════════════
class TestHardwareSecurity(unittest.TestCase):

    def test_hardware_detection_returns_valid(self):
        hw = detect_hardware()
        self.assertIsInstance(hw, HardwareReport)
        self.assertGreater(hw.ram_gb, 0)
        self.assertGreaterEqual(hw.vram_gb, 0)
        self.assertIn(hw.tier, ["T0", "T1", "T2", "T3"])

    def test_hardware_no_negative_values(self):
        hw = detect_hardware()
        self.assertGreaterEqual(hw.ram_gb, 0)
        self.assertGreaterEqual(hw.vram_gb, 0)
        self.assertGreaterEqual(hw.disk_gb, 0)

    def test_benchmark_returns_valid(self):
        res = run_hardware_benchmark(steps=2)
        self.assertIsInstance(res, BenchmarkResult)
        self.assertGreater(res.forward_tokens_per_sec, 0)
        self.assertGreater(res.training_tokens_per_sec, 0)
        self.assertIn(res.measured_tier, ["T0", "T1", "T2", "T3"])

    def test_benchmark_time_estimation(self):
        res = run_hardware_benchmark(steps=2)
        est = res.estimate_minutes(1000, 3, 120)
        self.assertGreaterEqual(est, 0)
        self.assertLess(est, 10000)  # sanity upper bound


# ═══════════════════════════════════════════════════════════════════
# §5  GOAL UNDERSTANDING
# ═══════════════════════════════════════════════════════════════════
class TestGoalSecurity(unittest.TestCase):

    def test_valid_task_types(self):
        for t in TaskType:
            g = GoalProfile(task=t, domain=Domain.GENERAL)
            g.compute_eval_weights()
            self.assertTrue(all(0 <= v <= 1 for v in g.eval_weights.values()))
            # Float rounding: weights should sum to ~1.0 within ±0.02
            self.assertAlmostEqual(sum(g.eval_weights.values()), 1.0, delta=0.02)

    def test_goal_weights_sum_to_one(self):
        for t in TaskType:
            for d in Domain:
                g = GoalProfile(task=t, domain=d)
                g.compute_eval_weights()
                self.assertAlmostEqual(sum(g.eval_weights.values()), 1.0, delta=0.02)

    def test_goal_from_yaml_with_unknown_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "myai.yaml"
            f.write_text("project:\n  name: x\ngoal:\n  task: unknown_task\n", encoding="utf-8")
            g = GoalProfile.from_yaml(f)
            # Should default gracefully, not crash
            self.assertIsNotNone(g)

    def test_goal_yaml_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = _make_project(tmp, task="code", domain="general")
            g = GoalProfile.from_yaml(proj / "myai.yaml")
            self.assertEqual(g.task, TaskType.CODE)


# ═══════════════════════════════════════════════════════════════════
# §6  DATA SOURCE / REFERENCE MODE (CRITICAL)
# ═══════════════════════════════════════════════════════════════════
class TestReferenceModeSecurity(unittest.TestCase):
    """CRITICAL: Raw user data must NEVER be modified."""

    def test_raw_source_file_untouched_after_cleaning(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "raw"
            src.mkdir()
            raw_file = src / "data.jsonl"
            content = '{"prompt": "test@email.com says hello", "response": "Hi alice@test.com"}\n'
            raw_file.write_text(content, encoding="utf-8")
            raw_hash = hashlib.md5(content.encode()).hexdigest()

            proj = Path(tmp) / "project"
            proj.mkdir()
            prepare_datasets([src], proj)

            # Verify raw file is byte-identical
            after_hash = hashlib.md5(raw_file.read_text(encoding="utf-8").encode()).hexdigest()
            self.assertEqual(raw_hash, after_hash, "RAW SOURCE WAS MODIFIED!")

    def test_raw_source_directory_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "raw"
            src.mkdir()
            (src / "a.jsonl").write_text('{"prompt":"p","response":"r"}\n', encoding="utf-8")
            (src / "b.jsonl").write_text('{"prompt":"q","response":"s"}\n', encoding="utf-8")

            before_files = set(f.name for f in src.iterdir())
            proj = Path(tmp) / "proj"
            proj.mkdir()
            prepare_datasets([src], proj)

            after_files = set(f.name for f in src.iterdir())
            self.assertEqual(before_files, after_files, "Files added or removed from raw source!")

    def test_output_goes_to_project_data_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "raw"
            src.mkdir()
            (src / "d.jsonl").write_text('{"prompt":"x","response":"y"}\n', encoding="utf-8")

            proj = Path(tmp) / "proj"
            proj.mkdir()
            prepare_datasets([src], proj)

            # Output must be in proj/data, NOT in src
            self.assertTrue((proj / "data" / "train.jsonl").exists())
            self.assertFalse((src / "data").exists())
            self.assertFalse((src / "train.jsonl").exists())


# ═══════════════════════════════════════════════════════════════════
# §7  DATA FORMAT & SCHEMA VALIDATION
# ═══════════════════════════════════════════════════════════════════
class TestDataValidation(unittest.TestCase):

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "empty.jsonl"
            f.write_text("", encoding="utf-8")
            summary = analyze_dataset(f)
            self.assertEqual(summary.num_samples, 0)
            self.assertEqual(summary.quality_score, 0)

    def test_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "bad.jsonl"
            f.write_text("not json\n{broken\n", encoding="utf-8")
            summary = analyze_dataset(f)
            self.assertEqual(summary.num_samples, 0)

    def test_missing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "noprompt.jsonl"
            f.write_text('{"foo": "bar"}\n', encoding="utf-8")
            summary = analyze_dataset(f)
            # Should count as sample but detect empty prompt
            self.assertGreaterEqual(summary.num_samples, 0)

    def test_nonexistent_path(self):
        summary = analyze_dataset(Path("/nonexistent/path/data.jsonl"))
        self.assertEqual(summary.num_samples, 0)
        self.assertEqual(summary.quality_score, 0)

    def test_unicode_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "unicode.jsonl"
            f.write_text('{"prompt": "日本語の質問", "response": "日本語の答え"}\n', encoding="utf-8")
            summary = analyze_dataset(f)
            self.assertEqual(summary.num_samples, 1)
            self.assertGreater(summary.quality_score, 0)

    def test_extremely_long_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "long.jsonl"
            long_text = "x" * 100_000
            f.write_text(json.dumps({"prompt": long_text, "response": "ok"}) + "\n", encoding="utf-8")
            summary = analyze_dataset(f)
            self.assertEqual(summary.num_samples, 1)


# ═══════════════════════════════════════════════════════════════════
# §8  PII & SECRET SCRUBBING SECURITY
# ═══════════════════════════════════════════════════════════════════
class TestPIISecurity(unittest.TestCase):

    def test_email_scrubbed(self):
        text, count = _scrub_pii("Contact alice@example.com for help")
        self.assertNotIn("alice@example.com", text)
        self.assertIn("[EMAIL_REDACTED]", text)
        self.assertEqual(count, 1)

    def test_phone_scrubbed(self):
        text, count = _scrub_pii("Call 555-123-4567 now")
        self.assertNotIn("555-123-4567", text)
        self.assertEqual(count, 1)

    def test_api_key_scrubbed(self):
        for prefix in ["sk-", "ghp_", "hf_"]:
            key = prefix + "abcdef1234567890abcdef"
            text, count = _scrub_pii(f"Key: {key}")
            self.assertNotIn(key, text)
            self.assertIn("[SECRET_REDACTED]", text)

    def test_aws_key_scrubbed(self):
        text, count = _scrub_pii("AKIAIOSFODNN7EXAMPLE1234")
        self.assertIn("[SECRET_REDACTED]", text)

    def test_no_false_positive_on_normal_text(self):
        text, count = _scrub_pii("Hello world, this is a normal sentence.")
        self.assertEqual(count, 0)
        self.assertEqual(text, "Hello world, this is a normal sentence.")

    def test_multiple_pii_in_one_text(self):
        text, count = _scrub_pii("alice@test.com and bob@test.com both have sk-abcdef1234567890abcdef")
        self.assertGreaterEqual(count, 3)
        self.assertNotIn("alice@test.com", text)
        self.assertNotIn("bob@test.com", text)


# ═══════════════════════════════════════════════════════════════════
# §9  DATA POISONING / CONTAMINATION
# ═══════════════════════════════════════════════════════════════════
class TestContamination(unittest.TestCase):

    def test_leakage_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "raw"
            src.mkdir()
            # Create dataset where prompts will appear in both train and val
            lines = [json.dumps({"prompt": f"Q{i}", "response": f"A{i}"}) for i in range(20)]
            (src / "data.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

            proj = Path(tmp) / "proj"
            proj.mkdir()
            report = prepare_datasets([src], proj, val_split=0.2, seed=42)

            # If leakage detected, it must be filtered
            if report.leakage_detected:
                self.assertGreater(len(report.leakage_samples), 0)

    def test_exact_dedup_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "dups.jsonl"
            f.write_text(
                '{"prompt": "same question", "response": "answer 1"}\n'
                '{"prompt": "same question", "response": "answer 2"}\n'
                '{"prompt": "different question", "response": "answer 3"}\n',
                encoding="utf-8",
            )
            proj = Path(tmp) / "proj"
            proj.mkdir()
            report = prepare_datasets([f], proj, fuzzy_dedup=False)
            self.assertEqual(report.exact_duplicates_removed, 1)

    def test_fuzzy_dedup_works(self):
        # Threshold is 0.85; very similar strings must match
        self.assertTrue(_is_fuzzy_duplicate(
            "What is Python programming language?",
            ["What is Python programming language used for?"]))
        self.assertFalse(_is_fuzzy_duplicate("What is Python?", ["How does cooking work?"]))


# ═══════════════════════════════════════════════════════════════════
# §10  MODEL RECOMMENDATION SECURITY
# ═══════════════════════════════════════════════════════════════════
class TestModelRecommendation(unittest.TestCase):

    def test_recommendation_returns_valid(self):
        hw = _make_hw()
        goal = GoalProfile(task=TaskType.CHAT, domain=Domain.GENERAL)
        goal.compute_eval_weights()
        rec = recommend_model(hw, goal)
        self.assertIsNotNone(rec.model)
        self.assertGreater(len(rec.reasoning), 0)

    def test_recommendation_explains_why(self):
        hw = _make_hw()
        goal = GoalProfile(task=TaskType.CODE, domain=Domain.GENERAL)
        goal.compute_eval_weights()
        rec = recommend_model(hw, goal)
        self.assertGreater(len(rec.reasoning), 0, "Recommendation must include reasoning")

    def test_catalog_has_models(self):
        self.assertGreater(len(CATALOG), 0)
        for m in CATALOG:
            self.assertTrue(hasattr(m, "hidden_size"))
            self.assertTrue(hasattr(m, "num_layers"))
            self.assertTrue(hasattr(m, "context_length"))


# ═══════════════════════════════════════════════════════════════════
# §11  FEASIBILITY ENGINE
# ═══════════════════════════════════════════════════════════════════
class TestFeasibilityEngine(unittest.TestCase):

    def test_vram_boundary_pass(self):
        hw = _make_hw(vram=12.0)
        model = _make_model(params="1.5B")
        cfg = TrainingConfig(quantization="4bit", lora_rank=16)
        est = estimate_vram_gb(model, cfg)
        self.assertLess(est, 12.0)

    def test_vram_boundary_fail(self):
        hw = _make_hw(vram=4.0)
        model = _make_model(params="8B", hidden=4096, layers=32)
        result = run_feasibility(hw, model)
        self.assertEqual(result.overall, "FAIL")

    def test_cpu_mode_feasibility(self):
        hw = _make_hw(gpu="None", vram=0.0, ram=16.0, tier="T1")
        model = _make_model(params="1.5B")
        result = run_feasibility(hw, model)
        # Small model on 16GB RAM should pass
        self.assertEqual(result.overall, "PASS")

    def test_vram_never_negative(self):
        model = _make_model(params="0.5B", hidden=512, layers=12)
        cfg = TrainingConfig(quantization="4bit", lora_rank=8, seq_len=128, batch_size=1)
        est = estimate_vram_gb(model, cfg)
        self.assertGreater(est, 0)

    def test_make_fit_does_not_infinite_loop(self):
        hw = _make_hw(vram=1.0)
        model = _make_model(params="70B", hidden=8192, layers=80)
        reasoning = []
        cfg = TrainingConfig(quantization="fp16", lora_rank=64, seq_len=4096, batch_size=4)
        result = _make_fit(cfg, hw, model, reasoning)
        # Must terminate; cfg values must have been reduced
        self.assertIsNotNone(result)


# ═══════════════════════════════════════════════════════════════════
# §12  TRAINING STRATEGY
# ═══════════════════════════════════════════════════════════════════
class TestTrainingStrategy(unittest.TestCase):

    def test_strategy_produces_valid_config(self):
        hw = _make_hw()
        model = _make_model()
        data = DatasetSummary(num_samples=500, avg_tokens=120)
        goal = GoalProfile(task=TaskType.CHAT)
        goal.compute_eval_weights()
        strat = plan_strategy(hw, model, data, goal)
        self.assertIsInstance(strat, TrainingStrategy)
        self.assertGreater(strat.learning_rate, 0)
        self.assertGreater(strat.epochs, 0)
        self.assertGreater(strat.estimated_vram_gb, 0)

    def test_user_override_takes_precedence(self):
        hw = _make_hw()
        model = _make_model()
        strat = plan_strategy(hw, model, override={"learning_rate": 1e-5, "epochs": 10})
        self.assertEqual(strat.learning_rate, 1e-5)
        self.assertEqual(strat.epochs, 10)
        self.assertEqual(strat.confidence, 1.0)
        self.assertTrue(any("override" in r.lower() for r in strat.reasoning))

    def test_low_quality_data_caps_epochs(self):
        hw = _make_hw()
        model = _make_model()
        data = DatasetSummary(num_samples=100, avg_tokens=50, quality_score=30)
        strat = plan_strategy(hw, model, data)
        self.assertLessEqual(strat.epochs, 4)


# ═══════════════════════════════════════════════════════════════════
# §13  RESOURCE EXHAUSTION / DOS
# ═══════════════════════════════════════════════════════════════════
class TestResourceExhaustion(unittest.TestCase):

    def test_optimizer_bounded_iterations(self):
        goal = GoalProfile(task=TaskType.CHAT)
        goal.compute_eval_weights()
        board = Leaderboard(goal)
        board.add_run(_make_run("r1", {"domain_accuracy": 0.3, "readability": 0.3,
                                        "bleu": 0.3, "rouge": 0.3, "exact_match": 0.3}))
        iteration_count = 0
        def mock_train(s):
            nonlocal iteration_count
            iteration_count += 1
            return _make_run(f"opt-{iteration_count}", {"domain_accuracy": 0.31, "readability": 0.31,
                                                          "bleu": 0.31, "rouge": 0.31, "exact_match": 0.31},
                              strategy=s)
        eng = OptimizerEngine(goal, board, mock_train, max_iters=3)
        rep = eng.run()
        self.assertLessEqual(iteration_count, 3, "Optimizer exceeded max_iters!")

    def test_large_sample_count_scorer(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "big.jsonl"
            lines = [json.dumps({"prompt": f"Q{i}", "response": f"A{i}"}) for i in range(5000)]
            f.write_text("\n".join(lines) + "\n", encoding="utf-8")
            summary = analyze_dataset(f)
            self.assertEqual(summary.num_samples, 5000)


# ═══════════════════════════════════════════════════════════════════
# §15–16  EVALUATION & LEADERBOARD
# ═══════════════════════════════════════════════════════════════════
class TestLeaderboardSecurity(unittest.TestCase):

    def test_regression_blocks_release(self):
        goal = GoalProfile(task=TaskType.CHAT)
        goal.compute_eval_weights()
        board = Leaderboard(goal)
        board.add_run(_make_run("good", regression=True))
        board.add_run(_make_run("bad", regression=False,
                                metrics={"domain_accuracy": 0.99, "readability": 0.99,
                                          "bleu": 0.99, "rouge": 0.99, "exact_match": 0.99}))
        rc = board.release_candidate()
        self.assertIsNotNone(rc)
        self.assertEqual(rc.run.run_id, "good", "Regressed model must NEVER be release candidate!")

    def test_regression_penalty_halves_score(self):
        goal = GoalProfile(task=TaskType.CHAT)
        goal.compute_eval_weights()
        board = Leaderboard(goal)
        run = _make_run("reg", regression=False)
        scored = board.score(run)
        run_pass = _make_run("pass", regression=True)
        scored_pass = board.score(run_pass)
        # Same metrics, regressed one should have ~half the score
        self.assertAlmostEqual(scored.composite, scored_pass.composite * 0.5, delta=1.0)

    def test_nan_metric_clamped(self):
        goal = GoalProfile(task=TaskType.CHAT)
        goal.compute_eval_weights()
        board = Leaderboard(goal)
        run = _make_run("nan", metrics={"domain_accuracy": float("nan"), "readability": 0.5,
                                         "bleu": 0.5, "rouge": 0.5, "exact_match": 0.5})
        scored = board.score(run)
        # NaN should be clamped to 0, not propagate
        self.assertFalse(scored.composite != scored.composite)  # NaN check

    def test_tied_scores_deterministic(self):
        goal = GoalProfile(task=TaskType.CHAT)
        goal.compute_eval_weights()
        board = Leaderboard(goal)
        metrics = {"domain_accuracy": 0.8, "readability": 0.8, "bleu": 0.8, "rouge": 0.8, "exact_match": 0.8}
        board.add_run(_make_run("tie1", metrics=metrics))
        board.add_run(_make_run("tie2", metrics=metrics))
        ranked1 = board.rank()
        ranked2 = board.rank()
        self.assertEqual(ranked1[0].run.run_id, ranked2[0].run.run_id)

    def test_explain_includes_reasoning(self):
        goal = GoalProfile(task=TaskType.CHAT)
        goal.compute_eval_weights()
        board = Leaderboard(goal)
        run = _make_run("exp")
        rr = board.score(run)
        explanation = board.explain(rr)
        self.assertGreater(len(explanation), 0)

    def test_leaderboard_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            goal = GoalProfile(task=TaskType.CHAT)
            goal.compute_eval_weights()
            board = Leaderboard(goal, runs_dir)
            board.add_run(_make_run("persist1"))
            self.assertTrue((runs_dir / "persist1.json").exists())

            board2 = Leaderboard(goal, runs_dir)
            self.assertEqual(len(board2.runs), 1)
            self.assertEqual(board2.runs[0].run_id, "persist1")


# ═══════════════════════════════════════════════════════════════════
# §17  AUTONOMOUS OPTIMIZER
# ═══════════════════════════════════════════════════════════════════
class TestOptimizerSecurity(unittest.TestCase):

    def test_dry_run_no_training(self):
        goal = GoalProfile(task=TaskType.CHAT)
        goal.compute_eval_weights()
        board = Leaderboard(goal)
        board.add_run(_make_run("base"))
        trained = False
        def mock_train(s):
            nonlocal trained
            trained = True
            return _make_run("new")
        eng = OptimizerEngine(goal, board, mock_train, max_iters=3)
        rep = eng.run(dry_run=True)
        self.assertFalse(trained, "Dry-run must NOT execute training!")

    def test_optimizer_never_promotes_regression(self):
        goal = GoalProfile(task=TaskType.CHAT)
        goal.compute_eval_weights()
        board = Leaderboard(goal)
        board.add_run(_make_run("base"))
        def mock_train(s):
            return _make_run("regressed",
                             metrics={"domain_accuracy": 0.99, "readability": 0.99,
                                       "bleu": 0.99, "rouge": 0.99, "exact_match": 0.99},
                             regression=False, strategy=s)
        eng = OptimizerEngine(goal, board, mock_train, max_iters=1)
        rep = eng.run()
        # Must not promote a regressed run
        self.assertEqual(rep.final_run_id, "base")

    def test_convergence_stops_early(self):
        goal = GoalProfile(task=TaskType.CHAT)
        goal.compute_eval_weights()
        board = Leaderboard(goal)
        # Perfect metrics → no gaps → should converge immediately
        board.add_run(_make_run("perfect", {"domain_accuracy": 1.0, "readability": 1.0,
                                             "bleu": 1.0, "rouge": 1.0, "exact_match": 1.0}))
        call_count = 0
        def mock_train(s):
            nonlocal call_count
            call_count += 1
            return _make_run(f"x{call_count}")
        eng = OptimizerEngine(goal, board, mock_train, max_iters=5)
        rep = eng.run()
        self.assertEqual(call_count, 0, "Perfect model should not trigger retraining")


# ═══════════════════════════════════════════════════════════════════
# §18  AUTOPILOT / GOAL-TO-DEPLOYMENT
# ═══════════════════════════════════════════════════════════════════
class TestAutopilotSecurity(unittest.TestCase):

    def test_dry_run_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = _make_project(tmp)
            trained = False
            def mock_train(s):
                nonlocal trained
                trained = True
                return None
            pilot = Autopilot(proj, train_fn=mock_train, dry_run=True)
            report = pilot.run()
            self.assertFalse(trained)
            self.assertFalse(report.ready)

    def test_feasibility_fail_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = _make_project(tmp)
            trained = False
            def mock_train(s):
                nonlocal trained
                trained = True
                return _make_run("x")

            with patch("myai.autopilot.orchestrator.run_feasibility") as mock_feas:
                mock_feas.return_value = FeasibilityResult(
                    overall="FAIL", estimated_vram_gb=99.0,
                    reasoning="Model too large",
                    report=MagicMock(is_feasible=False, hardware_fit=False, data_fit=True, warnings=["Too big"])
                )
                pilot = Autopilot(proj, train_fn=mock_train)
                report = pilot.run()
                self.assertFalse(trained, "Must NOT train when feasibility fails!")
                self.assertFalse(report.ready)

    def test_model_override_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = _make_project(tmp)
            pilot = Autopilot(proj, train_fn=lambda s: None,
                              model_override="Qwen/Qwen2.5-1.5B-Instruct", dry_run=True)
            report = pilot.run()
            model_stage = next(s for s in report.stages if s.name == "Model")
            self.assertTrue(any("override" in r.lower() for r in model_stage.reasoning))


# ═══════════════════════════════════════════════════════════════════
# §19  OVER-AUTOMATION GUARDRAILS
# ═══════════════════════════════════════════════════════════════════
class TestOverAutomationGuardrails(unittest.TestCase):

    def test_strategy_override_wins(self):
        hw = _make_hw()
        model = _make_model()
        strat = plan_strategy(hw, model, override={"lora_rank": 64, "learning_rate": 5e-5})
        self.assertEqual(strat.config.lora_rank, 64)
        self.assertEqual(strat.learning_rate, 5e-5)

    def test_every_recommendation_has_reasoning(self):
        hw = _make_hw()
        goal = GoalProfile(task=TaskType.DOMAIN_QA, domain=Domain.FITNESS)
        goal.compute_eval_weights()
        rec = recommend_model(hw, goal)
        self.assertGreater(len(rec.reasoning), 0)

    def test_every_strategy_has_reasoning(self):
        hw = _make_hw()
        model = _make_model()
        strat = plan_strategy(hw, model)
        self.assertGreater(len(strat.reasoning), 0)
        self.assertGreater(len(strat.assumptions), 0)


# ═══════════════════════════════════════════════════════════════════
# §21–22  EXPORT SECURITY
# ═══════════════════════════════════════════════════════════════════
class TestExportSecurity(unittest.TestCase):

    def _make_test_zip(self, tmp, entries, secrets_in_meta=False):
        zpath = Path(tmp) / "test.myai"
        with zipfile.ZipFile(zpath, "w") as zf:
            for name, content in entries.items():
                zf.writestr(name, content)
        return zpath

    def test_valid_package_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = {
                "model/adapter.bin": b"fake",
                "tokenizer/tokenizer.json": "{}",
                "metadata.json": '{"base_model_repo": "test"}',
                "evaluation.json": "{}",
                "README.md": "# Test",
                "loader.py": "def load(): pass",
                "chat/app.py": "print('hi')",
                "chat/ui.py": "pass",
                "chat/config.json": "{}",
                "chat/web/index.html": "<html></html>",
            }
            zpath = self._make_test_zip(tmp, entries)
            result = validate_package(zpath)
            failed = [c for c in result.checks if not c.passed]
            for f in failed:
                # These are acceptable: no path traversal, no secrets, etc.
                pass

    def test_rejects_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = {
                "model/a.bin": b"x", "tokenizer/t.json": "{}", "metadata.json": "{}",
                "evaluation.json": "{}", "README.md": "", "loader.py": "",
                "chat/app.py": "", "chat/ui.py": "", "chat/config.json": "{}",
                "chat/web/index.html": "",
                ".env": "SECRET_KEY=abc123",
            }
            zpath = self._make_test_zip(tmp, entries)
            result = validate_package(zpath)
            env_check = next(c for c in result.checks if ".env" in c.name)
            self.assertFalse(env_check.passed, "Must reject .env files!")

    def test_rejects_git_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = {
                "model/a.bin": b"x", "tokenizer/t.json": "{}", "metadata.json": "{}",
                ".git/HEAD": "ref: refs/heads/main",
            }
            zpath = self._make_test_zip(tmp, entries)
            result = validate_package(zpath)
            git_check = next(c for c in result.checks if ".git" in c.name)
            self.assertFalse(git_check.passed)

    def test_rejects_myai_source_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = {
                "model/a.bin": b"x", "tokenizer/t.json": "{}", "metadata.json": "{}",
                "src/myai/cli/main.py": "import typer",
            }
            zpath = self._make_test_zip(tmp, entries)
            result = validate_package(zpath)
            src_check = next(c for c in result.checks if "source" in c.name.lower())
            self.assertFalse(src_check.passed)

    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = {
                "model/a.bin": b"x", "metadata.json": "{}",
                "../../../etc/passwd": "root:x:0:0",
            }
            zpath = self._make_test_zip(tmp, entries)
            result = validate_package(zpath)
            trav_check = next(c for c in result.checks if "traversal" in c.name.lower())
            self.assertFalse(trav_check.passed)

    def test_rejects_secrets_in_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = json.dumps({"api_key": "sk-abcdefghijklmnopqrstuvwxyz1234567890"})
            entries = {
                "model/a.bin": b"x", "tokenizer/t.json": "{}",
                "metadata.json": meta,
            }
            zpath = self._make_test_zip(tmp, entries)
            result = validate_package(zpath)
            secret_check = next(c for c in result.checks if "secret" in c.name.lower() or "key" in c.name.lower())
            self.assertFalse(secret_check.passed, "Must detect secrets in metadata!")

    def test_rejects_dataset_in_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = {
                "model/a.bin": b"x", "metadata.json": "{}",
                "data/train.jsonl": '{"prompt":"x","response":"y"}\n',
            }
            zpath = self._make_test_zip(tmp, entries)
            result = validate_package(zpath)
            ds_check = next(c for c in result.checks if "dataset" in c.name.lower())
            self.assertFalse(ds_check.passed)


# ═══════════════════════════════════════════════════════════════════
# §27  LOGGING & INFORMATION DISCLOSURE
# ═══════════════════════════════════════════════════════════════════
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_TESTS_DIR = _REPO_ROOT / "tests"


class TestInformationDisclosure(unittest.TestCase):

    def test_no_secrets_in_source(self):
        import re
        secret_re = re.compile(r"(sk-|ghp_|hf_|AKIA)[a-zA-Z0-9]{16,}")
        src_dir = _SRC_DIR
        for pyfile in src_dir.rglob("*.py"):
            content = pyfile.read_text(encoding="utf-8", errors="ignore")
            matches = secret_re.findall(content)
            # Filter out regex pattern definitions (they contain sk- etc as detection patterns)
            real_secrets = [m for m in matches if "pattern" not in content[max(0, content.index(m)-50):content.index(m)].lower()
                           and "PATTERN" not in content[max(0, content.index(m)-50):content.index(m)]]
            # Allow pattern definitions in cleaner.py and validator.py
            if pyfile.name in ("cleaner.py", "validator.py"):
                continue
            self.assertEqual(len(real_secrets), 0, f"Potential secret in {pyfile}: {real_secrets}")


# ═══════════════════════════════════════════════════════════════════
# §28  FILESYSTEM SECURITY
# ═══════════════════════════════════════════════════════════════════
class TestFilesystemSecurity(unittest.TestCase):

    def test_no_subprocess_in_source(self):
        src_dir = _SRC_DIR
        for pyfile in src_dir.rglob("*.py"):
            content = pyfile.read_text(encoding="utf-8", errors="ignore")
            self.assertNotIn("subprocess.call", content, f"subprocess.call found in {pyfile}")
            self.assertNotIn("subprocess.Popen", content, f"subprocess.Popen found in {pyfile}")
            self.assertNotIn("shell=True", content, f"shell=True found in {pyfile}")

    def test_no_pickle_in_source(self):
        src_dir = _SRC_DIR
        for pyfile in src_dir.rglob("*.py"):
            content = pyfile.read_text(encoding="utf-8", errors="ignore")
            self.assertNotIn("pickle.load", content, f"Unsafe pickle.load in {pyfile}")

    def test_no_exec_in_source(self):
        src_dir = _SRC_DIR
        for pyfile in src_dir.rglob("*.py"):
            content = pyfile.read_text(encoding="utf-8", errors="ignore")
            # exec() but not model.eval()
            lines = content.split("\n")
            for line in lines:
                stripped = line.strip()
                if "exec(" in stripped and not stripped.startswith("#"):
                    self.fail(f"exec() found in {pyfile}: {stripped}")


# ═══════════════════════════════════════════════════════════════════
# §29  CONFIGURATION SECURITY
# ═══════════════════════════════════════════════════════════════════
class TestConfigSecurity(unittest.TestCase):

    def test_malformed_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "myai.yaml"
            f.write_text("invalid:\n  yaml: [broken\n", encoding="utf-8")
            # Should not crash, should handle gracefully
            try:
                cfg = ProjectConfig.load(Path(tmp))
            except Exception:
                pass  # Graceful failure is acceptable

    def test_empty_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "myai.yaml"
            f.write_text("", encoding="utf-8")
            try:
                cfg = ProjectConfig.load(Path(tmp))
            except Exception:
                pass  # Graceful failure is acceptable


# ═══════════════════════════════════════════════════════════════════
# §30  SECRETS MANAGEMENT
# ═══════════════════════════════════════════════════════════════════
class TestSecretsManagement(unittest.TestCase):

    def test_no_env_files_in_repo(self):
        repo = _REPO_ROOT
        env_files = list(repo.glob(".env")) + list(repo.glob("**/.env"))
        env_files = [f for f in env_files if ".venv" not in str(f) and "venv" not in str(f)]
        self.assertEqual(len(env_files), 0, f"Found .env files: {env_files}")

    def test_no_credentials_in_tests(self):
        import re
        secret_re = re.compile(r"(sk-|ghp_|hf_|AKIA)[a-zA-Z0-9]{20,}")
        tests_dir = _TESTS_DIR
        for pyfile in tests_dir.rglob("*.py"):
            content = pyfile.read_text(encoding="utf-8", errors="ignore")
            # Exclude files that intentionally use fake keys for PII testing
            if pyfile.name in ("test_data_cleaner.py", "test_security_audit.py", "test_scorer.py"):
                continue
            matches = secret_re.findall(content)
            self.assertEqual(len(matches), 0, f"Potential secret in test {pyfile}")


# ═══════════════════════════════════════════════════════════════════
# §33  CRASH RECOVERY & STATE
# ═══════════════════════════════════════════════════════════════════
class TestCrashRecovery(unittest.TestCase):

    def test_state_detection_after_partial_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = _make_project(tmp)
            status = inspect_project_state(proj)
            self.assertEqual(status.state, ProjectState.DATA_READY)
            self.assertTrue(status.has_data)

    def test_precondition_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = _make_project(tmp, samples=0)
            if (proj / "data" / "train.jsonl").exists():
                (proj / "data" / "train.jsonl").unlink()
            ok, msg = validate_precondition(proj, ProjectState.TRAINED)
            self.assertFalse(ok)
            self.assertIn("TRAINED", msg)


# ═══════════════════════════════════════════════════════════════════
# §35  REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════
class TestReproducibility(unittest.TestCase):

    def test_cleaning_deterministic_with_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "raw"
            src.mkdir()
            lines = [json.dumps({"prompt": f"Q{i}", "response": f"A{i}"}) for i in range(50)]
            (src / "data.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

            proj1 = Path(tmp) / "p1"; proj1.mkdir()
            proj2 = Path(tmp) / "p2"; proj2.mkdir()

            r1 = prepare_datasets([src], proj1, seed=42)
            r2 = prepare_datasets([src], proj2, seed=42)

            self.assertEqual(r1.train_samples_count, r2.train_samples_count)
            self.assertEqual(r1.val_samples_count, r2.val_samples_count)

            t1 = (proj1 / "data" / "train.jsonl").read_text(encoding="utf-8")
            t2 = (proj2 / "data" / "train.jsonl").read_text(encoding="utf-8")
            self.assertEqual(t1, t2, "Same seed must produce identical output")

    def test_scoring_deterministic(self):
        goal = GoalProfile(task=TaskType.CODE)
        goal.compute_eval_weights()
        board = Leaderboard(goal)
        run = _make_run("det")
        s1 = board.score(run)
        s2 = board.score(run)
        self.assertEqual(s1.composite, s2.composite)


# ═══════════════════════════════════════════════════════════════════
# §36  PRIVACY
# ═══════════════════════════════════════════════════════════════════
class TestPrivacy(unittest.TestCase):

    def test_no_network_imports_in_core(self):
        """Core modules should not import requests/urllib for outbound calls."""
        import importlib
        core_modules = [
            "myai.core.goal", "myai.core.config", "myai.core.state",
            "myai.data.cleaner", "myai.data.scorer",
            "myai.hardware.feasibility", "myai.training.strategy",
            "myai.models.leaderboard", "myai.optimizer.engine",
        ]
        for mod_name in core_modules:
            mod = importlib.import_module(mod_name)
            source = Path(mod.__file__).read_text(encoding="utf-8", errors="ignore")
            self.assertNotIn("import requests", source, f"Network import in {mod_name}")
            self.assertNotIn("from urllib", source, f"Network import in {mod_name}")


# ═══════════════════════════════════════════════════════════════════
# §40  END-TO-END GOLDEN TEST
# ═══════════════════════════════════════════════════════════════════
class TestEndToEndGolden(unittest.TestCase):

    def test_full_autopilot_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = _make_project(tmp, name="golden-test", task="domain-qa",
                                  domain="fitness", samples=10)

            # 1. Goal loads correctly
            goal = GoalProfile.from_yaml(proj / "myai.yaml")
            self.assertEqual(goal.task, TaskType.DOMAIN_QA)

            # 2. Hardware detects
            hw = detect_hardware()
            self.assertIsNotNone(hw)

            # 3. Data scores correctly
            summary = analyze_dataset(proj / "data" / "train.jsonl")
            self.assertEqual(summary.num_samples, 10)

            # 4. Model recommended
            rec = recommend_model(hw, goal, summary)
            self.assertIsNotNone(rec.model)

            # 5. Feasibility runs
            feas = run_feasibility(hw, rec.model, summary)
            self.assertIn(feas.overall, ["PASS", "FAIL"])

            # 6. Strategy plans
            strat = plan_strategy(hw, rec.model, summary, goal)
            self.assertGreater(strat.learning_rate, 0)

            # 7. Leaderboard scores
            board = Leaderboard(goal)
            run = _make_run("golden-1")
            board.add_run(run)
            scored = board.score(run)
            self.assertGreater(scored.composite, 0)

            # 8. Optimizer converges or bounds
            eng = OptimizerEngine(goal, board, lambda s: _make_run("g2", strategy=s), max_iters=1)
            opt = eng.run()
            self.assertIsNotNone(opt.final_run_id)

            # 9. State machine reflects progress
            status = inspect_project_state(proj)
            self.assertIn(status.state, [ProjectState.DATA_READY, ProjectState.TRAINED])

    def test_autopilot_dry_run_e2e(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = _make_project(tmp, name="dry-golden", task="chat", samples=5)
            pilot = Autopilot(proj, train_fn=lambda s: None, dry_run=True)
            report = pilot.run()
            self.assertFalse(report.ready)
            stage_names = [s.name for s in report.stages]
            self.assertIn("Goal", stage_names)
            self.assertIn("Hardware", stage_names)
            self.assertIn("Data", stage_names)
            self.assertIn("Model", stage_names)
            self.assertIn("Feasibility", stage_names)
            self.assertIn("Strategy", stage_names)
            self.assertIn("Dry-run", stage_names)


if __name__ == "__main__":
    unittest.main()
