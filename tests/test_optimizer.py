import unittest
from typer.testing import CliRunner

from myai.core.goal import GoalProfile, TaskType, Domain
from myai.models.leaderboard import Leaderboard, RunRecord, RankedRun
from myai.optimizer.engine import OptimizerEngine, Diagnosis, OptimizationStep, OptimizationReport
from myai.cli.main import app


class TestOptimizerEngine(unittest.TestCase):

    def setUp(self):
        # Goal: Code task with high exact_match & domain_accuracy
        self.code_goal = GoalProfile(task=TaskType.CODE, domain=Domain.GENERAL)
        self.code_goal.compute_eval_weights()

        self.initial_run = RunRecord(
            run_id="run-100",
            model_name="qwen2.5-coder-7b",
            timestamp="2026-08-15T10:00:00",
            strategy={"learning_rate": 2e-4, "seq_len": 512, "lora_rank": 8, "epochs": 2},
            metrics={
                "exact_match": 0.50,       # High gap
                "domain_accuracy": 0.60,   # Moderate gap
                "bleu": 0.70,
                "rouge": 0.70,
                "readability": 0.80,
            },
            regression_passed=True,
            vram_peak_gb=6.5,
            train_minutes=15.0,
        )

    def test_diagnose(self):
        board = Leaderboard(self.code_goal)
        board.add_run(self.initial_run)
        ranked = board.score(self.initial_run)

        engine = OptimizerEngine(self.code_goal, board, train_fn=lambda s: self.initial_run)
        diags = engine.diagnose(ranked)

        self.assertTrue(len(diags) >= 1)
        self.assertEqual(diags[0].metric, "exact_match")
        self.assertGreater(diags[0].lost_points, 0.03)

    def test_prescribe(self):
        board = Leaderboard(self.code_goal)
        board.add_run(self.initial_run)
        ranked = board.score(self.initial_run)

        engine = OptimizerEngine(self.code_goal, board, train_fn=lambda s: self.initial_run)
        diags = engine.diagnose(ranked)
        mutation, reasoning = engine.prescribe(diags, self.initial_run.strategy)

        self.assertIn("learning_rate", mutation)
        self.assertEqual(mutation["learning_rate"], 1e-4)
        self.assertTrue(any("exact_match" in r for r in reasoning))

    def test_dry_run(self):
        board = Leaderboard(self.code_goal)
        board.add_run(self.initial_run)

        trained_called = False
        def mock_train(s):
            nonlocal trained_called
            trained_called = True
            return self.initial_run

        engine = OptimizerEngine(self.code_goal, board, train_fn=mock_train, max_iters=2)
        report = engine.run(dry_run=True)

        self.assertFalse(trained_called)
        self.assertEqual(len(report.steps), 2)
        self.assertTrue(any("[dry-run]" in r for r in report.steps[0].reasoning))
        self.assertFalse(report.improved)

    def test_autonomous_promotion_and_rejection(self):
        board = Leaderboard(self.code_goal)
        board.add_run(self.initial_run)

        call_count = 0
        def mock_train(strategy):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Iteration 1: Significant improvement (+10 composite points)
                return RunRecord(
                    run_id="run-101",
                    model_name="qwen2.5-coder-7b",
                    timestamp="2026-08-15T11:00:00",
                    strategy=strategy,
                    metrics={
                        "exact_match": 0.85,
                        "domain_accuracy": 0.80,
                        "bleu": 0.75,
                        "rouge": 0.75,
                        "readability": 0.85,
                    },
                    regression_passed=True,
                )
            else:
                # Iteration 2: Inferior / regressed
                return RunRecord(
                    run_id="run-102",
                    model_name="qwen2.5-coder-7b",
                    timestamp="2026-08-15T12:00:00",
                    strategy=strategy,
                    metrics={
                        "exact_match": 0.86,
                        "domain_accuracy": 0.81,
                        "bleu": 0.75,
                        "rouge": 0.75,
                        "readability": 0.85,
                    },
                    regression_passed=False,  # Unstable
                )

        engine = OptimizerEngine(self.code_goal, board, train_fn=mock_train, min_delta=2.0, max_iters=2)
        report = engine.run()

        self.assertTrue(report.improved)
        self.assertEqual(report.final_run_id, "run-101")
        self.assertEqual(len(report.steps), 2)
        self.assertTrue(report.steps[0].promoted)
        self.assertFalse(report.steps[1].promoted)

    def test_cli_optimize_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["optimize", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--max-iters", result.stdout)
        self.assertIn("--min-delta", result.stdout)
        self.assertIn("--dry-run", result.stdout)
