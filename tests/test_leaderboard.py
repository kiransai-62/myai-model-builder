import tempfile
import unittest
from pathlib import Path

from myai.core.goal import GoalProfile, TaskType, Domain
from myai.models.leaderboard import Leaderboard, RunRecord, RankedRun


class TestLeaderboard(unittest.TestCase):

    def setUp(self):
        # Code generation goal: exact_match and domain_accuracy prioritized
        self.code_goal = GoalProfile(
            task=TaskType.CODE,
            domain=Domain.GENERAL,
        )
        self.code_goal.compute_eval_weights()

        # Chat support goal: readability and rouge prioritized
        self.chat_goal = GoalProfile(
            task=TaskType.CHAT,
            domain=Domain.CUSTOMER_SUPPORT,
        )
        self.chat_goal.compute_eval_weights()

        self.run_a = RunRecord(
            run_id="run_001",
            model_name="qwen2.5-coder-7b",
            timestamp="2026-08-15T10:00:00",
            strategy={"lora_rank": 16, "learning_rate": 2e-4, "quantization": "4bit"},
            metrics={
                "exact_match": 0.85,
                "domain_accuracy": 0.80,
                "bleu": 0.70,
                "rouge": 0.60,
                "readability": 0.50,
            },
            regression_passed=True,
            vram_peak_gb=7.2,
            train_minutes=35.0,
        )

        self.run_b = RunRecord(
            run_id="run_002",
            model_name="llama-3-8b-instruct",
            timestamp="2026-08-15T12:00:00",
            strategy={"lora_rank": 8, "learning_rate": 1e-4, "quantization": "4bit"},
            metrics={
                "exact_match": 0.40,
                "domain_accuracy": 0.50,
                "bleu": 0.65,
                "rouge": 0.85,
                "readability": 0.90,
            },
            regression_passed=True,
            vram_peak_gb=6.5,
            train_minutes=25.0,
        )

        self.run_regressed = RunRecord(
            run_id="run_003_regressed",
            model_name="qwen2.5-coder-7b",
            timestamp="2026-08-15T14:00:00",
            strategy={"lora_rank": 32, "learning_rate": 5e-4, "quantization": "4bit"},
            metrics={
                "exact_match": 0.95,
                "domain_accuracy": 0.95,
                "bleu": 0.90,
                "rouge": 0.90,
                "readability": 0.90,
            },
            regression_passed=False,
            vram_peak_gb=9.0,
            train_minutes=45.0,
        )

    def test_goal_relative_scoring(self):
        lb_code = Leaderboard(self.code_goal)
        score_a_code = lb_code.score(self.run_a)
        score_b_code = lb_code.score(self.run_b)

        # For code task, run_a (high exact_match/domain_accuracy) should beat run_b
        self.assertGreater(score_a_code.composite, score_b_code.composite)

        lb_chat = Leaderboard(self.chat_goal)
        score_a_chat = lb_chat.score(self.run_a)
        score_b_chat = lb_chat.score(self.run_b)

        # For chat task, run_b (high readability/rouge) should beat run_a
        self.assertGreater(score_b_chat.composite, score_a_chat.composite)

    def test_regression_penalty(self):
        lb = Leaderboard(self.code_goal)
        ranked = lb.score(self.run_regressed)

        self.assertFalse(ranked.stable)
        # Even with 0.95 metrics, composite should be halved due to regression
        raw_composite = sum(ranked.breakdown.values()) * 100
        self.assertAlmostEqual(ranked.composite, round(raw_composite * 0.5, 1), places=1)

    def test_ranking_and_release_candidate(self):
        lb = Leaderboard(self.code_goal)
        lb.add_run(self.run_a)
        lb.add_run(self.run_b)
        lb.add_run(self.run_regressed)

        ranked = lb.rank()
        self.assertEqual(len(ranked), 3)
        # Top ranked should be run_a (stable + highest code score)
        self.assertEqual(ranked[0].run.run_id, "run_001")

        # Release candidate must be stable run_a
        rc = lb.release_candidate()
        self.assertIsNotNone(rc)
        self.assertEqual(rc.run.run_id, "run_001")

    def test_no_release_candidate_when_all_regressed(self):
        lb = Leaderboard(self.code_goal)
        lb.add_run(self.run_regressed)

        rc = lb.release_candidate()
        self.assertIsNone(rc)

    def test_compare_ab(self):
        lb = Leaderboard(self.code_goal)
        lb.add_run(self.run_a)
        lb.add_run(self.run_b)

        winner, delta = lb.compare("run_001", "run_002")
        self.assertEqual(winner.run.run_id, "run_001")
        self.assertGreater(delta, 0.0)

    def test_explain(self):
        lb = Leaderboard(self.code_goal)
        scored = lb.score(self.run_a)
        explanation = lb.explain(scored)
        self.assertTrue(len(explanation) >= 2)
        self.assertTrue(any("contributed" in line for line in explanation))

        reg_scored = lb.score(self.run_regressed)
        reg_explanation = lb.explain(reg_scored)
        self.assertTrue(any("regression gate FAILED" in line for line in reg_explanation))

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            lb1 = Leaderboard(self.code_goal, runs_dir=runs_dir)
            lb1.add_run(self.run_a)
            lb1.add_run(self.run_b)

            # Re-instantiate from disk
            lb2 = Leaderboard(self.code_goal, runs_dir=runs_dir)
            self.assertEqual(len(lb2.runs), 2)
            ids = [r.run_id for r in lb2.runs]
            self.assertIn("run_001", ids)
            self.assertIn("run_002", ids)
