import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typer.testing import CliRunner

from myai.core.config import ProjectConfig, EvaluationConfig
from myai.evaluation.datasets import (
    split_pairs,
    write_holdout,
    read_holdout,
    facts_from_text,
    knowledge_cases_from_holdout,
    load_eval_cases,
)
from myai.evaluation.validators import validate_artifacts, Check
from myai.evaluation.metrics import (
    knowledge_case_score,
    task_score,
    quality_score,
    overall_score,
)
from myai.evaluation.regression import regression_score, REGRESSION_PROMPTS
from myai.evaluation.report import EvaluationReport, MetricResult
from myai.evaluation.runner import run_evaluation, _next_eval_id
from myai.training.runs import RunManager
from myai.models.trained_registry import list_trained, register_trained
from myai.cli.main import app

class TestEvaluationSubsystem(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="myai_eval_test_"))
        self.runner = CliRunner()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_evaluation_config(self):
        cfg = ProjectConfig(name="eval-test")
        self.assertEqual(cfg.evaluation.eval_split, 0.1)
        self.assertEqual(cfg.evaluation.knowledge_min, 0.8)
        self.assertEqual(cfg.evaluation.overall_min, 0.85)

        cfg.evaluation.eval_split = 0.2
        cfg.evaluation.overall_min = 0.90
        cfg.save(self.temp_dir)

        loaded = ProjectConfig.load(self.temp_dir)
        self.assertEqual(loaded.evaluation.eval_split, 0.2)
        self.assertEqual(loaded.evaluation.overall_min, 0.90)

    def test_dataset_splitting_and_holdout(self):
        pairs = [{"prompt": f"Q{i}", "response": f"Answer {i}"} for i in range(20)]
        train_p, eval_p = split_pairs(pairs, eval_fraction=0.1, seed=42)
        self.assertEqual(len(eval_p), 2)
        self.assertEqual(len(train_p), 18)

        # Deterministic
        train_p2, eval_p2 = split_pairs(pairs, eval_fraction=0.1, seed=42)
        self.assertEqual(eval_p, eval_p2)

        # Small dataset (<4 items) returns all in train
        small = [{"prompt": "q", "response": "a"}]
        t_small, e_small = split_pairs(small)
        self.assertEqual(len(t_small), 1)
        self.assertEqual(len(e_small), 0)

        # Write & read holdout
        holdout_file = self.temp_dir / "holdout.jsonl"
        write_holdout(holdout_file, eval_p)
        read_back = read_holdout(holdout_file)
        self.assertEqual(len(read_back), 2)
        self.assertEqual(read_back[0]["prompt"], eval_p[0]["prompt"])

    def test_facts_extraction_and_knowledge_cases(self):
        text = "Our refund period is 30 days and costs $19.99. Contact support@myai.org for assistance."
        facts = facts_from_text(text)
        self.assertTrue(any("$19.99" in f for f in facts))
        self.assertTrue(any("30 days" in f for f in facts))
        self.assertTrue(any("support@myai.org" in f for f in facts))

        holdout = [{"prompt": "What is the policy?", "response": text}]
        cases = knowledge_cases_from_holdout(holdout)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["prompt"], "What is the policy?")
        self.assertTrue(len(cases[0]["required_facts"]) > 0)

    def test_metrics_scoring(self):
        # 1. Knowledge scoring
        case = {
            "required_facts": ["30 days", "$19.99"],
            "must_not_claim": ["unlimited", "free"],
        }
        score, found, violations = knowledge_case_score("We offer 30 days refund for $19.99.", case)
        self.assertEqual(score, 1.0)
        self.assertEqual(len(found), 2)
        self.assertEqual(len(violations), 0)

        # With violation
        score_bad, _, violations_bad = knowledge_case_score("We offer 30 days refund for $19.99 with free access.", case)
        self.assertLess(score_bad, 1.0)
        self.assertIn("free", violations_bad)

        # 2. Task scoring (F1)
        t_score = task_score("Click forgot password on login screen", "Click forgot password on login screen")
        self.assertEqual(t_score, 1.0)

        # 3. Quality score (jargon penalty)
        clean_q = quality_score("This is a clear and simple sentence.")
        jargon_q = quality_score("We utilize robust holistic synergy to leverage paradigms.")
        self.assertGreater(clean_q, jargon_q)

        # 4. Overall score
        ov = overall_score(knowledge=0.9, task=0.9, regression=1.0, quality=0.8)
        self.assertAlmostEqual(ov, 0.4*0.9 + 0.3*0.9 + 0.2*1.0 + 0.1*0.8, places=2)

    def test_regression_scoring(self):
        base_fn = lambda p: "Here is a helpful answer to your question."
        ft_fn = lambda p: "Here is a helpful answer to your question."
        score, delta = regression_score(base_fn, ft_fn)
        self.assertEqual(score, 1.0)
        self.assertEqual(delta, 0.0)

        bad_ft_fn = lambda p: ""
        score_bad, delta_bad = regression_score(base_fn, bad_ft_fn)
        self.assertLess(score_bad, 1.0)
        self.assertGreater(delta_bad, 0.0)

    def test_validators(self):
        base_dir = self.temp_dir / "models" / "base" / "test-model"
        base_dir.mkdir(parents=True)
        (base_dir / "config.json").write_text("{}", encoding="utf-8")

        adapter_dir = self.temp_dir / "adapter"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
        (adapter_dir / "adapter_model.bin").write_text("WEIGHTS", encoding="utf-8")

        cfg = ProjectConfig(name="test", model_id="test-model")
        checks = validate_artifacts(self.temp_dir, cfg, adapter_dir, inference_fn=lambda p: "OK")
        self.assertTrue(all(c.passed for c in checks))

    def test_run_evaluation_and_report_artifacts(self):
        manager = RunManager(self.temp_dir)
        cfg = ProjectConfig(name="fittrack", model_id="test-model", dataset_id="ds_001")
        run = manager.create({
            "project": cfg.name,
            "base_model": cfg.model_id,
            "dataset_id": cfg.dataset_id,
        })
        run.write_result("SUCCESS")

        adapter_dir = run.root / "adapter"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
        (adapter_dir / "adapter_model.bin").write_text("WEIGHTS", encoding="utf-8")

        # Base model files
        base_dir = self.temp_dir / "models" / "base" / cfg.model_id
        base_dir.mkdir(parents=True)
        (base_dir / "config.json").write_text("{}", encoding="utf-8")

        # Create holdout
        holdout = [
            {"prompt": "What is the return policy?", "response": "30 days full refund guaranteed."},
            {"prompt": "How do I reset password?", "response": "Click forgot password on login page."},
        ]
        write_holdout(run.root / "evaluation_holdout.jsonl", holdout)

        source_dir = self.temp_dir / "data"
        source_dir.mkdir(parents=True)

        report = run_evaluation(
            home=self.temp_dir,
            root=self.temp_dir,
            cfg=cfg,
            run=run,
            adapter_path=adapter_dir,
            source=source_dir,
        )

        self.assertIsInstance(report, EvaluationReport)
        self.assertTrue(report.eval_id.startswith("eval_"))
        self.assertEqual(report.status, "PASS")

        # Verify evaluation artifacts on disk
        eval_dir = run.root / "evaluation" / report.eval_id
        self.assertTrue(eval_dir.exists())
        self.assertTrue((eval_dir / "config.json").exists())
        self.assertTrue((eval_dir / "results.json").exists())
        self.assertTrue((eval_dir / "report.json").exists())

    def test_cli_evaluate_and_list_info(self):
        import os
        old_env = os.environ.get("MYAI_HOME")
        os.environ["MYAI_HOME"] = str(self.temp_dir)
        try:
            cfg = ProjectConfig(name="fittrack", model_id="test-model", dataset_id="ds_001")
            cfg.save(self.temp_dir)

            manager = RunManager(self.temp_dir)
            run = manager.create({
                "project": cfg.name,
                "base_model": cfg.model_id,
                "dataset_id": cfg.dataset_id,
            })
            run.write_result("SUCCESS")

            adapter_dir = run.root / "adapter"
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
            (adapter_dir / "adapter_model.bin").write_text("WEIGHTS", encoding="utf-8")

            base_dir = self.temp_dir / "models" / "base" / cfg.model_id
            base_dir.mkdir(parents=True)
            (base_dir / "config.json").write_text("{}", encoding="utf-8")

            # Data directory
            data_dir = self.temp_dir / "data"
            data_dir.mkdir(parents=True)
            with open(data_dir / "train.jsonl", "w", encoding="utf-8") as f:
                f.write(json.dumps({"prompt": "Hello", "response": "Hi there!"}) + "\n")

            # Run myai evaluate
            old_cwd = os.getcwd()
            try:
                os.chdir(self.temp_dir)
                res = self.runner.invoke(app, ["evaluate"])
                self.assertEqual(res.exit_code, 0, f"evaluate output: {res.stdout}")
                self.assertIn("MYAI EVALUATION", res.stdout)
                self.assertIn("PASS", res.stdout)

                # Test evaluate list
                res_list = self.runner.invoke(app, ["evaluate", "list"])
                self.assertEqual(res_list.exit_code, 0)
                self.assertIn("EVALUATIONS", res_list.stdout)
                self.assertIn("fittrack", res_list.stdout)

                # Test evaluate info
                eval_ids = [p.name for p in (run.root / "evaluation").glob("eval_*")]
                self.assertTrue(len(eval_ids) > 0)
                res_info = self.runner.invoke(app, ["evaluate", "info", eval_ids[0]])
                self.assertEqual(res_info.exit_code, 0)
                self.assertIn(eval_ids[0], res_info.stdout)

                # Verify model promoted in trained registry
                trained = list_trained(self.temp_dir)
                self.assertEqual(len(trained), 1)
                self.assertEqual(trained[0]["id"], "fittrack")
                self.assertEqual(trained[0]["status"], "READY")
                self.assertIn("evaluation", trained[0])
            finally:
                os.chdir(old_cwd)
        finally:
            if old_env is not None:
                os.environ["MYAI_HOME"] = old_env
            else:
                os.environ.pop("MYAI_HOME", None)

    def test_cli_evaluate_failure_path(self):
        import os
        old_env = os.environ.get("MYAI_HOME")
        os.environ["MYAI_HOME"] = str(self.temp_dir)
        try:
            cfg = ProjectConfig(name="fittrack-fail", model_id="test-model", dataset_id="ds_001")
            cfg.evaluation.overall_min = 1.05
            cfg.save(self.temp_dir)

            manager = RunManager(self.temp_dir)
            run = manager.create({
                "project": cfg.name,
                "base_model": cfg.model_id,
                "dataset_id": cfg.dataset_id,
            })
            run.write_result("SUCCESS")

            adapter_dir = run.root / "adapter"
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
            (adapter_dir / "adapter_model.bin").write_text("WEIGHTS", encoding="utf-8")

            base_dir = self.temp_dir / "models" / "base" / cfg.model_id
            base_dir.mkdir(parents=True)
            (base_dir / "config.json").write_text("{}", encoding="utf-8")

            data_dir = self.temp_dir / "data"
            data_dir.mkdir(parents=True)
            with open(data_dir / "train.jsonl", "w", encoding="utf-8") as f:
                f.write(json.dumps({"prompt": "Hello", "response": "Hi there!"}) + "\n")

            old_cwd = os.getcwd()
            try:
                os.chdir(self.temp_dir)
                res = self.runner.invoke(app, ["evaluate"])
                self.assertEqual(res.exit_code, 0)
                self.assertIn("Status: FAILED", res.stdout)
                self.assertIn("Reason:", res.stdout)
                self.assertIn("Recommended action:", res.stdout)

                # Ensure NOT registered
                trained = list_trained(self.temp_dir)
                self.assertEqual(len([m for m in trained if m.get("id") == "fittrack-fail"]), 0)
            finally:
                os.chdir(old_cwd)
        finally:
            if old_env is not None:
                os.environ["MYAI_HOME"] = old_env
            else:
                os.environ.pop("MYAI_HOME", None)

if __name__ == "__main__":
    unittest.main()

