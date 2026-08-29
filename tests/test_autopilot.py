import tempfile
import unittest
from pathlib import Path
from typer.testing import CliRunner

from myai.autopilot.orchestrator import Autopilot, AutopilotReport, _load_sources
from myai.models.leaderboard import RunRecord
from myai.cli.main import app


class TestAutopilot(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.proj_dir = Path(self.tmpdir.name)

        # Create basic myai.yaml
        (self.proj_dir / "myai.yaml").write_text(
            """
project:
  name: test-pilot-ai
goal:
  task: domain-qa
  domain: fitness
  context_priority: balanced
  latency_priority: balanced
training:
  method: qlora
  epochs: 3
  learning_rate: 0.0002
""",
            encoding="utf-8",
        )

        # Create dummy training data
        data_dir = self.proj_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "train.jsonl").write_text(
            '{"prompt": "How many reps?", "response": "3 sets of 10 reps."}\n'
            '{"prompt": "What is protein intake?", "response": "1.6g to 2.2g per kg."}\n',
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_autopilot_dry_run(self):
        trained = False
        def mock_train(strat):
            nonlocal trained
            trained = True
            return None

        pilot = Autopilot(
            self.proj_dir,
            train_fn=mock_train,
            dry_run=True,
        )
        report = pilot.run()

        self.assertFalse(trained)
        self.assertFalse(report.ready)
        stage_names = [s.name for s in report.stages]
        self.assertIn("Goal", stage_names)
        self.assertIn("Hardware", stage_names)
        self.assertIn("Data", stage_names)
        self.assertIn("Model", stage_names)
        self.assertIn("Feasibility", stage_names)
        self.assertIn("Strategy", stage_names)
        self.assertIn("Dry-run", stage_names)

    def test_autopilot_full_run_with_export(self):
        exported_id = None
        def mock_train(strat):
            return RunRecord(
                run_id="run-pilot-001",
                model_name="qwen2.5-1.5b",
                timestamp="2026-08-15T12:00:00",
                strategy={"learning_rate": 2e-4, "epochs": 2, "lora_rank": 16},
                metrics={
                    "readability": 0.85,
                    "domain_accuracy": 0.88,
                    "exact_match": 0.80,
                    "rouge": 0.80,
                    "bleu": 0.80,
                },
                regression_passed=True,
                train_minutes=5.0,
            )

        def mock_export(run_id):
            nonlocal exported_id
            exported_id = run_id
            return self.proj_dir / "export" / f"{self.proj_dir.name}.myai"

        pilot = Autopilot(
            self.proj_dir,
            train_fn=mock_train,
            export_fn=mock_export,
            export=True,
            dry_run=False,
            max_opt_iters=1,
        )
        report = pilot.run()

        self.assertTrue(report.ready)
        self.assertIsNotNone(report.final_run_id)
        self.assertIsNotNone(report.export_path)
        self.assertEqual(exported_id, report.final_run_id)

    def test_autopilot_overrides(self):
        trained_strat = None
        def mock_train(strat):
            nonlocal trained_strat
            trained_strat = strat
            return RunRecord(
                run_id="run-pilot-override",
                model_name="qwen2.5-1.5b",
                timestamp="2026-08-15T12:00:00",
                strategy={"learning_rate": 1e-4, "epochs": 5, "lora_rank": 32},
                metrics={"domain_accuracy": 0.9, "readability": 0.9},
                regression_passed=True,
            )

        pilot = Autopilot(
            self.proj_dir,
            train_fn=mock_train,
            model_override="Qwen/Qwen2.5-1.5B-Instruct",
            strategy_override={"learning_rate": 1e-4, "lora_rank": 32, "epochs": 5},
            dry_run=True,
        )
        report = pilot.run()

        model_stage = next(s for s in report.stages if s.name == "Model")
        self.assertTrue(any("User override" in r for r in model_stage.reasoning))

        strat_stage = next(s for s in report.stages if s.name == "Strategy")
        self.assertIn("r32", strat_stage.summary)
        self.assertIn("5 ep", strat_stage.summary)

    def test_cli_auto_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["auto", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Goal-to-deployment autonomous build", result.stdout)
        self.assertIn("--dry-run", result.stdout)
        self.assertIn("--export", result.stdout)
        self.assertIn("--model", result.stdout)
        self.assertIn("--override", result.stdout)
