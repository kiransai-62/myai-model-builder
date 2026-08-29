import tempfile
import unittest
from pathlib import Path
from typer.testing import CliRunner

from myai.cli.main import app
from myai.core.home import ensure_home


class TestAutoPipelineCLI(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.proj_dir = Path(self.tmpdir.name)

        # Initialize project
        (self.proj_dir / "myai.yaml").write_text(
            """
project:
  name: pipeline-test-ai
goal:
  task: code
  domain: general
  context_priority: balanced
  latency_priority: balanced
training:
  method: qlora
  epochs: 2
  learning_rate: 0.0002
""",
            encoding="utf-8",
        )

        data_dir = self.proj_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "train.jsonl").write_text(
            '{"prompt": "def add(a, b):", "response": "return a + b"}\n'
            '{"prompt": "def mul(a, b):", "response": "return a * b"}\n',
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_train_auto_dry_run_cli(self):
        from unittest.mock import patch
        runner = CliRunner()
        with patch("myai.core.paths.find_project_root", return_value=self.proj_dir), \
             patch("myai.core.paths.require_project_root", return_value=self.proj_dir):
            result = runner.invoke(app, ["train", "--auto", "--dry-run"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("MYAI AUTOPILOT", result.stdout)
            self.assertIn("Goal", result.stdout)
            self.assertIn("Hardware", result.stdout)
            self.assertIn("Feasibility", result.stdout)
            self.assertIn("Dry-run", result.stdout)

    def test_auto_command_dry_run_cli(self):
        from unittest.mock import patch
        runner = CliRunner()
        with patch("myai.core.paths.find_project_root", return_value=self.proj_dir), \
             patch("myai.core.paths.require_project_root", return_value=self.proj_dir):
            result = runner.invoke(app, ["auto", "--dry-run"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("MYAI AUTOPILOT", result.stdout)
            self.assertIn("Strategy", result.stdout)

