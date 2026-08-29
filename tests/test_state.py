import tempfile
import unittest
from pathlib import Path
from typer.testing import CliRunner

from myai.core.state import inspect_project_state, validate_precondition, ProjectState
from myai.cli.main import app


class TestProjectState(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.proj_dir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_state_transitions(self):
        # 1. Initialized
        (self.proj_dir / "myai.yaml").write_text("project:\n  name: state-ai\ngoal:\n  task: chat\n", encoding="utf-8")
        status = inspect_project_state(self.proj_dir)
        self.assertEqual(status.state, ProjectState.INITIALIZED)
        self.assertFalse(status.has_data)

        # 2. Data Ready
        data_dir = self.proj_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "train.jsonl").write_text('{"prompt": "hi", "response": "hello"}\n', encoding="utf-8")
        status_data = inspect_project_state(self.proj_dir)
        self.assertEqual(status_data.state, ProjectState.DATA_READY)
        self.assertTrue(status_data.has_data)
        self.assertEqual(status_data.data_samples, 1)

        # 3. Exported
        export_dir = self.proj_dir / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "state-ai.myai").write_text("BINARY_MOCK_PACKAGE", encoding="utf-8")
        status_exp = inspect_project_state(self.proj_dir)
        self.assertTrue(status_exp.has_export)

    def test_validate_precondition(self):
        (self.proj_dir / "myai.yaml").write_text("project:\n  name: state-ai\n", encoding="utf-8")
        ok, msg = validate_precondition(self.proj_dir, ProjectState.TRAINED)
        self.assertFalse(ok)
        self.assertIn("INITIALIZED", msg)
        self.assertIn("TRAINED", msg)

    def test_cli_status(self):
        (self.proj_dir / "myai.yaml").write_text("project:\n  name: state-ai\ngoal:\n  task: code\n", encoding="utf-8")
        data_dir = self.proj_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "train.jsonl").write_text('{"prompt": "def f():", "response": "pass"}\n', encoding="utf-8")

        from unittest.mock import patch
        runner = CliRunner()
        with patch("myai.core.paths.find_project_root", return_value=self.proj_dir), \
             patch("myai.core.paths.require_project_root", return_value=self.proj_dir):
            result = runner.invoke(app, ["status"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("MYAI PROJECT STATUS", result.stdout)
            self.assertIn("DATA_READY", result.stdout)
            self.assertIn("Recommendation", result.stdout)
