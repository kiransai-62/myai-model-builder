import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typer.testing import CliRunner

from myai.data.cleaner import prepare_datasets, _scrub_pii, _is_fuzzy_duplicate, CleaningReport
from myai.core.config import ProjectConfig
from myai.cli.main import app


class TestDatasetCleaner(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="myai_cleaner_test_"))
        self.raw_data_dir = self.temp_dir / "external_raw_data"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.project_dir = self.temp_dir / "project"
        self.project_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pii_and_secret_scrubbing(self):
        sample_text = "Contact support@example.com or call 555-123-4567. Key is sk-abcdef123456789012345."
        scrubbed, count = _scrub_pii(sample_text)
        self.assertEqual(count, 3)
        self.assertNotIn("support@example.com", scrubbed)
        self.assertNotIn("555-123-4567", scrubbed)
        self.assertNotIn("sk-abcdef123456789012345", scrubbed)
        self.assertIn("[EMAIL_REDACTED]", scrubbed)
        self.assertIn("[PHONE_REDACTED]", scrubbed)
        self.assertIn("[SECRET_REDACTED]", scrubbed)

    def test_fuzzy_duplicate_detection(self):
        existing = ["What is the refund policy for annual subscriptions?"]
        similar = "What is the refund policy for annual subscription?"
        different = "How do I upgrade my plan?"

        self.assertTrue(_is_fuzzy_duplicate(similar, existing, threshold=0.85))
        self.assertFalse(_is_fuzzy_duplicate(different, existing, threshold=0.85))

    def test_clean_and_prepare_reference_mode(self):
        # Create raw reference dataset with duplicates, PII, and malformed samples
        raw_file = self.raw_data_dir / "raw_dataset.jsonl"
        raw_content = [
            {"prompt": "How do I contact support?", "response": "Email help@company.com anytime."},
            {"prompt": "How do I contact support?", "response": "Email help@company.com anytime."}, # exact duplicate
            {"prompt": "How do I contact support?", "response": "Duplicate query."}, # exact prompt duplicate
            {"prompt": "How to contact support?", "response": "Fuzzy duplicate query."}, # fuzzy duplicate
            {"prompt": "What is your token?", "response": "Secret is sk-1234567890abcdef12345."}, # secret
            {"prompt": "   ", "response": "empty prompt"}, # malformed
            {"prompt": "Valid question 1", "response": "Valid response 1"},
            {"prompt": "Valid question 2", "response": "Valid response 2"},
            {"prompt": "Valid question 3", "response": "Valid response 3"},
            {"prompt": "Valid question 4", "response": "Valid response 4"},
            {"prompt": "Valid question 5", "response": "Valid response 5"},
        ]
        with open(raw_file, "w", encoding="utf-8") as f:
            for item in raw_content:
                f.write(json.dumps(item) + "\n")

        raw_mtime_before = raw_file.stat().st_mtime

        # Run prepare_datasets
        report = prepare_datasets([raw_file], self.project_dir, val_split=0.2, fuzzy_dedup=True, seed=42)

        # 1. Verify Reference Mode: original file was untouched
        raw_mtime_after = raw_file.stat().st_mtime
        self.assertEqual(raw_mtime_before, raw_mtime_after)

        # 2. Verify Cleaning Stats
        self.assertEqual(report.total_raw_samples, len(raw_content))
        self.assertTrue(report.exact_duplicates_removed >= 1)
        self.assertTrue(report.pii_redactions_made >= 1)
        self.assertTrue(report.empty_malformed_removed >= 1)

        # 3. Verify Output files exist in workspace
        train_file = self.project_dir / "data" / "train.jsonl"
        val_file = self.project_dir / "data" / "validation.jsonl"
        self.assertTrue(train_file.exists())
        self.assertTrue(val_file.exists())

        # Verify no secrets in processed output
        train_text = train_file.read_text(encoding="utf-8")
        self.assertNotIn("help@company.com", train_text)
        self.assertNotIn("sk-1234567890abcdef12345", train_text)

    def test_contamination_leakage_detection(self):
        raw_file = self.raw_data_dir / "samples.json"
        raw_content = [
            {"prompt": f"Unique prompt {i}", "response": f"Response {i}"}
            for i in range(10)
        ]
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(raw_content, f)

        report = prepare_datasets([raw_file], self.project_dir, val_split=0.3, seed=42)
        # Verify train and val sets are non-overlapping
        train_file = self.project_dir / "data" / "train.jsonl"
        val_file = self.project_dir / "data" / "validation.jsonl"

        train_prompts = [json.loads(line)["prompt"] for line in train_file.read_text(encoding="utf-8").splitlines() if line]
        val_prompts = [json.loads(line)["prompt"] for line in val_file.read_text(encoding="utf-8").splitlines() if line]

        overlap = set(train_prompts).intersection(set(val_prompts))
        self.assertEqual(len(overlap), 0, "Train and Validation sets must have zero overlap")
        self.assertFalse(report.leakage_detected)

    def test_cli_data_prepare(self):
        # Set up a mock project
        raw_file = self.project_dir / "data" / "source.jsonl"
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": "Hello", "response": "Hi there!"}) + "\n")

        cfg = ProjectConfig(name="cli-test-project", data_path="data/source.jsonl")
        cfg.save(self.project_dir)

        runner = CliRunner()
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(self.project_dir)
            result = runner.invoke(app, ["data", "prepare", "--val-split", "0.2"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Dataset Intelligence & Cleaning Report", result.stdout)
            self.assertIn("Processed datasets saved", result.stdout)
        finally:
            os.chdir(old_cwd)
