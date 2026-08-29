import tempfile
import unittest
from pathlib import Path

from myai.data.scorer import analyze_dataset, DatasetSummary


class TestDatasetScorer(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_clean_dataset_scoring(self):
        file_path = self.tmp_path / "clean.jsonl"
        file_path.write_text(
            '{"prompt": "What is Python?", "response": "Python is a high-level programming language."}\n'
            '{"prompt": "Explain list comprehension.", "response": "List comprehension provides a concise syntax to create lists."}\n'
            '{"prompt": "What is a dictionary?", "response": "A dictionary is an associative array of key-value pairs."}\n',
            encoding="utf-8",
        )

        summary = analyze_dataset(file_path)
        self.assertIsInstance(summary, DatasetSummary)
        self.assertEqual(summary.num_samples, 3)
        self.assertEqual(summary.exact_duplicates, 0)
        self.assertEqual(summary.dup_pct, 0.0)
        self.assertGreaterEqual(summary.quality_score, 80)
        self.assertGreater(summary.tokens_approx, 0)

    def test_duplicate_and_pii_detection(self):
        file_path = self.tmp_path / "dirty.jsonl"
        file_path.write_text(
            '{"prompt": "Contact me at alice@test.com", "response": "Sure, emailing alice@test.com"}\n'
            '{"prompt": "Contact me at alice@test.com", "response": "Duplicate prompt query"}\n'
            '{"prompt": "My key is sk-1234567890abcdef1234", "response": "Secret noted"}\n',
            encoding="utf-8",
        )

        summary = analyze_dataset(file_path)
        self.assertEqual(summary.num_samples, 3)
        self.assertEqual(summary.exact_duplicates, 1)
        self.assertGreater(summary.dup_pct, 30.0)
        self.assertGreater(summary.pii_count, 0)
        self.assertTrue(any("duplication" in issue.lower() for issue in summary.issues))
        self.assertTrue(any("pii" in issue.lower() for issue in summary.issues))

    def test_empty_and_missing_source(self):
        missing_path = self.tmp_path / "nonexistent.jsonl"
        summary = analyze_dataset(missing_path)
        self.assertEqual(summary.num_samples, 0)
        self.assertEqual(summary.quality_score, 0)

        empty_path = self.tmp_path / "empty.jsonl"
        empty_path.write_text("", encoding="utf-8")
        summary_empty = analyze_dataset(empty_path)
        self.assertEqual(summary_empty.num_samples, 0)
        self.assertEqual(summary_empty.quality_score, 0)
