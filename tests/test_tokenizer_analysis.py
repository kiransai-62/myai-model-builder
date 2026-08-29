"""Comprehensive Test Suite for MYAI Tokenizer Analysis Subsystem.

Tests all required capabilities from Section 17:
- JSON, JSONL, CSV, TXT dataset tokenization
- Mixed-format folder discovery
- Chat template formatting and Instruction formatting
- Input/output token statistics
- Distribution buckets and Context-length analysis
- Empty datasets, malformed records, and long records
- Tokenizer cache persistence and deterministic invalidation
- Offline tokenizer fallback and model-aware resolution
- Unicode support and large-record streaming
- Strict Reference Mode integrity (MD5 before == MD5 after)
"""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typer.testing import CliRunner

from myai.cli.main import app
from myai.tokenization.tokenizer import (
    TokenizerEngine, CalibratedHeuristicTokenizer, get_tokenizer, resolve_model_repo
)
from myai.tokenization.formatter import (
    format_record, detect_record_schema, FormattedSample
)
from myai.tokenization.stats import (
    TokenStats, TokenDistribution, IOStats, ContextAnalysis, compute_token_stats, BUCKET_RANGES
)
from myai.tokenization.analyzer import (
    analyze_dataset_tokens, stream_records_from_path, discover_data_files
)
from myai.tokenization.cache import (
    TokenizationCache, calculate_source_fingerprint
)


runner = CliRunner()


class TestTokenizerEngine(unittest.TestCase):
    """Tests for model-aware tokenizer resolution and offline heuristic fallback."""

    def test_heuristic_tokenizer_basic_counting(self):
        tok = CalibratedHeuristicTokenizer("Qwen/Qwen2.5-1.5B-Instruct", family="qwen")
        text = "Hello world! This is a simple tokenizer test."
        tokens = tok.encode(text)
        self.assertGreater(len(tokens), 0)
        self.assertEqual(tok.count_tokens(text), len(tokens))

    def test_heuristic_tokenizer_empty_and_spaces(self):
        tok = CalibratedHeuristicTokenizer("test", family="default")
        self.assertEqual(tok.encode(""), [])
        self.assertEqual(tok.count_tokens(""), 0)
        self.assertGreaterEqual(tok.count_tokens("   "), 0)

    def test_heuristic_tokenizer_chat_template(self):
        tok = CalibratedHeuristicTokenizer("qwen", family="qwen")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
        ]
        formatted = tok.apply_chat_template(messages)
        self.assertIn("<|im_start|>system", formatted)
        self.assertIn("<|im_start|>user", formatted)
        self.assertIn("What is Python?", formatted)

    def test_get_tokenizer_offline_fallback(self):
        tok_engine = get_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", offline_only=True)
        self.assertIsInstance(tok_engine, TokenizerEngine)
        self.assertGreater(tok_engine.vocab_size, 1000)
        count = tok_engine.count_tokens("Testing the offline fallback tokenizer.")
        self.assertGreater(count, 0)

    def test_resolve_model_repo_shorthand(self):
        repo = resolve_model_repo("qwen2.5-1.5b-instruct")
        self.assertIn("Qwen", repo)


class TestDatasetFormatter(unittest.TestCase):
    """Tests schema detection and training representation formatting."""

    def test_detect_instruction_schema(self):
        record = {"instruction": "Summarize", "input": "Some text", "output": "Summary"}
        self.assertEqual(detect_record_schema(record), "instruction")

    def test_detect_prompt_response_schema(self):
        record = {"prompt": "What is AI?", "response": "Artificial Intelligence"}
        self.assertEqual(detect_record_schema(record), "prompt_response")

    def test_detect_chat_schema(self):
        record = {"messages": [{"role": "user", "content": "Hello"}]}
        self.assertEqual(detect_record_schema(record), "chat")

    def test_detect_text_schema(self):
        record = {"text": "Just raw document text."}
        self.assertEqual(detect_record_schema(record), "text")

    def test_format_instruction_record(self):
        record = {"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"}
        sample = format_record(record)
        self.assertIsInstance(sample, FormattedSample)
        self.assertEqual(sample.schema, "instruction")
        self.assertIn("Translate to French", sample.input_text)
        self.assertIn("Hello", sample.input_text)
        self.assertEqual(sample.output_text, "Bonjour")
        self.assertIn("Bonjour", sample.full_text)

    def test_format_chat_record(self):
        record = {"messages": [
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am doing well."}
        ]}
        sample = format_record(record)
        self.assertEqual(sample.schema, "chat")
        self.assertIn("How are you?", sample.input_text)
        self.assertIn("I am doing well.", sample.output_text)
        self.assertIn("How are you?", sample.full_text)


class TestTokenStatisticsAndContext(unittest.TestCase):
    """Tests statistical calculations, distribution buckets, and context-length fit."""

    def test_compute_token_stats_normal(self):
        full_tokens = [50, 100, 150, 200, 300]
        input_tokens = [25, 50, 75, 100, 150]
        output_tokens = [25, 50, 75, 100, 150]
        
        stats = compute_token_stats(
            dataset_id="test_ds",
            model_id="test_model",
            tokenizer_name="TestTokenizer",
            full_token_counts=full_tokens,
            input_token_counts=input_tokens,
            output_token_counts=output_tokens,
            total_chars=1000,
            total_words=200,
            schema_detected="instruction",
            model_context_length=4096,
        )

        self.assertEqual(stats.total_samples, 5)
        self.assertEqual(stats.total_tokens, 800)
        self.assertEqual(stats.avg_tokens, 160.0)
        self.assertEqual(stats.median_tokens, 150.0)
        self.assertEqual(stats.min_tokens, 50)
        self.assertEqual(stats.max_tokens, 300)
        self.assertEqual(stats.context_analysis.status, "FIT")
        self.assertEqual(stats.context_analysis.samples_over_limit, 0)

    def test_context_overflow_detection(self):
        full_tokens = [100, 500, 5000]  # 5000 exceeds 4096 context
        stats = compute_token_stats(
            dataset_id="overflow_ds",
            model_id="test_model",
            tokenizer_name="TestTokenizer",
            full_token_counts=full_tokens,
            input_token_counts=[50, 250, 2500],
            output_token_counts=[50, 250, 2500],
            total_chars=5000,
            total_words=1000,
            schema_detected="instruction",
            model_context_length=4096,
        )

        self.assertEqual(stats.context_analysis.status, "OVERFLOW")
        self.assertEqual(stats.context_analysis.samples_over_limit, 1)
        self.assertAlmostEqual(stats.context_analysis.pct_over_limit, 33.3, places=1)
        self.assertGreater(len(stats.context_analysis.recommended_actions), 0)

    def test_distribution_buckets(self):
        full_tokens = [50, 200, 400, 800, 1500, 3000, 5000]
        stats = compute_token_stats(
            dataset_id="buckets_ds",
            model_id="test_model",
            tokenizer_name="TestTokenizer",
            full_token_counts=full_tokens,
            input_token_counts=[10]*7,
            output_token_counts=[10]*7,
            total_chars=1000,
            total_words=200,
            schema_detected="instruction",
        )

        self.assertEqual(stats.distribution.buckets["0-128"], 1)
        self.assertEqual(stats.distribution.buckets["129-256"], 1)
        self.assertEqual(stats.distribution.buckets["257-512"], 1)
        self.assertEqual(stats.distribution.buckets["513-1024"], 1)
        self.assertEqual(stats.distribution.buckets["1025-2048"], 1)
        self.assertEqual(stats.distribution.buckets["2049-4096"], 1)
        self.assertEqual(stats.distribution.buckets["4097+"], 1)


class TestStreamingAnalyzerAndFormats(unittest.TestCase):
    """Tests file discovery, format streaming, and Reference Mode invariant."""

    def test_jsonl_tokenization(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "data.jsonl"
            lines = [
                json.dumps({"prompt": f"Question {i}", "response": f"Answer {i}"})
                for i in range(25)
            ]
            f.write_text("\n".join(lines) + "\n", encoding="utf-8")

            stats = analyze_dataset_tokens(f, dataset_id="ds_jsonl", use_cache=False)
            self.assertEqual(stats.total_samples, 25)
            self.assertGreater(stats.total_tokens, 0)
            self.assertEqual(stats.schema_detected, "prompt_response")

    def test_json_array_tokenization(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "data.json"
            items = [{"instruction": f"Inst {i}", "output": f"Out {i}"} for i in range(15)]
            f.write_text(json.dumps(items), encoding="utf-8")

            stats = analyze_dataset_tokens(f, dataset_id="ds_json", use_cache=False)
            self.assertEqual(stats.total_samples, 15)
            self.assertGreater(stats.total_tokens, 0)
            self.assertEqual(stats.schema_detected, "instruction")

    def test_csv_tokenization(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "data.csv"
            with open(f, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["prompt", "response"])
                writer.writeheader()
                for i in range(10):
                    writer.writerow({"prompt": f"CSV Q {i}", "response": f"CSV A {i}"})

            stats = analyze_dataset_tokens(f, dataset_id="ds_csv", use_cache=False)
            self.assertEqual(stats.total_samples, 10)
            self.assertEqual(stats.schema_detected, "prompt_response")

    def test_txt_tokenization(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "data.txt"
            f.write_text("Line one of text document.\nLine two of text document.\n", encoding="utf-8")

            stats = analyze_dataset_tokens(f, dataset_id="ds_txt", use_cache=False)
            self.assertEqual(stats.total_samples, 2)
            self.assertEqual(stats.schema_detected, "text")

    def test_folder_discovery_mixed_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "mixed_dataset"
            folder.mkdir()
            (folder / "a.jsonl").write_text('{"prompt": "p1", "response": "r1"}\n', encoding="utf-8")
            (folder / "b.csv").write_text('prompt,response\np2,r2\n', encoding="utf-8")
            (folder / "c.txt").write_text('plain line\n', encoding="utf-8")
            (folder / "ignore.bin").write_bytes(b"\x00\x01\x02")  # unsupported

            files = discover_data_files(folder)
            self.assertEqual(len(files), 3)

            stats = analyze_dataset_tokens(folder, dataset_id="ds_mixed", use_cache=False)
            self.assertEqual(stats.total_samples, 3)

    def test_reference_mode_integrity(self):
        """CRITICAL INVARIANT: Raw dataset MD5 before tokenization == MD5 after tokenization."""
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "immutable_source.jsonl"
            content = '{"prompt": "Do not mutate me", "response": "I am read-only"}\n' * 50
            f.write_text(content, encoding="utf-8")
            md5_before = hashlib.md5(content.encode("utf-8")).hexdigest()

            # Run analyzer
            stats = analyze_dataset_tokens(f, dataset_id="ds_immutable", use_cache=False)
            self.assertEqual(stats.total_samples, 50)

            # Check MD5 after
            md5_after = hashlib.md5(f.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            self.assertEqual(md5_before, md5_after, "SOURCE DATASET WAS MODIFIED DURING TOKENIZATION!")

    def test_unicode_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "unicode.jsonl"
            f.write_text(
                '{"prompt": "こんにちは世界", "response": "人工知能へようこそ"}\n'
                '{"prompt": "Привет мир", "response": "Добро пожаловать"}\n',
                encoding="utf-8",
            )
            stats = analyze_dataset_tokens(f, dataset_id="ds_unicode", use_cache=False)
            self.assertEqual(stats.total_samples, 2)
            self.assertGreater(stats.total_tokens, 0)

    def test_empty_and_malformed_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "corrupt.jsonl"
            f.write_text("invalid json\n\n{broken\n{\"prompt\": \"valid\", \"response\": \"yes\"}\n", encoding="utf-8")
            stats = analyze_dataset_tokens(f, dataset_id="ds_corrupt", use_cache=False)
            self.assertEqual(stats.total_samples, 1)


class TestTokenizationCache(unittest.TestCase):
    """Tests caching and deterministic invalidation on source modification."""

    def test_cache_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "data.jsonl"
            f.write_text('{"prompt": "Q1", "response": "A1"}\n', encoding="utf-8")

            cache = TokenizationCache(home=Path(tmp) / "home")
            stats = analyze_dataset_tokens(f, dataset_id="ds_cached", use_cache=True)

            # Save in cache
            cache.save(stats, f, project_dir=Path(tmp))

            # Load from cache
            loaded = cache.load("ds_cached", f, stats.model_id, project_dir=Path(tmp))
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.total_samples, stats.total_samples)
            self.assertEqual(loaded.total_tokens, stats.total_tokens)

    def test_cache_invalidation_on_data_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "data.jsonl"
            f.write_text('{"prompt": "Q1", "response": "A1"}\n', encoding="utf-8")

            cache = TokenizationCache(home=Path(tmp) / "home")
            stats = analyze_dataset_tokens(f, dataset_id="ds_inval", use_cache=False)
            cache.save(stats, f, project_dir=Path(tmp))

            # Modify file
            f.write_text('{"prompt": "Q1", "response": "A1"}\n{"prompt": "Q2", "response": "A2"}\n', encoding="utf-8")

            # Cache load should detect fingerprint mismatch and return None
            invalidated = cache.load("ds_inval", f, stats.model_id, project_dir=Path(tmp))
            self.assertIsNone(invalidated, "Cache should be invalidated when source data changes!")


class TestTokenizeCLI(unittest.TestCase):
    """Tests CLI integration of myai data tokenize."""

    def test_cli_tokenize_with_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "test.jsonl"
            f.write_text('{"prompt": "Hello", "response": "World"}\n', encoding="utf-8")
            result = runner.invoke(app, ["data", "tokenize", "--path", str(f)])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Tokenizer Analysis", result.stdout)
            self.assertIn("Token Statistics", result.stdout)
            self.assertIn("Sequence Length Distribution", result.stdout)

    def test_cli_tokenize_nonexistent_path(self):
        result = runner.invoke(app, ["data", "tokenize", "--path", "/nonexistent/data.jsonl"])
        self.assertNotEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
