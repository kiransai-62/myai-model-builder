import unittest
from typer.testing import CliRunner

from myai.hardware.benchmark import run_hardware_benchmark, BenchmarkResult
from myai.cli.main import app


class TestHardwareBenchmark(unittest.TestCase):

    def test_run_hardware_benchmark(self):
        res = run_hardware_benchmark(steps=2)
        self.assertIsInstance(res, BenchmarkResult)
        self.assertGreater(res.forward_tokens_per_sec, 0.0)
        self.assertGreater(res.training_tokens_per_sec, 0.0)
        self.assertIn(res.measured_tier, ["T0", "T1", "T2", "T3"])

        est_time = res.estimate_minutes(num_samples=100, epochs=1)
        self.assertGreaterEqual(est_time, 0.0)

    def test_cli_system_benchmark(self):
        runner = CliRunner()
        result = runner.invoke(app, ["system", "benchmark", "--steps", "2"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("MYAI LIVE HARDWARE BENCHMARK", result.stdout)
        self.assertIn("Inference Throughput", result.stdout)
        self.assertIn("Training Throughput", result.stdout)
