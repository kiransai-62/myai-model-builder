import unittest
from myai.hardware.detector import HardwareReport
from myai.hardware.feasibility import TrainingConfig, estimate_vram_gb, check_feasibility
from myai.models.schema import RegistryModel
from myai.core.goal import GoalProfile, TaskType
from myai.training.strategy import plan_strategy, TrainingStrategy, estimate_storage_gb


class TestTrainingStrategyPlanner(unittest.TestCase):

    def setUp(self):
        self.hw_gpu_12gb = HardwareReport(
            cpu="12 cores", ram_gb=32.0, disk_gb=150.0,
            gpu="NVIDIA GeForce RTX 3060", vram_gb=12.0, tier="T2"
        )
        self.hw_cpu_only = HardwareReport(
            cpu="8 cores", ram_gb=16.0, disk_gb=80.0,
            gpu="None detected", vram_gb=0.0, tier="T1"
        )
        self.model_8b = RegistryModel(
            id="llama-3-8b-instruct",
            name="Llama 3 8B Instruct",
            parameters="8B",
            vram_min=12.0,
            methods=["QLoRA"],
            repository="meta-llama/Meta-Llama-3-8B-Instruct",
            license="Llama-3"
        )
        self.model_1b = RegistryModel(
            id="qwen2.5-1.5b",
            name="Qwen 2.5 1.5B",
            parameters="1.5B",
            vram_min=4.0,
            methods=["LoRA", "QLoRA"],
            repository="Qwen/Qwen2.5-1.5B-Instruct",
            license="Apache 2.0"
        )

    def test_vram_estimation(self):
        cfg = TrainingConfig(quantization="4bit", lora_rank=16, seq_len=512, batch_size=1, grad_checkpointing=True)
        vram_est = estimate_vram_gb(self.model_8b, cfg)
        self.assertTrue(5.0 <= vram_est <= 8.5, f"8B 4-bit model should estimate ~6-8GB VRAM, got {vram_est}GB")

    def test_plan_strategy_gpu_12gb(self):
        class MockData:
            num_samples = 1240
            avg_tokens = 330
            quality_score = 78

        goal = GoalProfile(task=TaskType.DOMAIN_QA, context_priority="balanced")
        strategy = plan_strategy(self.hw_gpu_12gb, self.model_8b, data=MockData(), goal=goal)

        self.assertIsInstance(strategy, TrainingStrategy)
        self.assertEqual(strategy.config.quantization, "4bit")
        self.assertEqual(strategy.config.lora_rank, 16)
        self.assertEqual(strategy.epochs, 2)
        self.assertTrue(strategy.estimated_vram_gb < 12.0)
        self.assertTrue(strategy.storage_required_gb > 0)
        self.assertTrue(len(strategy.reasoning) >= 2)

    def test_plan_strategy_cpu_fallback(self):
        class MockData:
            num_samples = 250
            avg_tokens = 200

        goal = GoalProfile(task=TaskType.INSTRUCTION)
        strategy = plan_strategy(self.hw_cpu_only, self.model_1b, data=MockData(), goal=goal)

        self.assertEqual(strategy.config.quantization, "4bit")
        self.assertEqual(strategy.config.lora_rank, 8)
        self.assertEqual(strategy.epochs, 4)  # Small dataset -> 4 epochs

    def test_strategy_user_override(self):
        class MockData:
            num_samples = 1000
            avg_tokens = 400

        override = {"learning_rate": 1e-4, "lora_rank": 32, "epochs": 5}
        strategy = plan_strategy(self.hw_gpu_12gb, self.model_8b, data=MockData(), override=override)

        self.assertEqual(strategy.learning_rate, 1e-4)
        self.assertEqual(strategy.config.lora_rank, 32)
        self.assertEqual(strategy.epochs, 5)
        self.assertEqual(strategy.confidence, 1.0)
        self.assertTrue(any("User override applied" in r for r in strategy.reasoning))

    def test_dual_feasibility_check(self):
        report = check_feasibility(self.hw_gpu_12gb, self.model_8b)
        self.assertTrue(report.is_feasible)
        self.assertTrue(report.hardware_fit)
        self.assertTrue(report.data_fit)
