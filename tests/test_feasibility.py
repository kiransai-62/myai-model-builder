import unittest
from myai.hardware.detector import HardwareReport
from myai.hardware.feasibility import (
    TrainingConfig,
    estimate_vram_gb,
    check_feasibility,
    run_feasibility,
    FeasibilityReport,
    FeasibilityResult,
)
from myai.models.schema import RegistryModel
from myai.data.scorer import DatasetSummary


class TestFeasibility(unittest.TestCase):

    def setUp(self):
        self.hw_gpu_12gb = HardwareReport(
            cpu="12 cores", ram_gb=32.0, disk_gb=150.0,
            gpu="NVIDIA RTX 3060", vram_gb=12.0, tier="T2"
        )
        self.hw_gpu_4gb = HardwareReport(
            cpu="8 cores", ram_gb=16.0, disk_gb=80.0,
            gpu="NVIDIA GTX 1650", vram_gb=4.0, tier="T1"
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
            license="Llama-3",
            hidden_size=4096,
            num_layers=32,
            context_length=8192,
        )

        self.model_1b = RegistryModel(
            id="qwen2.5-1.5b",
            name="Qwen 2.5 1.5B",
            parameters="1.5B",
            vram_min=4.0,
            methods=["LoRA", "QLoRA"],
            repository="Qwen/Qwen2.5-1.5B-Instruct",
            license="Apache 2.0",
            hidden_size=1536,
            num_layers=28,
            context_length=32768,
        )

    def test_vram_quantization_scaling(self):
        cfg_4bit = TrainingConfig(quantization="4bit", lora_rank=16, seq_len=512, batch_size=1)
        cfg_8bit = TrainingConfig(quantization="8bit", lora_rank=16, seq_len=512, batch_size=1)
        cfg_fp16 = TrainingConfig(quantization="fp16", lora_rank=16, seq_len=512, batch_size=1)

        vram_4bit = estimate_vram_gb(self.model_8b, cfg_4bit)
        vram_8bit = estimate_vram_gb(self.model_8b, cfg_8bit)
        vram_fp16 = estimate_vram_gb(self.model_8b, cfg_fp16)

        self.assertLess(vram_4bit, vram_8bit)
        self.assertLess(vram_8bit, vram_fp16)
        self.assertTrue(5.0 <= vram_4bit <= 8.5)

    def test_dual_gate_pass_gpu_12gb(self):
        summary = DatasetSummary(num_samples=500, avg_tokens=256)
        feas = run_feasibility(self.hw_gpu_12gb, self.model_8b, data=summary)

        self.assertIsInstance(feas, FeasibilityResult)
        self.assertEqual(feas.overall, "PASS")
        self.assertTrue(feas.report.is_feasible)
        self.assertTrue(feas.report.hardware_fit)
        self.assertTrue(feas.report.data_fit)

    def test_dual_gate_fail_gpu_4gb_8b_model(self):
        summary = DatasetSummary(num_samples=500, avg_tokens=256)
        # 8B model will not fit in 4GB GPU
        feas = run_feasibility(self.hw_gpu_4gb, self.model_8b, data=summary)

        self.assertEqual(feas.overall, "FAIL")
        self.assertFalse(feas.report.is_feasible)
        self.assertFalse(feas.report.hardware_fit)
        self.assertTrue(any("exceeds" in w.lower() for w in feas.report.warnings))

    def test_cpu_mode_ram_budget(self):
        summary = DatasetSummary(num_samples=200, avg_tokens=150)
        feas = run_feasibility(self.hw_cpu_only, self.model_1b, data=summary)

        self.assertEqual(feas.overall, "PASS")
        self.assertTrue(feas.report.hardware_fit)
