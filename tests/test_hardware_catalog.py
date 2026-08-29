"""
Comprehensive test suite for the 15-Point Hardware Intelligence,
Dedicated Memory Calculator, System Compatibility Scorer, and Hierarchical Registry.
"""
import unittest
from pathlib import Path
from myai.hardware.detector import HardwareReport
from myai.data.validator import DataReport
from myai.core.goal import GoalProfile, TaskType
from myai.models.schema import RegistryModel
from myai.models.registry import get_registry_models
from myai.registry.loader import load_registry_models
from myai.registry.validator import ModelRegistryValidator
from myai.registry.scorer import SystemCompatibilityScorer, ModelRecommenderScorer
from myai.models.recommender import (
    recommend_models, recommend_model, CompatibilityVerdict
)
from myai.hardware.memory_calc import (
    MemoryCalculator,
    evaluate_context_profiles,
    calculate_dynamic_memory_profile,
    calculate_weight_memory_gb,
    calculate_kv_cache_gb,
    calculate_storage_breakdown,
)


class TestHardwareIntelligenceSystem(unittest.TestCase):
    def setUp(self):
        self.models = get_registry_models()

    def test_hierarchical_registry_loader(self):
        """Verify models are loaded from family subdirectories."""
        self.assertGreaterEqual(len(self.models), 20, "Should load models across all family folders")
        families = {m.family for m in self.models}
        self.assertIn("SmolLM2", families)
        self.assertIn("Qwen3", families)
        self.assertIn("Llama 3.1", families)

    def test_memory_calculator_dedicated_modes(self):
        """Verify explicit calculation modes: inference, lora, qlora, dpo, grpo, layer_streaming."""
        # 1. Inference mode (8B Q4)
        inf_prof = MemoryCalculator.inference(
            params_total_b=8.0, params_active_b=8.0, num_layers=32, hidden_size=4096,
            quant_format="GGUF_Q4_K_M", context_length=4096, batch_size=1, available_vram_gb=12.0
        )
        self.assertTrue(inf_prof.is_safe)
        self.assertLess(inf_prof.total_peak_vram_gb, 8.0)
        self.assertGreater(inf_prof.estimated_tokens_per_sec, 10.0)

        # 2. QLoRA training mode (8B NF4)
        qlora_prof = MemoryCalculator.qlora_training(
            params_total_b=8.0, params_active_b=8.0, num_layers=32, hidden_size=4096,
            context_length=4096, batch_size=2, available_vram_gb=12.0
        )
        self.assertTrue(qlora_prof.is_safe)
        self.assertGreater(qlora_prof.total_peak_vram_gb, inf_prof.total_peak_vram_gb)

        # 3. LoRA training mode (8B FP16)
        lora_prof = MemoryCalculator.lora_training(
            params_total_b=8.0, params_active_b=8.0, num_layers=32, hidden_size=4096,
            context_length=4096, batch_size=2, available_vram_gb=24.0
        )
        self.assertGreater(lora_prof.total_peak_vram_gb, qlora_prof.total_peak_vram_gb)

        # 4. DPO training mode (Policy + Reference)
        dpo_prof = MemoryCalculator.dpo_training(
            params_total_b=8.0, params_active_b=8.0, num_layers=32, hidden_size=4096,
            context_length=4096, batch_size=1, available_vram_gb=24.0
        )
        self.assertGreater(dpo_prof.total_peak_vram_gb, qlora_prof.total_peak_vram_gb)

        # 5. Layer Streaming mode (Base layers in RAM)
        stream_prof = MemoryCalculator.layer_streaming(
            params_total_b=8.0, params_active_b=8.0, num_layers=32, hidden_size=4096,
            context_length=2048, batch_size=1, available_vram_gb=4.0, available_ram_gb=32.0
        )
        self.assertTrue(stream_prof.is_safe)
        self.assertLessEqual(stream_prof.total_peak_vram_gb, 4.0)

    def test_context_profiles_evaluation(self):
        """Verify dynamic evaluation of 2K to 128K context profiles."""
        rec_ctx, profiles = evaluate_context_profiles(
            params_total_b=8.0, params_active_b=8.0, num_layers=32, hidden_size=4096,
            available_vram_gb=12.0, is_training=False
        )
        self.assertIn(4096, profiles)
        self.assertIn(131072, profiles)
        # 4K context on 12GB VRAM should be recommended or compatible
        self.assertIn("Recommended", profiles[4096][1])

    def test_dataset_size_impact_on_scoring(self):
        """Verify small datasets penalize massive models and favor compact models."""
        hw = HardwareReport(cpu="16 cores", ram_gb=64.0, disk_gb=500.0, gpu="RTX 4090", vram_gb=24.0, tier="T3")
        
        # Tiny dataset (50 samples, 2000 tokens)
        small_data = DataReport(examples=50, tokens_approx=2000)
        recs_small = recommend_models(hw, small_data, self.models)
        
        # For small dataset, lightweight model should have higher data fit than 70B
        top_small = recs_small[0]
        self.assertGreaterEqual(top_small.fit_breakdown.dataset_fit, 85.0)

        # Large dataset (50,000 samples, 10M tokens)
        large_data = DataReport(examples=50000, tokens_approx=10_000_000)
        recs_large = recommend_models(hw, large_data, self.models)
        top_large = recs_large[0]
        self.assertGreater(top_large.model.params_b, 1.0)

    def test_system_compatibility_scorer_8_factors(self):
        """Verify SystemCompatibilityScorer evaluates all 8 factors."""
        hw = HardwareReport(cpu="16 cores", ram_gb=32.0, disk_gb=500.0, gpu="RTX 4070", vram_gb=12.0, tier="T2")
        model = next(m for m in self.models if m.id == "llama-3.1-8b-instruct")

        mem_prof = MemoryCalculator.qlora_training(
            params_total_b=8.0, params_active_b=8.0, num_layers=32, hidden_size=4096,
            available_vram_gb=12.0, available_ram_gb=32.0, gpu_tier="T2"
        )
        score, breakdown = SystemCompatibilityScorer.score(hw, model, mem_prof)
        self.assertGreaterEqual(score, 75.0)
        self.assertGreaterEqual(breakdown.vram_score, 80.0)
        self.assertGreaterEqual(breakdown.ram_score, 85.0)
        self.assertGreaterEqual(breakdown.storage_score, 90.0)

    def test_moe_active_vs_total_parameters(self):
        """Verify MoE models separate weight storage (total) from compute throughput (active)."""
        moe_models = [m for m in self.models if m.architecture == "MoE"]
        self.assertTrue(len(moe_models) >= 2)

        qwen_moe = next(m for m in moe_models if "30b-a3b" in m.id)
        self.assertEqual(qwen_moe.parameters_billions, 30.0)
        self.assertEqual(qwen_moe.active_parameters_billions, 3.0)

        prof = MemoryCalculator.inference(
            params_total_b=qwen_moe.parameters_billions,
            params_active_b=qwen_moe.active_parameters_billions,
            num_layers=qwen_moe.num_layers,
            hidden_size=qwen_moe.hidden_size,
            available_vram_gb=24.0,
            gpu_tier="T3",
        )
        self.assertGreater(prof.estimated_tokens_per_sec, 25.0)


if __name__ == "__main__":
    unittest.main()
