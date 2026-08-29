import tempfile
import shutil
import unittest
from pathlib import Path

from myai.core.goal import GoalProfile, TaskType, Domain, prompt_for_goal
from myai.core.config import ProjectConfig
from myai.hardware.detector import HardwareReport
from myai.data.validator import DataReport
from myai.models.schema import RegistryModel
from myai.models.recommender import recommend_models


class TestGoalUnderstanding(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="myai_goal_test_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_weights(self):
        profile = GoalProfile(task=TaskType.INSTRUCTION, domain=Domain.GENERAL)
        weights = profile.eval_weights
        self.assertAlmostEqual(sum(weights.values()), 1.0, delta=0.05)
        self.assertIn("rouge", weights)
        self.assertIn("bleu", weights)
        self.assertIn("readability", weights)
        self.assertIn("domain_accuracy", weights)
        self.assertIn("exact_match", weights)

    def test_code_weights(self):
        profile = GoalProfile(task=TaskType.CODE, domain=Domain.GENERAL)
        weights = profile.eval_weights
        self.assertAlmostEqual(sum(weights.values()), 1.0, delta=0.05)
        # Exact match should dominate code tasks
        self.assertTrue(weights["exact_match"] >= 0.5)

    def test_domain_chat_weights(self):
        profile = GoalProfile(task=TaskType.DOMAIN_QA, domain=Domain.FITNESS)
        weights = profile.eval_weights
        self.assertAlmostEqual(sum(weights.values()), 1.0, delta=0.05)
        # Domain accuracy and readability should be prioritized
        self.assertTrue(weights["domain_accuracy"] >= 0.3)
        self.assertTrue(weights["readability"] >= 0.3)

    def test_yaml_persistence(self):
        profile = GoalProfile(
            task=TaskType.DOMAIN_QA,
            domain=Domain.MEDICAL,
            context_priority="long-context",
            latency_priority="high-quality",
            target_deployment="server"
        )
        cfg = ProjectConfig(name="medical-ai", goal=profile)
        cfg.save(self.temp_dir)

        loaded_cfg = ProjectConfig.load(self.temp_dir)
        self.assertEqual(loaded_cfg.goal.task, TaskType.DOMAIN_QA)
        self.assertEqual(loaded_cfg.goal.domain, Domain.MEDICAL)
        self.assertEqual(loaded_cfg.goal.context_priority, "long-context")
        self.assertEqual(loaded_cfg.goal.latency_priority, "high-quality")
        self.assertEqual(loaded_cfg.goal.target_deployment, "server")
        self.assertTrue(loaded_cfg.goal.eval_weights["domain_accuracy"] >= 0.3)

    def test_non_interactive_prompt_for_goal(self):
        profile = prompt_for_goal(
            non_interactive=True,
            task="code",
            domain="finance",
            context_priority="short",
            latency_priority="fast",
            target_deployment="edge"
        )
        self.assertEqual(profile.task, TaskType.CODE)
        self.assertEqual(profile.domain, Domain.FINANCE)
        self.assertEqual(profile.target_deployment, "edge")
        self.assertTrue(profile.eval_weights["exact_match"] >= 0.4)

    def test_goal_driven_model_recommendations(self):
        dummy_models = [
            RegistryModel(
                id="qwen-coder-1.5b",
                name="Qwen 2.5 Coder 1.5B",
                parameters="1.5B",
                vram_min=4.0,
                methods=["LoRA", "QLoRA"],
                repository="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                license="Apache 2.0"
            ),
            RegistryModel(
                id="llama-3-8b-instruct",
                name="Llama 3 8B Instruct",
                parameters="8B",
                vram_min=12.0,
                methods=["QLoRA"],
                repository="meta-llama/Meta-Llama-3-8B-Instruct",
                license="Llama 3"
            )
        ]

        hw = HardwareReport(
            cpu="8 cores", ram_gb=32.0, disk_gb=200.0,
            gpu="NVIDIA RTX 4090", vram_gb=24.0, tier="T3"
        )
        data_rep = DataReport(examples=1000, tokens_approx=500_000, duplicates=0)

        # 1. Code Goal -> Coder model should receive significant boost
        code_goal = GoalProfile(task=TaskType.CODE, domain=Domain.GENERAL)
        recs_code = recommend_models(hw, data_rep, dummy_models, goal=code_goal)
        top_code = recs_code[0]
        self.assertEqual(top_code.model.id, "qwen-coder-1.5b")

        # 2. Edge deployment Goal -> Small model should be boosted, 8B model penalized
        edge_goal = GoalProfile(task=TaskType.CHAT, domain=Domain.GENERAL, target_deployment="edge")
        recs_edge = recommend_models(hw, data_rep, dummy_models, goal=edge_goal)
        top_edge = recs_edge[0]
        self.assertEqual(top_edge.model.id, "qwen-coder-1.5b")
