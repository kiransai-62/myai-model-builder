import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typer.testing import CliRunner

from myai.core.config import ProjectConfig
from myai.hardware.detector import HardwareReport
from myai.data.validator import DataReport
from myai.models.schema import RegistryModel
from myai.models.recommender import recommend_models, get_top_recommendation
from myai.models.registry import get_registry_models
from myai.training.engine import run_training_engine
from myai.training.runs import RunManager
from myai.data.manager import resolve_dataset_source
from myai.cli.main import app

class TestModelSelectionAndTraining(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="myai_test_"))
        self.data_dir = self.temp_dir / "data" / "train"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a sample domain dataset
        sample_data = [
            {"prompt": "What is the return policy?", "response": "30 days full refund guaranteed."},
            {"prompt": "How do I reset my password?", "response": "Click on Forgot Password on the login page."},
            {"prompt": "Where are your servers hosted?", "response": "All servers are in US-East regions."}
        ]
        with open(self.data_dir / "faq.json", "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
            
        self.cfg = ProjectConfig(name="test-project", data_path="data", model_id="qwen2.5-3b-instruct")
        self.cfg.save(self.temp_dir)

        # Populate dummy base models in test directory to prevent huggingface network downloads
        models = get_registry_models()
        for m in models:
            m_dir = self.temp_dir / "models" / "base" / m.id
            m_dir.mkdir(parents=True, exist_ok=True)
            (m_dir / "config.json").write_text("{}", encoding="utf-8")

        self.runner = CliRunner()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_recommender_scoring_cpu_vs_gpu(self):
        """Test scoring differences between GPU and CPU hardware configurations."""
        models = get_registry_models()
        self.assertTrue(len(models) >= 2, "Should have registry models loaded")

        # 1. CPU-only with 16GB RAM
        cpu_hw = HardwareReport(
            cpu="8 cores", ram_gb=16.0, disk_gb=100.0,
            gpu="None detected", vram_gb=0.0, tier="T1"
        )
        data_rep = DataReport(examples=3, tokens_approx=50, duplicates=0)
        recs_cpu = recommend_models(cpu_hw, data_rep, models)
        
        self.assertTrue(all(r.method == "LoRA" for r in recs_cpu))
        self.assertTrue(all(r.fits_vram for r in recs_cpu))
        
        # 2. Low-spec GPU (4GB VRAM)
        gpu_low = HardwareReport(
            cpu="8 cores", ram_gb=16.0, disk_gb=100.0,
            gpu="GTX 1650", vram_gb=4.0, tier="T0"
        )
        recs_gpu_low = recommend_models(gpu_low, data_rep, models)
        top_fit = recs_gpu_low[0]
        # Should recommend 0.5B, 1B, or 1.5B model that fits within 4GB
        self.assertTrue(top_fit.fits_vram)
        self.assertIn(top_fit.model.id, ["qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct", "llama-3.2-1b-instruct"])

    def _train_and_save_meta(self, spec, selection_mode: str) -> dict:
        manager = RunManager(self.temp_dir)
        source = resolve_dataset_source(self.temp_dir, self.cfg)
        run = manager.create({
            "project": self.cfg.name,
            "dataset_id": self.cfg.dataset_id,
            "base_model": spec.id,
            "selection_mode": selection_mode,
            "training_method": self.cfg.training.method,
            "epochs": self.cfg.training.epochs,
            "batch_size": self.cfg.training.batch_size,
            "learning_rate": self.cfg.training.learning_rate,
        })
        meta = run_training_engine(run, {
            "cfg": self.cfg,
            "spec": spec,
            "source": source,
            "home": self.temp_dir,
            "root": self.temp_dir,
            "selection_mode": selection_mode,
            "budget_gb": 0.0,
        })
        trained_dir = self.temp_dir / "models" / "trained" / self.cfg.name
        trained_dir.mkdir(parents=True, exist_ok=True)
        (trained_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def test_train_user_selected_model_provenance(self):
        """Verify that training on an explicit user-selected model logs user_selected in metadata."""
        models = get_registry_models()
        spec = next(m for m in models if m.id == "qwen2.5-3b-instruct")
        
        meta = self._train_and_save_meta(spec, selection_mode="user_selected")
        
        self.assertEqual(meta["base_model_id"], "qwen2.5-3b-instruct")
        self.assertEqual(meta["selection_mode"], "user_selected")
        self.assertEqual(meta["examples"], 3)
        
        saved_meta_path = self.temp_dir / "models" / "trained" / self.cfg.name / "metadata.json"
        self.assertTrue(saved_meta_path.exists())
        saved_meta = json.loads(saved_meta_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_meta["selection_mode"], "user_selected")
        self.assertEqual(saved_meta["base_model_id"], "qwen2.5-3b-instruct")

    def test_train_recommended_model_provenance(self):
        """Verify that training with system-recommended fallback logs recommended in metadata."""
        models = get_registry_models()
        spec = next(m for m in models if m.id == "qwen2.5-1.5b-instruct")
        
        self.cfg.model_id = spec.id
        meta = self._train_and_save_meta(spec, selection_mode="recommended")
        
        self.assertEqual(meta["base_model_id"], "qwen2.5-1.5b-instruct")
        self.assertEqual(meta["selection_mode"], "recommended")
        
        saved_meta = json.loads((self.temp_dir / "models" / "trained" / self.cfg.name / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_meta["selection_mode"], "recommended")

    def test_cli_train_with_model_override(self):
        """Verify myai train --model <id> --yes selects and trains the user-specified model."""
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            result = self.runner.invoke(app, ["train", "--model", "llama-3.2-1b-instruct", "--yes"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("User Selected", result.stdout)
            self.assertIn("llama-3.2-1b-instruct", result.stdout)
            
            # Check updated config and metadata
            updated_cfg = ProjectConfig.load(self.temp_dir)
            self.assertEqual(updated_cfg.model_id, "llama-3.2-1b-instruct")
            
            saved_meta = json.loads((self.temp_dir / "models" / "trained" / updated_cfg.name / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_meta["base_model_id"], "llama-3.2-1b-instruct")
            self.assertEqual(saved_meta["selection_mode"], "user_selected")
        finally:
            os.chdir(old_cwd)

    def test_cli_train_with_auto_recommend_fallback(self):
        """Verify myai train --yes automatically selects and trains recommended model when no model is set."""
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            # Clear model_id
            cfg = ProjectConfig.load(self.temp_dir)
            cfg.model_id = ""
            cfg.save(self.temp_dir)
            
            result = self.runner.invoke(app, ["train", "--yes"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("MODEL AUTO-RECOMMENDATION", result.stdout)
            self.assertIn("Recommended", result.stdout)
            
            updated_cfg = ProjectConfig.load(self.temp_dir)
            self.assertTrue(len(updated_cfg.model_id) > 0)
            
            saved_meta = json.loads((self.temp_dir / "models" / "trained" / updated_cfg.name / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_meta["selection_mode"], "recommended")
        finally:
            os.chdir(old_cwd)

    def test_cli_recommend_apply(self):
        """Verify myai recommend --apply sets the top model in myai.yaml."""
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            result = self.runner.invoke(app, ["recommend", "--apply", "--yes"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Project active model updated to", result.stdout)
            
            updated_cfg = ProjectConfig.load(self.temp_dir)
            self.assertTrue(len(updated_cfg.model_id) > 0)
        finally:
            os.chdir(old_cwd)

if __name__ == "__main__":
    unittest.main()
