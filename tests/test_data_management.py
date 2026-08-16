import os
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typer.testing import CliRunner

from myai.core.home import ensure_home, get_home, SUBDIRS
from myai.core.config import ProjectConfig
from myai.data.scanner import scan_directory, ScanResult, CATEGORY_BY_EXT
from myai.data.validator import validate_data, DataReport
from myai.data.manager import DatasetManager, resolve_dataset_source, make_dataset_id, _manifest_checksum
from myai.system.storage import StorageBudget, estimate_storage, print_budget
from myai.models.schema import RegistryModel
from myai.cli.main import app

class TestDataManagement(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="myai_datamgmt_"))
        self.home_dir = self.temp_dir / ".myai"
        os.environ["MYAI_HOME"] = str(self.home_dir)

        # Create external user dataset folder
        self.user_data_dir = self.temp_dir / "user_dataset"
        self.user_data_dir.mkdir(parents=True, exist_ok=True)

        # Create sample files of various modalities
        (self.user_data_dir / "training.json").write_text(
            json.dumps([
                {"prompt": "What is AI?", "response": "Artificial Intelligence."},
                {"prompt": "What is MYAI?", "response": "Local-first AI platform."}
            ]),
            encoding="utf-8"
        )
        (self.user_data_dir / "items.csv").write_text(
            "prompt,response\nHow are you?,I am good\nWhat time is it?,It is noon\n",
            encoding="utf-8"
        )
        img_dir = self.user_data_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        (img_dir / "sample.jpg").write_bytes(b"dummy_jpeg_bytes")
        (img_dir / "icon.png").write_bytes(b"dummy_png_bytes")
        (self.user_data_dir / "notes.txt").write_text("Some text notes.", encoding="utf-8")

        # Create a MYAI project workspace
        self.proj_dir = self.temp_dir / "project"
        self.proj_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = ProjectConfig(name="test-project", data_path="data", model_id="qwen2.5-1.5b-instruct")
        self.cfg.save(self.proj_dir)

        # Create dummy base model weights so tests don't download from network
        base_model_dir = self.proj_dir / "models" / "base" / "qwen2.5-1.5b-instruct"
        base_model_dir.mkdir(parents=True, exist_ok=True)
        (base_model_dir / "config.json").write_text("{}", encoding="utf-8")

        self.runner = CliRunner()

    def tearDown(self):
        os.environ.pop("MYAI_HOME", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ensure_home_creates_subdirs(self):
        """Test ensure_home creates all required directories in MYAI store."""
        home = ensure_home()
        self.assertEqual(home, self.home_dir)
        for sub in SUBDIRS:
            self.assertTrue((home / sub).is_dir(), f"Expected directory: {home / sub}")

    def test_scanner_categorization(self):
        """Test scan_directory correctly counts files, bytes, and categories."""
        scan = scan_directory(self.user_data_dir)
        self.assertEqual(scan.total_files, 5)
        self.assertGreater(scan.total_bytes, 0)
        self.assertEqual(scan.categories.get("json"), 1)
        self.assertEqual(scan.categories.get("csv"), 1)
        self.assertEqual(scan.categories.get("image"), 2)
        self.assertEqual(scan.categories.get("text"), 1)
        self.assertEqual(len(scan.errors), 0)

    def test_dataset_manager_registration_and_manifest(self):
        """Test registering dataset in DatasetManager."""
        scan = scan_directory(self.user_data_dir)
        val = validate_data(self.user_data_dir)
        mgr = DatasetManager(self.home_dir)

        meta = mgr.register("user_dataset", self.user_data_dir, scan, val)
        self.assertTrue(meta["dataset_id"].startswith("ds_"))
        self.assertEqual(meta["source"], str(self.user_data_dir))
        self.assertEqual(meta["total_files"], 5)
        self.assertEqual(meta["privacy"]["upload"], False)
        self.assertEqual(meta["privacy"]["location"], "local")
        self.assertTrue(meta["manifest_checksum"].startswith("sha256:"))

        # Check directory structure in .myai/datasets/<id>
        ds_path = self.home_dir / "datasets" / meta["dataset_id"]
        self.assertTrue((ds_path / "metadata.json").exists())
        self.assertTrue((ds_path / "processed").is_dir())
        self.assertTrue((ds_path / "cache").is_dir())

        # Check list & get
        datasets = mgr.list()
        self.assertEqual(len(datasets), 1)
        self.assertEqual(datasets[0]["dataset_id"], meta["dataset_id"])

        fetched = mgr.get(meta["dataset_id"])
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["name"], "user_dataset")

    def test_resolve_dataset_source(self):
        """Test resolve_dataset_source resolves original path when dataset_id is present."""
        mgr = DatasetManager(self.home_dir)
        scan = scan_directory(self.user_data_dir)
        val = validate_data(self.user_data_dir)
        meta = mgr.register("user_dataset", self.user_data_dir, scan, val)

        self.cfg.dataset_id = meta["dataset_id"]
        self.cfg.save(self.proj_dir)

        resolved = resolve_dataset_source(self.proj_dir, self.cfg)
        self.assertEqual(resolved.resolve(), self.user_data_dir.resolve())

        # Fallback when no dataset_id
        self.cfg.dataset_id = ""
        resolved_fallback = resolve_dataset_source(self.proj_dir, self.cfg)
        self.assertEqual(resolved_fallback, self.proj_dir / "data")

    def test_storage_budget_calculation(self):
        """Test StorageBudget estimation for QLoRA and LoRA."""
        dataset_bytes = 8 * 1024**3  # 8 GB
        budget_qlora = estimate_storage(dataset_bytes, model_billions=3.0, method="qlora", epochs=3)

        self.assertEqual(budget_qlora.dataset_gb, 8.0)
        self.assertEqual(budget_qlora.model_gb, 2.4)   # 3.0 * 0.8
        self.assertEqual(budget_qlora.cache_gb, 2.0)   # min(8*0.25, 5.0)
        # adapter_gb = max(0.05, 3.0*0.04) = 0.12, checkpoints_gb = 3 * 0.12 = 0.36 -> round 0.4
        self.assertEqual(budget_qlora.checkpoints_gb, 0.4)
        self.assertEqual(budget_qlora.additional_gb, 4.8)

        # Budget print & disk check
        from rich.console import Console
        c = Console(record=True)
        self.assertTrue(print_budget(budget_qlora, free_gb=20.0, console=c))
        self.assertFalse(print_budget(budget_qlora, free_gb=2.0, console=c))

    def test_cli_data_add_list_info(self):
        """Test CLI commands: myai data add, list, info."""
        old_cwd = os.getcwd()
        try:
            os.chdir(self.proj_dir)

            # 1. data add
            result = self.runner.invoke(app, ["data", "add", str(self.user_data_dir), "--yes"])
            self.assertEqual(result.exit_code, 0, f"data add output: {result.stdout}")
            self.assertIn("MYAI Dataset Manager", result.stdout)
            self.assertIn("LOCAL ONLY", result.stdout)
            self.assertIn("✓ Attached to project: test-project", result.stdout)

            # Check config updated
            updated_cfg = ProjectConfig.load(self.proj_dir)
            self.assertTrue(updated_cfg.dataset_id.startswith("ds_"))

            # 2. data list
            list_res = self.runner.invoke(app, ["data", "list"])
            self.assertEqual(list_res.exit_code, 0)
            self.assertIn(updated_cfg.dataset_id, list_res.stdout)
            self.assertIn("READY", list_res.stdout)

            # 3. data info
            info_res = self.runner.invoke(app, ["data", "info"])
            self.assertEqual(info_res.exit_code, 0)
            self.assertIn("DATASET INFORMATION", info_res.stdout)
            self.assertIn(updated_cfg.dataset_id, info_res.stdout)
            self.assertIn("Manifest Checksum", info_res.stdout)

            # 4. data validate
            val_res = self.runner.invoke(app, ["data", "validate"])
            self.assertEqual(val_res.exit_code, 0)
            self.assertIn("DATA ANALYSIS", val_res.stdout)
            self.assertIn("Examples", val_res.stdout)

        finally:
            os.chdir(old_cwd)

    def test_cli_train_with_registered_dataset_and_storage_budget(self):
        """Test that train command checks storage budget and resolves external user dataset."""
        old_cwd = os.getcwd()
        try:
            os.chdir(self.proj_dir)
            # Register user dataset first
            self.runner.invoke(app, ["data", "add", str(self.user_data_dir), "--yes"])

            # Train with --yes
            result = self.runner.invoke(app, ["train", "--yes"])
            self.assertEqual(result.exit_code, 0, f"train output: {result.stdout}")
            self.assertIn("STORAGE BUDGET & DISK PROTECTION", result.stdout)
            self.assertIn("read in place", result.stdout)
            self.assertIn("Estimated additional storage", result.stdout)
            self.assertIn("TRAINING COMPLETE", result.stdout)

        finally:
            os.chdir(old_cwd)

if __name__ == "__main__":
    unittest.main()
