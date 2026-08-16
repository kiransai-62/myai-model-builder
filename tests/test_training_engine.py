import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typer.testing import CliRunner

from myai.training.runs import Run, RunManager
from myai.training.failure import (
    TrainingInterrupted,
    classify,
    require_disk,
    disk_free_gb,
)
from myai.training.live_ui import (
    TrainingDisplay,
    fmt_time,
    LiveCallback,
    DiskWatchCallback,
)
from myai.models.trained_registry import (
    register_trained,
    list_trained,
    resolve_adapter,
)
from myai.evaluation.report import EvaluationReport, MetricResult
from myai.cli.main import app

class TestTrainingEngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="myai_train_test_"))
        self.runner = CliRunner()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_and_run_manager(self):
        manager = RunManager(self.temp_dir)
        cfg_data = {
            "project": "fittrack",
            "base_model": "qwen2.5-3b-instruct",
            "dataset_id": "ds_001",
            "training_method": "qlora",
        }
        run1 = manager.create(cfg_data)
        self.assertTrue(run1.run_id.startswith("run_"))
        self.assertEqual(run1.read_config()["project"], "fittrack")

        # Test writing metrics and results
        run1.write_metrics([{"loss": 1.2, "step": 10}, {"loss": 0.8, "step": 20}])
        run1.write_result("SUCCESS", extra={"steps": 20})
        res = run1.read_result()
        self.assertIsNotNone(res)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["steps"], 20)

        # Test ID increments
        run2 = manager.create(cfg_data)
        self.assertNotEqual(run1.run_id, run2.run_id)
        self.assertTrue(
            run2.run_id.endswith("_002")
            or int(run2.run_id.split("_")[-1]) > int(run1.run_id.split("_")[-1])
        )

        # Test listing
        all_runs = manager.list()
        self.assertEqual(len(all_runs), 2)
        self.assertEqual(all_runs[0]["run_id"], run2.run_id)

    def test_resumable_run_detection(self):
        manager = RunManager(self.temp_dir)
        cfg_data = {
            "project": "fittrack",
            "base_model": "qwen2.5-3b-instruct",
            "dataset_id": "ds_001",
        }
        run = manager.create(cfg_data)

        # Without checkpoints, not resumable
        run.write_result("INTERRUPTED", reason="Disk full")
        self.assertIsNone(
            manager.find_resumable("fittrack", "qwen2.5-3b-instruct", "ds_001")
        )

        # Add a checkpoint
        ckpt_dir = run.ckpt_dir / "checkpoint-50"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "adapter_model.bin").write_text("test")

        resumable = manager.find_resumable("fittrack", "qwen2.5-3b-instruct", "ds_001")
        self.assertIsNotNone(resumable)
        self.assertEqual(resumable.run_id, run.run_id)
        self.assertEqual(resumable.latest_checkpoint().name, "checkpoint-50")

    def test_failure_classification(self):
        # 1. VRAM
        ti_vram = classify(
            RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), "training"
        )
        self.assertEqual(ti_vram.code, "VRAM")
        self.assertIn("smaller batch size", ti_vram.hint)

        # 2. Disk
        ti_disk = classify(OSError(28, "No space left on device"), "training")
        self.assertEqual(ti_disk.code, "DISK")

        # 3. Model load phase
        ti_model = classify(FileNotFoundError("weights not found"), "loading model")
        self.assertEqual(ti_model.code, "MODEL_LOAD")
        self.assertIn("myai model add", ti_model.hint)

        # 4. Dataset load phase
        ti_data = classify(ValueError("corrupt jsonl"), "loading dataset")
        self.assertEqual(ti_data.code, "DATASET")

        # 5. CUDA
        ti_cuda = classify(RuntimeError("CUDA driver failure error"), "training")
        self.assertEqual(ti_cuda.code, "CUDA")

        # 6. General crash
        ti_crash = classify(ZeroDivisionError("division by zero"), "preprocess")
        self.assertEqual(ti_crash.code, "CRASH")

    def test_require_disk(self):
        # Extremely large requirement should trigger TrainingInterrupted
        with self.assertRaises(TrainingInterrupted) as ctx:
            require_disk(self.temp_dir, 999999.0)
        self.assertEqual(ctx.exception.code, "DISK")
        self.assertIn("Insufficient disk space", ctx.exception.message)

    def test_live_ui_display_and_callbacks(self):
        self.assertEqual(fmt_time(125), "02:05")
        self.assertEqual(fmt_time(0), "00:00")

        display = TrainingDisplay(
            "run_20260815_001", "Qwen 2.5 3B", "my_dataset", "qlora", 3
        )
        display.step = 100
        display.total_steps = 200
        display.loss = 0.523
        display.vram_used = 6.5
        display.vram_total = 8.0
        display.checkpoints = ["checkpoint-100"]

        rendered = display.__rich__()
        self.assertIsNotNone(rendered)

        # Test LiveCallback
        manager = RunManager(self.temp_dir)
        run = manager.create({"project": "test"})
        cb = LiveCallback(display, run)
        
        class DummyState:
            max_steps = 500
            global_step = 150
            epoch = 2

        cb.on_train_begin(None, DummyState(), None)
        self.assertEqual(display.total_steps, 500)

        cb.on_log(None, DummyState(), None, logs={"loss": 0.4321})
        self.assertEqual(display.loss, 0.432)

        cb.on_step_end(None, DummyState(), None)
        self.assertEqual(display.step, 150)
        self.assertEqual(display.epoch, 2)

        # Test DiskWatchCallback
        disk_cb = DiskWatchCallback(self.temp_dir, min_free_gb=999999.0)
        class Step50State:
            global_step = 50
        with self.assertRaises(TrainingInterrupted):
            disk_cb.on_step_end(None, Step50State(), None)

    def test_trained_model_registry(self):
        dummy_adapter = self.temp_dir / "adapter_src"
        dummy_adapter.mkdir(parents=True)
        (dummy_adapter / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
        (dummy_adapter / "adapter_model.bin").write_text("WEIGHTS", encoding="utf-8")

        eval_report = EvaluationReport(
            eval_id="eval_20260815_001",
            model_id="fittrack-v1",
            dataset_id="ds_8f31",
            run_id="run_20260815_001",
            status="PASS",
            overall=0.92,
        )

        dst = register_trained(
            home=self.temp_dir,
            name="fittrack-v1",
            base_model_id="qwen2.5-3b-instruct",
            dataset_id="ds_8f31",
            run_id="run_20260815_001",
            adapter_src=dummy_adapter,
            eval_report=eval_report,
            method="QLORA",
        )
        self.assertTrue(dst.exists())
        self.assertTrue((dst / "metadata.json").exists())
        self.assertTrue((dst / "evaluation.json").exists())

        # List trained
        trained_list = list_trained(self.temp_dir)
        self.assertEqual(len(trained_list), 1)
        self.assertEqual(trained_list[0]["id"], "fittrack-v1")
        self.assertEqual(trained_list[0]["status"], "READY")

        # Resolve adapter
        adapter_path = resolve_adapter(self.temp_dir, "fittrack-v1")
        self.assertIsNotNone(adapter_path)
        self.assertTrue(adapter_path.exists())

    def test_cli_runs_list_and_info(self):
        import os
        old_env = os.environ.get("MYAI_HOME")
        os.environ["MYAI_HOME"] = str(self.temp_dir)
        try:
            manager = RunManager(self.temp_dir)
            run = manager.create({
                "project": "fittrack",
                "base_model": "qwen2.5-3b-instruct",
                "dataset_id": "ds_8f31",
            })
            run.write_result("SUCCESS", extra={"steps": 50})

            # myai runs list
            res_list = self.runner.invoke(app, ["runs", "list"])
            self.assertEqual(res_list.exit_code, 0)
            self.assertIn(run.run_id, res_list.stdout)
            self.assertIn("qwen2.5-3b-instruct", res_list.stdout)

            # myai runs info
            res_info = self.runner.invoke(app, ["runs", "info", run.run_id])
            self.assertEqual(res_info.exit_code, 0)
            self.assertIn("fittrack", res_info.stdout)
            self.assertIn("SUCCESS", res_info.stdout)

            # myai model list shows trained models
            res_models = self.runner.invoke(app, ["model", "list"])
            self.assertEqual(res_models.exit_code, 0)
            self.assertIn("Model Registry", res_models.stdout)
        finally:
            if old_env is not None:
                os.environ["MYAI_HOME"] = old_env
            else:
                os.environ.pop("MYAI_HOME", None)

if __name__ == "__main__":
    unittest.main()
