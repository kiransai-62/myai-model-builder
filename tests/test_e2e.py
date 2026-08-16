import json
import os
import sys
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from typer.testing import CliRunner

from myai.cli.main import app
from myai.core.config import ProjectConfig
from myai.models.trained_registry import list_trained
from myai.serving.runtime import MyAIRuntime, InferenceRequest

class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="myai_e2e_"))
        self.home_dir = self.test_dir / "myai_home"
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.project_dir = self.test_dir / "fitness_assistant"
        
        self.old_env = os.environ.get("MYAI_HOME")
        os.environ["MYAI_HOME"] = str(self.home_dir)
        self.old_cwd = os.getcwd()
        self.runner = CliRunner()

    def tearDown(self):
        os.chdir(self.old_cwd)
        if self.old_env is not None:
            os.environ["MYAI_HOME"] = self.old_env
        else:
            os.environ.pop("MYAI_HOME", None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_complete_e2e_pipeline(self):
        # ── 1. Init Project ───────────────────────────────────────────────────
        os.chdir(self.test_dir)
        res_init = self.runner.invoke(app, ["init", "fitness_assistant"])
        self.assertEqual(res_init.exit_code, 0, f"init output: {res_init.stdout}")
        self.assertTrue(self.project_dir.exists())
        self.assertTrue((self.project_dir / "myai.yaml").exists())

        os.chdir(self.project_dir)

        # ── 2. System Check ───────────────────────────────────────────────────
        res_sys = self.runner.invoke(app, ["system", "check"])
        self.assertEqual(res_sys.exit_code, 0, f"system check output: {res_sys.stdout}")
        self.assertIn("MYAI SYSTEM ANALYSIS", res_sys.stdout)

        # ── 3. Prepare Real Dataset & Evaluation Data ─────────────────────────
        dataset_source = self.test_dir / "raw_data"
        dataset_source.mkdir(parents=True, exist_ok=True)
        (dataset_source / "evaluation").mkdir(parents=True, exist_ok=True)

        qa_data = [
            {"prompt": "What is the refund policy?", "response": "FitTrack offers a 30-day money-back guarantee via support@fittrack.com."},
            {"prompt": "How much does the Pro subscription cost?", "response": "The Pro plan is $19.99 per month and includes a 7-day free trial."},
            {"prompt": "Which wearable devices are supported?", "response": "We support Apple Watch, Fitbit, Garmin, and Wear OS smartwatches."},
            {"prompt": "How do I reset my password?", "response": "Click forgot password on the login screen to receive a 24-hour reset link."},
            {"prompt": "Is my workout telemetry data private?", "response": "All biometric and workout data is encrypted and never sold to third parties."},
            {"prompt": "What should I eat after high-intensity training?", "response": "Consume 20-30g of protein and fast-digesting carbohydrates within 45 minutes."},
            {"prompt": "How many rest days should a beginner take?", "response": "Beginners should schedule 2-3 rest days per week for optimal recovery."},
            {"prompt": "Can I export my workout logs?", "response": "Yes, you can export workout logs as CSV or JSON from account settings."},
            {"prompt": "What is the maximum daily calorie goal?", "response": "Daily targets can be adjusted up to 10,000 calories in custom nutrition plans."},
            {"prompt": "Does the app work offline?", "response": "Workout tracking works offline and syncs automatically when reconnected."},
        ]

        with open(dataset_source / "faq.jsonl", "w", encoding="utf-8") as f:
            for item in qa_data:
                f.write(json.dumps(item) + "\n")

        with open(dataset_source / "knowledge.md", "w", encoding="utf-8") as f:
            f.write("# FitTrack Knowledge Base\n\nFitTrack offers a 30-day refund policy with dedicated email support at support@fittrack.com.\nThe Pro plan subscription costs $19.99 monthly.\nCompatible devices include Apple Watch, Fitbit, and Garmin.\n")

        eval_cases = [
            {
                "prompt": "What is the refund policy?",
                "required_facts": ["30-day", "support@fittrack.com"],
                "must_not_claim": ["no refund", "lifetime"],
                "expected_meaning": "FitTrack provides a 30-day refund guarantee via support@fittrack.com."
            },
            {
                "prompt": "How much does the Pro subscription cost?",
                "required_facts": ["$19.99", "month"],
                "must_not_claim": ["free forever"],
                "expected_meaning": "Pro membership is $19.99 per month."
            }
        ]

        with open(dataset_source / "evaluation" / "test_cases.jsonl", "w", encoding="utf-8") as f:
            for c in eval_cases:
                f.write(json.dumps(c) + "\n")

        # ── 4. Add & Scan Dataset ─────────────────────────────────────────────
        res_data_add = self.runner.invoke(app, ["data", "add", str(dataset_source), "--yes"])
        self.assertEqual(res_data_add.exit_code, 0, f"data add output: {res_data_add.stdout}")
        self.assertIn("READY FOR TRAINING", res_data_add.stdout)

        # ── 5. Validate & List Data ───────────────────────────────────────────
        res_data_val = self.runner.invoke(app, ["data", "validate"])
        self.assertEqual(res_data_val.exit_code, 0)
        self.assertIn("DATA ANALYSIS", res_data_val.stdout)

        res_data_list = self.runner.invoke(app, ["data", "list"])
        self.assertEqual(res_data_list.exit_code, 0)
        self.assertIn("raw_data", res_data_list.stdout)

        cfg = ProjectConfig.load(self.project_dir)
        self.assertTrue(len(cfg.dataset_id) > 0)

        res_data_info = self.runner.invoke(app, ["data", "info", cfg.dataset_id])
        self.assertEqual(res_data_info.exit_code, 0)
        self.assertIn(cfg.dataset_id, res_data_info.stdout)

        # ── 6. Model Recommendation ───────────────────────────────────────────
        res_rec = self.runner.invoke(app, ["recommend", "--apply", "--yes"])
        self.assertEqual(res_rec.exit_code, 0, f"recommend output: {res_rec.stdout}")
        self.assertIn("MYAI RECOMMENDATION ENGINE", res_rec.stdout)

        cfg = ProjectConfig.load(self.project_dir)
        self.assertTrue(len(cfg.model_id) > 0)

        # Create mock base model files for testing
        base_dir = self.home_dir / "models" / "base" / cfg.model_id
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "config.json").write_text(json.dumps({"architectures": ["CausalLM"]}), encoding="utf-8")
        (base_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (base_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

        res_models = self.runner.invoke(app, ["model", "list"])
        self.assertEqual(res_models.exit_code, 0)
        self.assertIn(cfg.model_id, res_models.stdout)

        # ── 7. Train Model ────────────────────────────────────────────────────
        res_train = self.runner.invoke(app, ["train", "--yes"])
        self.assertEqual(res_train.exit_code, 0, f"train output: {res_train.stdout}")
        self.assertIn("TRAINING COMPLETE", res_train.stdout)

        # Verify holdout was separated
        runs = list((self.home_dir / "runs").glob("run_*"))
        self.assertTrue(len(runs) > 0)
        latest_run = runs[-1]
        self.assertTrue((latest_run / "evaluation_holdout.jsonl").exists())

        # ── 8. Evaluate Model ─────────────────────────────────────────────────
        res_eval = self.runner.invoke(app, ["evaluate"])
        self.assertEqual(res_eval.exit_code, 0, f"evaluate output: {res_eval.stdout}")
        self.assertIn("MYAI EVALUATION", res_eval.stdout)
        self.assertIn("Status: PASS", res_eval.stdout)

        res_eval_list = self.runner.invoke(app, ["evaluate", "list"])
        self.assertEqual(res_eval_list.exit_code, 0)
        self.assertIn("PASS", res_eval_list.stdout)

        eval_dirs = list((latest_run / "evaluation").glob("eval_*"))
        self.assertTrue(len(eval_dirs) > 0)
        eval_id = eval_dirs[0].name
        res_eval_info = self.runner.invoke(app, ["evaluate", "info", eval_id])
        self.assertEqual(res_eval_info.exit_code, 0)
        self.assertIn(eval_id, res_eval_info.stdout)

        # ── 9. Model Registry (Trained) ───────────────────────────────────────
        trained = list_trained(self.home_dir)
        self.assertEqual(len(trained), 1)
        self.assertEqual(trained[0]["id"], "fitness_assistant")
        self.assertEqual(trained[0]["status"], "READY")
        self.assertIn("evaluation", trained[0])

        res_trained = self.runner.invoke(app, ["model", "trained"])
        self.assertEqual(res_trained.exit_code, 0)
        self.assertIn("fitness_assistant", res_trained.stdout)

        # ── 10. Knowledge Index & RAG Gate ────────────────────────────────────
        res_idx = self.runner.invoke(app, ["index", "build"])
        self.assertEqual(res_idx.exit_code, 0)
        self.assertIn("Indexed", res_idx.stdout)
        self.assertTrue((self.project_dir / "indexes" / "chunks.jsonl").exists())

        # In-domain ask
        res_ask_valid = self.runner.invoke(app, ["ask", "What is the refund policy?"])
        self.assertEqual(res_ask_valid.exit_code, 0)
        self.assertIn("ALLOWED", res_ask_valid.stdout)

        # Out-of-domain ask
        res_ask_refused = self.runner.invoke(app, ["ask", "Who won the FIFA 1994 World Cup tournament in football?"])
        self.assertEqual(res_ask_refused.exit_code, 0)
        self.assertIn("REFUSED", res_ask_refused.stdout)

        # ── 11. Model Export ──────────────────────────────────────────────────
        res_export = self.runner.invoke(app, ["export", "fitness_assistant", "--yes"])
        self.assertEqual(res_export.exit_code, 0, f"export output: {res_export.stdout}")
        self.assertIn("EXPORT", res_export.stdout)

        # ── 12. Portable Package Verification ─────────────────────────────────
        export_zip = Path(os.getcwd()) / "exports" / "fitness_assistant.zip"
        self.assertTrue(export_zip.exists(), f"ZIP not found at {export_zip}")

        # Verify ZIP contents
        with zipfile.ZipFile(export_zip, "r") as zf:
            names = zf.namelist()
            # Check required files exist inside the ZIP
            self.assertTrue(any("model/adapter_config.json" in n for n in names),
                            f"adapter_config.json not in ZIP: {names}")
            self.assertTrue(any("model/adapter_model.bin" in n for n in names),
                            f"adapter_model.bin not in ZIP: {names}")
            self.assertTrue(any("tokenizer/" in n for n in names),
                            f"tokenizer/ not in ZIP: {names}")
            self.assertTrue(any("metadata.json" in n for n in names),
                            f"metadata.json not in ZIP: {names}")
            self.assertTrue(any("README.md" in n for n in names),
                            f"README.md not in ZIP: {names}")
            self.assertTrue(any("loader.py" in n for n in names),
                            f"loader.py not in ZIP: {names}")
            self.assertTrue(any("chat/app.py" in n for n in names),
                            f"chat/app.py not in ZIP: {names}")
            self.assertTrue(any("chat/ui.py" in n for n in names),
                            f"chat/ui.py not in ZIP: {names}")
            self.assertTrue(any("chat/config.json" in n for n in names),
                            f"chat/config.json not in ZIP: {names}")
            self.assertTrue(any("chat/web/index.html" in n for n in names),
                            f"chat/web/index.html not in ZIP: {names}")

            # Verify metadata content
            meta_entry = next(n for n in names if n.endswith("metadata.json"))
            meta_content = json.loads(zf.read(meta_entry).decode("utf-8"))
            self.assertEqual(meta_content["package_type"], "myai-trained-model")
            self.assertEqual(meta_content["model_id"], "fitness_assistant")
            self.assertIn("base_model_repo", meta_content)
            self.assertIn("evaluation", meta_content)

        # ── 13. Test Standalone Chat UI Execution from Extracted ZIP ──────────
        extract_dir = self.test_dir / "extracted_pkg"
        with zipfile.ZipFile(export_zip, "r") as zf:
            zf.extractall(extract_dir)

        pkg_root = extract_dir / "fitness_assistant"
        self.assertTrue((pkg_root / "chat" / "app.py").exists())
        self.assertTrue((pkg_root / "chat" / "ui.py").exists())
        self.assertTrue((pkg_root / "chat" / "config.json").exists())

        # Test one-shot direct CLI invocation of standalone chat app
        import subprocess
        chat_cmd = [sys.executable, str(pkg_root / "chat" / "app.py"), "Hello fitness model"]
        proc = subprocess.run(chat_cmd, capture_output=True, text=True, cwd=str(pkg_root))
        self.assertEqual(proc.returncode, 0, f"chat error: {proc.stderr}")
        self.assertIn("fitness_assistant", proc.stdout)

if __name__ == "__main__":
    unittest.main()
