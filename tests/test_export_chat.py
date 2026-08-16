import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from myai.export.packager import build_package, build_zip_package, estimate_package_size
from myai.export.validator import validate_package


class TestExportChatUI(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="myai_chat_test_"))
        self.home_dir = self.test_dir / "myai_home"
        self.home_dir.mkdir(parents=True, exist_ok=True)

        # Setup mock trained model adapter
        self.adapter_dir = self.home_dir / "models" / "adapters" / "Qwen2.5-0.5B" / "test_model"
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        (self.adapter_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
        (self.adapter_dir / "adapter_model.bin").write_text("DUMMY_WEIGHTS", encoding="utf-8")
        (self.adapter_dir / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")

        # Setup mock trained model folder with evaluation.json
        self.trained_dir = self.home_dir / "models" / "trained" / "test_model"
        self.trained_dir.mkdir(parents=True, exist_ok=True)
        (self.trained_dir / "evaluation.json").write_text('{"overall": 0.95, "status": "PASS"}', encoding="utf-8")

        self.meta = {
            "id": "test_model",
            "base_model": "Qwen2.5-0.5B",
            "method": "LORA",
            "dataset": "test_dataset",
            "run_id": "run_test_001",
            "evaluation": "95%",
            "adapter_path": str(self.adapter_dir),
        }

    def test_build_package_includes_chat_ui(self):
        dest = self.test_dir / "exported_pkg"
        build_package(self.home_dir, self.meta, dest)

        # Check required files
        self.assertTrue((dest / "model" / "adapter_config.json").exists())
        self.assertTrue((dest / "metadata.json").exists())
        self.assertTrue((dest / "evaluation.json").exists())
        self.assertTrue((dest / "README.md").exists())
        self.assertTrue((dest / "loader.py").exists())

        # Check chat UI files
        self.assertTrue((dest / "chat" / "app.py").exists())
        self.assertTrue((dest / "chat" / "ui.py").exists())
        self.assertTrue((dest / "chat" / "config.json").exists())
        self.assertTrue((dest / "chat" / "web" / "index.html").exists())

        # Check README contains quick start chat instructions
        readme_content = (dest / "README.md").read_text(encoding="utf-8")
        self.assertIn("python chat/app.py", readme_content)

    def test_build_zip_and_validation(self):
        zip_path = self.test_dir / "test_model.zip"
        build_zip_package(self.home_dir, self.meta, zip_path)
        self.assertTrue(zip_path.exists())

        # Validate with validator
        result = validate_package(zip_path)
        self.assertTrue(result.passed, f"Validation failed with: {[c.name for c in result.failed_checks]}")

        # Check that 'Standalone Chat UI present' is in checks and passed
        chat_check = next((c for c in result.checks if c.name == "Standalone Chat UI present"), None)
        self.assertIsNotNone(chat_check)
        self.assertTrue(chat_check.passed)

    def test_chat_app_one_shot_execution(self):
        dest = self.test_dir / "pkg_run"
        build_package(self.home_dir, self.meta, dest)

        import subprocess
        cmd = [sys.executable, str(dest / "chat" / "app.py"), "How do I do pushups?"]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(dest))
        self.assertEqual(proc.returncode, 0)
        self.assertIn("test_model", proc.stdout)

    def test_chat_app_web_api(self):
        import urllib.request
        import time
        import subprocess

        dest = self.test_dir / "pkg_web"
        build_package(self.home_dir, self.meta, dest)

        # Launch web server with specific port and no browser
        port = 8795
        cmd = [sys.executable, str(dest / "chat" / "app.py"), "--port", str(port), "--no-browser"]
        proc = subprocess.Popen(cmd, cwd=str(dest))

        try:
            time.sleep(1.0)
            # Test GET /
            req = urllib.request.urlopen(f"http://127.0.0.1:{port}/")
            self.assertEqual(req.status, 200)
            html = req.read().decode("utf-8")
            self.assertIn("Ask our AI anything", html)
            self.assertIn("test_model", html)

            # Test POST /api/chat
            data = json.dumps({"prompt": "Hello test"}).encode("utf-8")
            post_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/chat",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            post_resp = urllib.request.urlopen(post_req)
            self.assertEqual(post_resp.status, 200)
            resp_json = json.loads(post_resp.read().decode("utf-8"))
            self.assertIn("response", resp_json)
            self.assertIn("test_model", resp_json["response"])
        finally:
            proc.terminate()
            proc.wait()



if __name__ == "__main__":
    unittest.main()
