import json
import time
from datetime import datetime
from pathlib import Path

class Run:
    def __init__(self, root: Path, run_id: str):
        self.root = Path(root)
        self.run_id = run_id
        self.log_dir = self.root / "logs"
        self.ckpt_dir = self.root / "checkpoints"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def write_config(self, cfg: dict):
        (self.root / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def read_config(self) -> dict:
        p = self.root / "config.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def write_result(self, status: str, reason: str = "", extra: dict = None):
        data = {
            "run_id": self.run_id,
            "status": status,
            "reason": reason,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **(extra or {}),
        }
        (self.root / "result.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    def write_metrics(self, log_history: list):
        (self.root / "metrics.json").write_text(json.dumps(log_history, indent=2), encoding="utf-8")

    def read_result(self) -> dict | None:
        p = self.root / "result.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def checkpoints(self) -> list:
        return sorted(
            [p.name for p in self.ckpt_dir.glob("checkpoint-*")],
            key=lambda n: int(n.split("-")[-1]) if n.split("-")[-1].isdigit() else 0,
        )

    def latest_checkpoint(self) -> Path | None:
        cks = self.checkpoints()
        return self.ckpt_dir / cks[-1] if cks else None


class RunManager:
    def __init__(self, home: Path):
        self.runs_dir = Path(home) / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _next_id(self) -> str:
        date = datetime.now().strftime("%Y%m%d")          # e.g. run_20260815_001
        nums = []
        for p in self.runs_dir.glob(f"run_{date}_*"):
            try:
                nums.append(int(p.name.rsplit("_", 1)[-1]))
            except ValueError:
                pass
        return f"run_{date}_{max(nums, default=0) + 1:03d}"

    def create(self, frozen_config: dict) -> Run:
        run_id = self._next_id()
        run = Run(self.runs_dir / run_id, run_id)
        run.write_config(frozen_config)
        return run

    def get(self, run_id: str) -> Run | None:
        root = self.runs_dir / run_id
        return Run(root, run_id) if root.exists() else None

    def list(self) -> list:
        out = []
        for p in sorted(self.runs_dir.glob("run_*"), reverse=True):
            run = Run(p, p.name)
            cfg = run.read_config() if (p / "config.json").exists() else {}
            result = run.read_result() or {}
            out.append({"run_id": p.name, "config": cfg, "result": result})
        return out

    def find_resumable(self, project: str, base_model: str, dataset_id: str) -> Run | None:
        for entry in self.list():
            cfg, result = entry["config"], entry["result"]
            if (
                cfg.get("project") == project
                and cfg.get("base_model") == base_model
                and cfg.get("dataset_id") == dataset_id
                and result.get("status") in ("INTERRUPTED", "FAILED")
            ):
                run = self.get(entry["run_id"])
                if run and run.latest_checkpoint():
                    return run
        return None
