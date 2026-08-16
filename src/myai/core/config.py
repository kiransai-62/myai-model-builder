from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class TrainingConfig:
    method: str = "qlora"          # qlora | lora
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    learning_rate: float = 2e-4
    seq_length: int = 1024

@dataclass
class GateConfig:
    threshold: float = 0.6
    top_k: int = 3

@dataclass
class EvaluationConfig:
    eval_split: float = 0.1        # configurable, not hard-coded
    seed: int = 42
    knowledge_min: float = 0.8
    task_min: float = 0.8
    regression_min: float = 0.95
    overall_min: float = 0.85

@dataclass
class ProjectConfig:
    name: str = "myai-project"
    data_path: str = "data"
    dataset_id: str = ""
    model_id: str = ""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def training_method(self) -> str:
        return self.training.method

    @property
    def epochs(self) -> int:
        return self.training.epochs

    @property
    def batch_size(self) -> int:
        return self.training.batch_size

    @property
    def gate_threshold(self) -> float:
        return self.gate.threshold

    @classmethod
    def load(cls, root: Path) -> "ProjectConfig":
        config_path = root / "myai.yaml"
        if not config_path.exists():
            return cls()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        t = raw.get("training", {})
        g = raw.get("gate", {})
        e = raw.get("evaluation", {})
        return cls(
            name=raw.get("project", {}).get("name", "myai-project"),
            data_path=raw.get("data", {}).get("path", "data"),
            dataset_id=raw.get("data", {}).get("dataset_id", "") or raw.get("dataset_id", ""),
            model_id=raw.get("model", {}).get("model_id", ""),
            training=TrainingConfig(
                method=t.get("method", "qlora"),
                epochs=t.get("epochs", 3),
                batch_size=t.get("batch_size", 4),
                grad_accum=t.get("grad_accum", 4),
                learning_rate=t.get("learning_rate", 2e-4),
                seq_length=t.get("seq_length", 1024),
            ),
            gate=GateConfig(
                threshold=g.get("threshold", 0.6),
                top_k=g.get("top_k", 3),
            ),
            evaluation=EvaluationConfig(
                eval_split=e.get("eval_split", 0.1),
                seed=e.get("seed", 42),
                knowledge_min=e.get("knowledge_min", 0.8),
                task_min=e.get("task_min", 0.8),
                regression_min=e.get("regression_min", 0.95),
                overall_min=e.get("overall_min", 0.85),
            ),
        )

    def save(self, root: Path):
        data = {
            "project": {"name": self.name},
            "data": {
                "path": self.data_path,
                "dataset_id": self.dataset_id,
            },
            "model": {"model_id": self.model_id},
            "training": {
                "method": self.training.method,
                "epochs": self.training.epochs,
                "batch_size": self.training.batch_size,
                "grad_accum": self.training.grad_accum,
                "learning_rate": self.training.learning_rate,
                "seq_length": self.training.seq_length,
            },
            "gate": {
                "threshold": self.gate.threshold,
                "top_k": self.gate.top_k,
            },
            "evaluation": {
                "eval_split": self.evaluation.eval_split,
                "seed": self.evaluation.seed,
                "knowledge_min": self.evaluation.knowledge_min,
                "task_min": self.evaluation.task_min,
                "regression_min": self.evaluation.regression_min,
                "overall_min": self.evaluation.overall_min,
            },
        }
        (root / "myai.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")