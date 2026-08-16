import errno
import shutil
from pathlib import Path

class TrainingInterrupted(Exception):
    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint

def disk_free_gb(path: Path) -> float:
    try:
        return round(shutil.disk_usage(str(path)).free / (1024**3), 1)
    except Exception:
        return 999.0

def require_disk(path: Path, needed_gb: float):
    free = disk_free_gb(path)
    if free < needed_gb:
        raise TrainingInterrupted(
            "DISK", "Insufficient disk space.",
            hint=f"You can resume after freeing: {round(needed_gb - free, 1)} GB"
        )

def classify(exc: Exception, phase: str, needed_gb: float = 0.0, home: Path = None) -> TrainingInterrupted:
    msg = str(exc).lower()

    if "out of memory" in msg or "oom" in msg or type(exc).__name__ == "OutOfMemoryError":
        return TrainingInterrupted(
            "VRAM", "GPU ran out of VRAM.",
            hint="Resume with a smaller batch size or lower sequence length."
        )

    if (isinstance(exc, OSError) and exc.errno == errno.ENOSPC) or "no space left" in msg:
        free = disk_free_gb(home) if home else 0
        return TrainingInterrupted(
            "DISK", "Insufficient disk space.",
            hint=f"You can resume after freeing: {max(0.0, round(needed_gb - free, 1))} GB"
        )

    if phase == "loading model":
        return TrainingInterrupted(
            "MODEL_LOAD", f"Model loading failed: {exc}",
            hint="Re-run `myai model add` to repair the base model."
        )

    if phase == "loading dataset":
        return TrainingInterrupted(
            "DATASET", f"Dataset failure: {exc}",
            hint="Your training data is safe. Fix the dataset and resume."
        )

    if "cuda" in msg and ("error" in msg or "failure" in msg):
        return TrainingInterrupted(
            "CUDA", f"CUDA failure: {exc}",
            hint="Check GPU drivers. Resume once the GPU is healthy."
        )

    return TrainingInterrupted(
        "CRASH", f"Unexpected failure during {phase}: {exc}",
        hint="Your training data is safe. Resume from the latest checkpoint."
    )
