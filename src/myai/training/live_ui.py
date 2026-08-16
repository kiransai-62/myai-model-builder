import time
from pathlib import Path
from rich.console import Group
from rich.text import Text

try:
    from transformers import TrainerCallback  # type: ignore
except ImportError:
    class TrainerCallback:
        pass

def fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"

class TrainingDisplay:
    def __init__(self, run_id, model_name, dataset_name, method, epochs):
        self.run_id = run_id
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.method = method
        self.epochs = epochs
        self.epoch, self.step, self.total_steps = 1, 0, 1
        self.loss = None
        self.checkpoints = []
        self.vram_used = self.vram_total = None
        self.gpu_util = None
        self.start = time.time()

    def _bar(self) -> str:
        pct = min(1.0, self.step / max(1, self.total_steps))
        filled = int(pct * 20)
        return "█" * filled + "░" * (20 - filled) + f" {pct * 100:.0f}%"

    def __rich__(self):
        t = Text()
        t.append("MYAI TRAINING\n", style="bold cyan")
        t.append(f"Run: {self.run_id}\n\n", style="dim")
        t.append(f"Model       {self.model_name}\n")
        t.append(f"Dataset     {self.dataset_name}\n")
        t.append(f"Method      {self.method.upper()}\n\n")
        t.append(f"Epoch       {self.epoch} / {self.epochs}\n")
        t.append(f"Step        {self.step:,} / {self.total_steps:,}\n\n")
        t.append(f"Loss        {self.loss if self.loss is not None else '—'}\n")
        t.append(f"Progress    {self._bar()}\n\n")
        if self.vram_used is not None:
            t.append(f"VRAM        {self.vram_used} / {self.vram_total} GB\n")
            t.append(f"GPU         {self.gpu_util if self.gpu_util is not None else '—'}\n\n")
        elapsed = time.time() - self.start
        speed = self.step / elapsed if elapsed > 0 and self.step > 0 else 0
        eta = (self.total_steps - self.step) / speed if speed > 0 else 0
        t.append(f"Elapsed     {fmt_time(elapsed)}\n")
        t.append(f"ETA         {fmt_time(eta)}\n\n")
        if self.checkpoints:
            t.append("Checkpoint:\n" + "\n".join(f"✓ {c}" for c in self.checkpoints), style="green")
        return Group(t)


class LiveCallback(TrainerCallback):
    def __init__(self, display: TrainingDisplay, run):
        self.display, self.run = display, run

    def on_train_begin(self, args, state, control, **kw):
        self.display.total_steps = max(1, state.max_steps)
        self.display.start = time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            self.display.loss = round(logs["loss"], 3)

    def on_step_end(self, args, state, control, **kw):
        self.display.step = state.global_step
        self.display.epoch = min(self.display.epochs, int(state.epoch or 1))

    def on_save(self, args, state, control, **kw):
        self.display.checkpoints = self.run.checkpoints()


class DiskWatchCallback(TrainerCallback):
    """Failure protection: abort early if the disk fills mid-training."""
    def __init__(self, home: Path, min_free_gb: float = 1.0):
        self.home, self.min_free = Path(home), min_free_gb

    def on_step_end(self, args, state, control, **kw):
        if state.global_step % 50 == 0:
            from .failure import disk_free_gb, TrainingInterrupted
            if disk_free_gb(self.home) < self.min_free:
                raise TrainingInterrupted(
                    "DISK", "Insufficient disk space.",
                    hint="Free space and resume from the latest checkpoint."
                )
