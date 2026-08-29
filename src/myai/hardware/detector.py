import os
import psutil
from dataclasses import dataclass

@dataclass
class HardwareReport:
    cpu: str
    ram_gb: float
    disk_gb: float
    gpu: str
    vram_gb: float
    tier: str

    @property
    def has_gpu(self) -> bool:
        return self.vram_gb > 0 and self.gpu not in ("None detected", "PyTorch not installed")

    @property
    def free_storage_gb(self) -> float:
        return self.disk_gb

def detect_hardware() -> HardwareReport:
    cpu = f"{psutil.cpu_count(logical=False)} cores"
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)

    # Check disk space where models and runs are stored (~/.myai)
    from ..core.home import get_home
    home_path = get_home()
    if os.name == "nt":
        drive = os.path.splitdrive(str(home_path.resolve()))[0] + os.sep
    else:
        drive = "/"
    disk_gb = round(psutil.disk_usage(drive).free / (1024**3), 1)
    
    gpu_name = "None detected"
    vram_gb = 0.0
    tier = "T0"
    
    try:
        import importlib
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            if vram_gb >= 24: tier = "T3"
            elif vram_gb >= 12: tier = "T2"
            elif vram_gb >= 8: tier = "T1"
    except ImportError:
        gpu_name = "PyTorch not installed"

    # CPU-only fallback: if no GPU but sufficient RAM, mark as T1 (CPU-capable)
    if tier == "T0" and ram_gb >= 8:
        tier = "T1"

    return HardwareReport(cpu, ram_gb, disk_gb, gpu_name, vram_gb, tier)