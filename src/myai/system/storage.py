from dataclasses import dataclass

@dataclass
class StorageBudget:
    dataset_gb: float       # original — read in place, NOT additional
    model_gb: float
    cache_gb: float
    checkpoints_gb: float

    @property
    def additional_gb(self) -> float:
        return round(self.model_gb + self.cache_gb + self.checkpoints_gb, 1)

def estimate_storage(dataset_bytes: int, model_billions: float, method: str, epochs: int) -> StorageBudget:
    dataset_gb = dataset_bytes / 1024**3
    b = model_billions or 3.0

    m = (method or "qlora").lower()
    model_gb = b * 0.8 if m == "qlora" else b * 2.0        # 4-bit weights vs fp16
    cache_gb = min(dataset_gb * 0.25, 5.0)                     # processed cache only
    adapter_gb = max(0.05, b * 0.04)
    checkpoints_gb = epochs * (adapter_gb if m in ("lora", "qlora") else b * 2.0)

    return StorageBudget(
        dataset_gb=round(dataset_gb, 1),
        model_gb=round(model_gb, 1),
        cache_gb=round(cache_gb, 1),
        checkpoints_gb=round(checkpoints_gb, 1),
    )

def print_budget(budget: StorageBudget, free_gb: float, console) -> bool:
    console.print(f"Dataset                         {budget.dataset_gb} GB  [dim](read in place)[/dim]")
    console.print(f"Model                           {budget.model_gb} GB")
    console.print(f"Processed/cache                 {budget.cache_gb} GB")
    console.print(f"Checkpoints                     {budget.checkpoints_gb} GB")
    console.print(f"[bold]Estimated additional storage   {budget.additional_gb} GB[/bold]\n")

    if budget.additional_gb > free_gb:
        console.print(f"[bold red]⚠ You have only {free_gb} GB free. Training cannot safely proceed.[/bold red]")
        return False
    return True
