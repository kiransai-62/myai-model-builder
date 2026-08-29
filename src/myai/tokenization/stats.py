"""MYAI Token Statistics & Distribution Analyzer.

Computes exact sample, input/output, and distribution token statistics,
as well as context-length compatibility against target model context limits.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


BUCKET_RANGES = [
    ("0-128", 0, 128),
    ("129-256", 129, 256),
    ("257-512", 257, 512),
    ("513-1024", 513, 1024),
    ("1025-2048", 1025, 2048),
    ("2049-4096", 2049, 4096),
    ("4097+", 4097, float("inf")),
]


@dataclass
class IOStats:
    min: int = 0
    max: int = 0
    avg: float = 0.0
    median: float = 0.0
    total: int = 0


@dataclass
class TokenDistribution:
    buckets: Dict[str, int] = field(default_factory=lambda: {b[0]: 0 for b in BUCKET_RANGES})
    percentages: Dict[str, float] = field(default_factory=dict)


@dataclass
class ContextAnalysis:
    model_context_length: int = 4096
    samples_over_limit: int = 0
    pct_over_limit: float = 0.0
    status: str = "FIT"                 # "FIT" | "OVERFLOW"
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class TokenStats:
    dataset_id: str
    model_id: str
    tokenizer_name: str
    total_samples: int = 0
    total_tokens: int = 0
    avg_tokens: float = 0.0
    median_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    total_chars: int = 0
    total_words: int = 0
    input_stats: IOStats = field(default_factory=IOStats)
    output_stats: IOStats = field(default_factory=IOStats)
    distribution: TokenDistribution = field(default_factory=TokenDistribution)
    context_analysis: ContextAnalysis = field(default_factory=ContextAnalysis)
    schema_detected: str = "instruction"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenStats":
        inp_data = data.get("input_stats", {})
        out_data = data.get("output_stats", {})
        dist_data = data.get("distribution", {})
        ctx_data = data.get("context_analysis", {})

        return cls(
            dataset_id=data.get("dataset_id", ""),
            model_id=data.get("model_id", ""),
            tokenizer_name=data.get("tokenizer_name", ""),
            total_samples=data.get("total_samples", 0),
            total_tokens=data.get("total_tokens", 0),
            avg_tokens=data.get("avg_tokens", 0.0),
            median_tokens=data.get("median_tokens", 0.0),
            min_tokens=data.get("min_tokens", 0),
            max_tokens=data.get("max_tokens", 0),
            total_chars=data.get("total_chars", 0),
            total_words=data.get("total_words", 0),
            input_stats=IOStats(**inp_data) if isinstance(inp_data, dict) else IOStats(),
            output_stats=IOStats(**out_data) if isinstance(out_data, dict) else IOStats(),
            distribution=TokenDistribution(**dist_data) if isinstance(dist_data, dict) else TokenDistribution(),
            context_analysis=ContextAnalysis(**ctx_data) if isinstance(ctx_data, dict) else ContextAnalysis(),
            schema_detected=data.get("schema_detected", "instruction"),
        )

    def print_report(self) -> None:
        """Render a clean Rich CLI report matching MYAI design principles."""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()
        console.print("\n[bold cyan]🔍 Tokenizer Analysis[/bold cyan]\n")

        # Dataset & Model info
        table_info = Table(show_header=False, box=None)
        table_info.add_column("Key", style="dim")
        table_info.add_column("Value")
        table_info.add_row("Dataset ID", f"[bold cyan]{self.dataset_id}[/bold cyan]")
        table_info.add_row("Schema", f"[bold]{self.schema_detected}[/bold]")
        table_info.add_row("Total Samples", f"{self.total_samples:,}")
        table_info.add_row("Target Model", f"[bold]{self.model_id}[/bold]")
        table_info.add_row("Tokenizer", self.tokenizer_name)
        console.print(table_info)

        # Token Statistics
        console.print("\n[bold]Token Statistics:[/bold]")
        table_stats = Table(show_header=True, header_style="bold magenta")
        table_stats.add_column("Metric")
        table_stats.add_column("Total Tokens", justify="right")
        table_stats.add_column("Average", justify="right")
        table_stats.add_column("Median", justify="right")
        table_stats.add_column("Min", justify="right")
        table_stats.add_column("Max", justify="right")

        table_stats.add_row(
            "Full Training Text",
            f"{self.total_tokens:,}",
            f"{self.avg_tokens:.1f}",
            f"{self.median_tokens:.1f}",
            f"{self.min_tokens:,}",
            f"{self.max_tokens:,}",
        )
        if self.input_stats.total > 0 or self.output_stats.total > 0:
            table_stats.add_row(
                "Input (Prompt/Context)",
                f"{self.input_stats.total:,}",
                f"{self.input_stats.avg:.1f}",
                f"{self.input_stats.median:.1f}",
                f"{self.input_stats.min:,}",
                f"{self.input_stats.max:,}",
            )
            table_stats.add_row(
                "Output (Response/Answer)",
                f"{self.output_stats.total:,}",
                f"{self.output_stats.avg:.1f}",
                f"{self.output_stats.median:.1f}",
                f"{self.output_stats.min:,}",
                f"{self.output_stats.max:,}",
            )

        console.print(table_stats)

        # Distribution Table
        console.print("\n[bold]Sequence Length Distribution:[/bold]")
        table_dist = Table(show_header=True, header_style="bold cyan")
        table_dist.add_column("Token Range")
        table_dist.add_column("Samples", justify="right")
        table_dist.add_column("Percentage", justify="right")
        table_dist.add_column("Bar", no_wrap=True)

        for label, count in self.distribution.buckets.items():
            pct = self.distribution.percentages.get(label, 0.0)
            bar_len = int(pct / 4.0)
            bar = "█" * bar_len
            table_dist.add_row(label, f"{count:,}", f"{pct:.1f}%", f"[blue]{bar}[/blue]")

        console.print(table_dist)

        # Context Analysis
        console.print("\n[bold]Context Window Analysis:[/bold]")
        ctx = self.context_analysis
        status_style = "[bold green]✓ FIT[/bold green]" if ctx.status == "FIT" else "[bold red]⚠️ OVERFLOW[/bold red]"
        console.print(f"  Model Context Limit : {ctx.model_context_length:,} tokens")
        console.print(f"  Samples Over Limit  : {ctx.samples_over_limit:,} ({ctx.pct_over_limit:.1f}%)")
        console.print(f"  Status              : {status_style}")

        if ctx.status == "OVERFLOW":
            console.print("\n[bold yellow]Recommended Actions:[/bold yellow]")
            for i, action in enumerate(ctx.recommended_actions, 1):
                console.print(f"  {i}. {action}")

        console.print("\n[bold green]✓ Tokenization analysis complete[/bold green]\n")


def compute_token_stats(
    dataset_id: str,
    model_id: str,
    tokenizer_name: str,
    full_token_counts: List[int],
    input_token_counts: List[int],
    output_token_counts: List[int],
    total_chars: int,
    total_words: int,
    schema_detected: str,
    model_context_length: int = 4096,
) -> TokenStats:
    """Computes comprehensive TokenStats dataclass from lists of token counts."""
    total_samples = len(full_token_counts)
    if total_samples == 0:
        return TokenStats(
            dataset_id=dataset_id,
            model_id=model_id,
            tokenizer_name=tokenizer_name,
            schema_detected=schema_detected,
        )

    total_tokens = sum(full_token_counts)
    avg_tokens = round(total_tokens / total_samples, 1)
    med_tokens = round(float(statistics.median(full_token_counts)), 1)
    min_tokens = min(full_token_counts)
    max_tokens = max(full_token_counts)

    # Input stats
    inp_total = sum(input_token_counts)
    inp_avg = round(inp_total / total_samples, 1) if total_samples else 0.0
    inp_med = round(float(statistics.median(input_token_counts)), 1) if input_token_counts else 0.0
    inp_min = min(input_token_counts) if input_token_counts else 0
    inp_max = max(input_token_counts) if input_token_counts else 0
    input_stats = IOStats(min=inp_min, max=inp_max, avg=inp_avg, median=inp_med, total=inp_total)

    # Output stats
    out_total = sum(output_token_counts)
    out_avg = round(out_total / total_samples, 1) if total_samples else 0.0
    out_med = round(float(statistics.median(output_token_counts)), 1) if output_token_counts else 0.0
    out_min = min(output_token_counts) if output_token_counts else 0
    out_max = max(output_token_counts) if output_token_counts else 0
    output_stats = IOStats(min=out_min, max=out_max, avg=out_avg, median=out_med, total=out_total)

    # Distribution buckets
    buckets = {b[0]: 0 for b in BUCKET_RANGES}
    for count in full_token_counts:
        for label, low, high in BUCKET_RANGES:
            if low <= count <= high:
                buckets[label] += 1
                break

    percentages = {label: round((count / total_samples) * 100.0, 1) for label, count in buckets.items()}
    distribution = TokenDistribution(buckets=buckets, percentages=percentages)

    # Context analysis
    samples_over = sum(1 for c in full_token_counts if c > model_context_length)
    pct_over = round((samples_over / total_samples) * 100.0, 1)
    status = "OVERFLOW" if samples_over > 0 else "FIT"
    
    actions = []
    if status == "OVERFLOW":
        actions = [
            f"Truncate inputs to {model_context_length} tokens during training",
            "Chunk long documents into smaller multi-turn conversation steps",
            f"Choose a model with longer native context (e.g. Qwen 2.5 supports up to 32k context)",
        ]

    context_analysis = ContextAnalysis(
        model_context_length=model_context_length,
        samples_over_limit=samples_over,
        pct_over_limit=pct_over,
        status=status,
        recommended_actions=actions,
    )

    return TokenStats(
        dataset_id=dataset_id,
        model_id=model_id,
        tokenizer_name=tokenizer_name,
        total_samples=total_samples,
        total_tokens=total_tokens,
        avg_tokens=avg_tokens,
        median_tokens=med_tokens,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        total_chars=total_chars,
        total_words=total_words,
        input_stats=input_stats,
        output_stats=output_stats,
        distribution=distribution,
        context_analysis=context_analysis,
        schema_detected=schema_detected,
    )
