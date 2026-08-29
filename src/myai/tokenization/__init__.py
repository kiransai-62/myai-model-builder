"""MYAI Tokenization Subsystem.

Provides model-aware dataset tokenization, schema detection, streaming analysis,
and context-length fit checking.
"""
from .tokenizer import TokenizerEngine, get_tokenizer, resolve_model_repo
from .formatter import FormattedSample, format_record, detect_record_schema
from .stats import TokenStats, TokenDistribution, IOStats, ContextAnalysis, compute_token_stats
from .analyzer import analyze_dataset_tokens, stream_records_from_path, discover_data_files
from .cache import TokenizationCache, calculate_source_fingerprint

__all__ = [
    "TokenizerEngine",
    "get_tokenizer",
    "resolve_model_repo",
    "FormattedSample",
    "format_record",
    "detect_record_schema",
    "TokenStats",
    "TokenDistribution",
    "IOStats",
    "ContextAnalysis",
    "compute_token_stats",
    "analyze_dataset_tokens",
    "stream_records_from_path",
    "discover_data_files",
    "TokenizationCache",
    "calculate_source_fingerprint",
]
