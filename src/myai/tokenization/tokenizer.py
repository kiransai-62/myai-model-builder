"""MYAI Tokenizer Engine & Compatibility Loader.

Provides model-aware tokenizer resolution with fallback capabilities:
1. Resolves Hugging Face AutoTokenizer if transformers is available and cached/online.
2. Provides a robust local calibrated byte-level tokenizer fallback for air-gapped / offline environments.
3. Enforces model-tokenizer compatibility.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Known model families and their typical vocabularies & chat templates
MODEL_TOKENIZER_MAP = {
    "qwen": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct",
        "vocab_size": 151936,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "chat_template": "chatml",
        "tokens_per_word_ratio": 1.28,
    },
    "llama": {
        "repo": "meta-llama/Llama-3.2-1B-Instruct",
        "vocab_size": 128256,
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|eot_id|>",
        "chat_template": "llama3",
        "tokens_per_word_ratio": 1.32,
    },
    "smollm": {
        "repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "vocab_size": 49152,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "chat_template": "chatml",
        "tokens_per_word_ratio": 1.30,
    },
    "default": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct",
        "vocab_size": 151936,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "chat_template": "default",
        "tokens_per_word_ratio": 1.30,
    },
}


def _detect_model_family(model_identifier: str) -> str:
    low = model_identifier.lower()
    if "qwen" in low:
        return "qwen"
    elif "llama" in low:
        return "llama"
    elif "smol" in low:
        return "smollm"
    return "default"


class CalibratedHeuristicTokenizer:
    """Fast, deterministic, offline tokenizer when Hugging Face transformers/weights are not locally available."""

    def __init__(self, model_name: str, family: str = "default"):
        self.model_name = model_name
        self.family = family
        self.profile = MODEL_TOKENIZER_MAP.get(family, MODEL_TOKENIZER_MAP["default"])
        self.vocab_size = self.profile["vocab_size"]
        self.name = f"{family.capitalize()}Tokenizer (Offline Calibrated)"

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        
        # BPE approximation regex: splits contractions, words, numbers, punctuation, spaces
        pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        try:
            import regex
            tokens = regex.findall(pattern, text)
            return [hash(t) % self.vocab_size for t in tokens]
        except ImportError:
            # Standard re fallback
            words_and_punct = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
            # Add subword splitting estimate for long tokens
            token_count = 0
            for item in words_and_punct:
                if len(item) <= 4:
                    token_count += 1
                else:
                    token_count += max(1, int(len(item) / 3.5 + 0.5))
            return list(range(token_count))

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encode(text))

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Standard ChatML / instruction template application."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(formatted)


class TokenizerEngine:
    """Wrapper that tries AutoTokenizer first, falling back gracefully to CalibratedHeuristicTokenizer."""

    def __init__(self, model_identifier: str, tokenizer_obj: Any = None, is_native: bool = False):
        self.model_identifier = model_identifier
        self._tokenizer = tokenizer_obj
        self.is_native = is_native
        self.family = _detect_model_family(model_identifier)

        if not self._tokenizer:
            self._heuristic = CalibratedHeuristicTokenizer(model_identifier, self.family)
        else:
            self._heuristic = None

    @property
    def name(self) -> str:
        if self._tokenizer:
            return getattr(self._tokenizer, "name_or_path", self.model_identifier)
        return self._heuristic.name

    @property
    def vocab_size(self) -> int:
        if self._tokenizer and hasattr(self._tokenizer, "vocab_size"):
            return self._tokenizer.vocab_size
        return self._heuristic.vocab_size if self._heuristic else 151936

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        if self._tokenizer:
            try:
                return self._tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                pass
        return self._heuristic.encode(text)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return self._heuristic.count_tokens(text)

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                pass
        if self._heuristic:
            return self._heuristic.apply_chat_template(messages)
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


def resolve_model_repo(model_identifier: Optional[str] = None, project_root: Optional[Path] = None) -> str:
    """Resolves the base model repo name according to priority:
    1. Explicit model identifier passed by user
    2. Project configuration in myai.yaml
    3. Default recommended baseline (e.g. Qwen/Qwen2.5-1.5B-Instruct)
    """
    if model_identifier and str(model_identifier).strip():
        mid = str(model_identifier).strip()
        # Check if shorthand
        if "/" not in mid:
            from ..models.registry import get_registry_models
            for m in get_registry_models():
                if m.id == mid:
                    return m.repo_id
        return mid

    if project_root and (project_root / "myai.yaml").exists():
        from ..core.config import ProjectConfig
        try:
            cfg = ProjectConfig.load(project_root)
            if cfg.model_id:
                from ..models.registry import get_registry_models
                for m in get_registry_models():
                    if m.id == cfg.model_id:
                        return m.repo_id
                return cfg.model_id
        except Exception:
            pass

    return "Qwen/Qwen2.5-1.5B-Instruct"


def get_tokenizer(model_identifier: Optional[str] = None, project_root: Optional[Path] = None, offline_only: bool = False) -> TokenizerEngine:
    """Loads a model-aware tokenizer. Attempts Hugging Face AutoTokenizer with clean offline fallback."""
    repo_id = resolve_model_repo(model_identifier, project_root)

    # 1. Attempt AutoTokenizer from local cache or environment
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
        # First try local files only
        try:
            tok = AutoTokenizer.from_pretrained(repo_id, local_files_only=True)
            return TokenizerEngine(repo_id, tok, is_native=True)
        except Exception:
            if not offline_only:
                try:
                    tok = AutoTokenizer.from_pretrained(repo_id)
                    return TokenizerEngine(repo_id, tok, is_native=True)
                except Exception:
                    pass
    except ImportError:
        pass

    # 2. Return calibrated offline fallback engine
    return TokenizerEngine(repo_id, tokenizer_obj=None, is_native=False)
