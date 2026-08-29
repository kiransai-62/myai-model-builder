import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from ..data.loader import load_file

def extract_pairs(data_dir: Path) -> list[dict]:
    pairs = []
    p = Path(data_dir)
    files = [p] if p.is_file() else list(p.rglob("*"))
    for f in files:
        if f.is_file() and f.suffix.lower() in [".json", ".jsonl", ".csv"]:
            for ex in load_file(f):
                prompt = ex.get("prompt") or ex.get("instruction") or ex.get("text", "") or ex.get("input", "")
                response = ex.get("response") or ex.get("output") or ex.get("chosen", "")
                if prompt and response:
                    pairs.append({"prompt": str(prompt), "response": str(response)})
    return pairs

def extract_preference_pairs(data_dir: Path) -> list[dict]:
    """Extracts preference pairs (prompt, chosen, rejected) for DPO/ORPO/SimPO/KTO."""
    pref_pairs = []
    p = Path(data_dir)
    files = [p] if p.is_file() else list(p.rglob("*"))
    for f in files:
        if f.is_file() and f.suffix.lower() in [".json", ".jsonl", ".csv"]:
            for ex in load_file(f):
                prompt = ex.get("prompt") or ex.get("instruction") or ex.get("input", "") or ex.get("question", "")
                chosen = ex.get("chosen") or ex.get("accepted") or ex.get("winner", "")
                rejected = ex.get("rejected") or ex.get("dismissed") or ex.get("loser", "")
                if prompt and chosen and rejected:
                    pref_pairs.append({
                        "prompt": str(prompt),
                        "chosen": str(chosen),
                        "rejected": str(rejected),
                    })
    return pref_pairs

def format_sample(pair: dict) -> str:
    return f"### Instruction:\n{pair['prompt']}\n\n### Response:\n{pair['response']}"

def format_preference_sample(pref: dict) -> dict:
    return {
        "prompt": pref["prompt"],
        "chosen": pref["chosen"],
        "rejected": pref["rejected"],
    }

class TextDataset:
    def __init__(self, pairs, tokenizer, seq_length: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.texts = [format_sample(p) for p in pairs]
        try:
            import torch  # type: ignore
            enc = tokenizer(self.texts, truncation=True, max_length=seq_length, padding="max_length")
            self.input_ids = torch.tensor(enc["input_ids"])
            self.attention_mask = torch.tensor(enc["attention_mask"])
            self.labels = self.input_ids.clone()
            self._has_tensors = True
        except Exception:
            self._has_tensors = False

    def __len__(self):
        if getattr(self, "_has_tensors", False):
            return len(self.input_ids)
        return len(self.pairs)

    def __getitem__(self, i):
        if getattr(self, "_has_tensors", False):
            return {
                "input_ids": self.input_ids[i],
                "attention_mask": self.attention_mask[i],
                "labels": self.labels[i],
            }
        return self.pairs[i]

def create_text_dataset(pairs, tokenizer, seq_length: int = 512):
    return TextDataset(pairs, tokenizer, seq_length)