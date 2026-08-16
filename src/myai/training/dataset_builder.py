import importlib
from pathlib import Path
from ..data.loader import load_file

def extract_pairs(data_dir: Path) -> list[dict]:
    pairs = []
    p = Path(data_dir)
    files = [p] if p.is_file() else list(p.rglob("*"))
    for f in files:
        if f.is_file() and f.suffix.lower() in [".json", ".jsonl", ".csv"]:
            for ex in load_file(f):
                prompt = ex.get("prompt") or ex.get("instruction") or ex.get("text", "")
                response = ex.get("response") or ex.get("output") or ""
                if prompt and response:
                    pairs.append({"prompt": str(prompt), "response": str(response)})
    return pairs

def format_sample(pair: dict) -> str:
    return f"### Instruction:\n{pair['prompt']}\n\n### Response:\n{pair['response']}"

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