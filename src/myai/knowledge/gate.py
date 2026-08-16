import json
import math
from pathlib import Path

class KnowledgeGate:
    def __init__(self, root: Path, cfg):
        self.threshold = cfg.gate.threshold
        self.top_k = cfg.gate.top_k
        self.chunks = []
        index_file = root / "indexes" / "chunks.jsonl"
        if index_file.exists():
            for line in index_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.chunks.append(json.loads(line))

    @staticmethod
    def _cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (na * nb)

    def decide(self, query: str, embedder):
        if not self.chunks:
            return False, 0.0, []
        qvec = embedder.embed([query])[0]
        scored = sorted(
            self.chunks,
            key=lambda c: self._cosine(qvec, c["embedding"]),
            reverse=True,
        )
        top = scored[: self.top_k]
        best = self._cosine(qvec, top[0]["embedding"]) if top else 0.0
        return best >= self.threshold, best, top