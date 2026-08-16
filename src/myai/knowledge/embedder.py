import hashlib
import math

class Embedder:
    """Uses sentence-transformers if available, else a deterministic hash fallback."""
    def __init__(self):
        self.dim = 384
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is not None:
            return [v.tolist() for v in self._model.encode(texts)]
        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]