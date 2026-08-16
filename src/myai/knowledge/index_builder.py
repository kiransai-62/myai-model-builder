import json
from pathlib import Path
from ..data.manager import resolve_dataset_source
from ..data.loader import load_file
from .chunker import chunk_text
from .embedder import Embedder

def build_index(root: Path, cfg) -> int:
    data_dir = resolve_dataset_source(root, cfg)
    texts = []

    files = [data_dir] if data_dir.is_file() else list(data_dir.rglob("*"))
    for f in files:
        if f.is_file() and f.suffix.lower() in [".json", ".jsonl", ".csv"]:
            for ex in load_file(f):
                text = str(ex.get("prompt", "") + " " + ex.get("response", ex.get("text", "")))
                texts.extend(chunk_text(text))

    if not texts:
        return 0

    embedder = Embedder()
    vectors = embedder.embed(texts)

    index_dir = root / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)

    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as out:
        for i, (text, vec) in enumerate(zip(texts, vectors)):
            out.write(json.dumps({"id": i, "text": text, "embedding": vec}) + "\n")

    return len(texts)