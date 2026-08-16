import json
import time
import threading
from pathlib import Path
from dataclasses import dataclass

try:
    import torch  # type: ignore[import-not-found]
except ImportError:
    torch = None

from ..core.console import console, print_info, print_error
from ..core.config import ProjectConfig
from ..core.home import ensure_home
from ..knowledge.gate import KnowledgeGate
from ..knowledge.embedder import Embedder
from ..models.trained_registry import resolve_adapter

@dataclass
class InferenceRequest:
    query: str
    temperature: float = 0.7
    max_tokens: int = 256

@dataclass
class InferenceResponse:
    allowed: bool
    score: float
    answer: str
    sources: list[str]
    latency_ms: float

class MyAIRuntime:
    """Loads and serves the trained model with Knowledge Gate."""
    
    def __init__(self, root: Path):
        self.root = root
        self.cfg = ProjectConfig.load(root)
        self.model = None
        self.tokenizer = None
        self.gate = None
        self.embedder = None
        self._load_time = None

    def load(self):
        """Load base model + adapter + index into memory."""
        start = time.time()
        
        home = ensure_home()
        adapter_path = resolve_adapter(home, self.cfg.name, root=self.root)
        base_dir = self.root / "models" / "base" / self.cfg.model_id
        if not base_dir.exists():
            base_dir = home / "models" / "base" / self.cfg.model_id
        
        if not adapter_path or not adapter_path.exists():
            print_error(f"No trained adapter found for {self.cfg.name}")
            raise RuntimeError("Run `myai train` first")
        
        if not (self.root / "indexes" / "chunks.jsonl").exists():
            print_error("No knowledge index found")
            raise RuntimeError("Run `myai index build` first")

        print_info("Loading Knowledge Gate...")
        self.embedder = Embedder()
        self.gate = KnowledgeGate(self.root, self.cfg)
        print_info(f"  Loaded {len(self.gate.chunks)} knowledge chunks")

        print_info("Loading model (this may take 30-60 seconds)...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
            from peft import PeftModel  # type: ignore[import-not-found]

            self.tokenizer = AutoTokenizer.from_pretrained(str(base_dir))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_kwargs = {}
            if torch is not None:
                model_kwargs = {
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                    "device_map": "auto" if torch.cuda.is_available() else None
                }

            base_model = AutoModelForCausalLM.from_pretrained(str(base_dir), **model_kwargs)
            self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
            self.model.eval()
            
            print_info(f"  Model loaded on {self.model.device}")
            
        except Exception as e:
            print_error(f"Failed to load model: {e}")
            print_info("Falling back to retrieval-only mode (no generation)")
            self.model = None

        self._load_time = time.time() - start
        console.print(f"[bold green]✓ Runtime ready in {self._load_time:.1f}s[/bold green]\n")

    def ask(self, request: InferenceRequest | str) -> InferenceResponse:
        """Process a query through Knowledge Gate + model."""
        start = time.time()

        if isinstance(request, str):
            request = InferenceRequest(query=request)

        # Step 1: Knowledge Gate check
        allowed, score, top_chunks = self.gate.decide(request.query, self.embedder)
        
        if not allowed:
            return InferenceResponse(
                allowed=False,
                score=score,
                answer="I can only answer questions related to the project knowledge base.",
                sources=[],
                latency_ms=(time.time() - start) * 1000
            )

        # Step 2: Generate answer (if model loaded)
        if self.model is None or self.tokenizer is None or torch is None:
            # Fallback: return retrieved chunks
            context = "\n".join(c["text"] for c in top_chunks)
            return InferenceResponse(
                allowed=True,
                score=score,
                answer=f"[Retrieved knowledge]\n{context}",
                sources=[str(c.get("id")) for c in top_chunks],
                latency_ms=(time.time() - start) * 1000
            )

        # Step 3: Build grounded prompt
        context = "\n".join(c["text"] for c in top_chunks)
        prompt = f"""### Context:
{context}

### Instruction:
{request.query}

### Response:
"""
        
        # Step 4: Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Step 5: Decode
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_text.split("### Response:")[-1].strip()
        
        latency_ms = (time.time() - start) * 1000
        
        return InferenceResponse(
            allowed=True,
            score=score,
            answer=answer,
            sources=[str(c.get("id")) for c in top_chunks],
            latency_ms=latency_ms
        )

    def stream_answer(self, request: InferenceRequest | str):
        """Generator that yields tokens as Server-Sent Events (SSE)."""
        if isinstance(request, str):
            request = InferenceRequest(query=request)

        # Check gate
        allowed, score, top_chunks = self.gate.decide(request.query, self.embedder)
        if not allowed:
            yield "data: I can only answer questions related to the project knowledge base.\n\n"
            yield "data: [DONE]\n\n"
            return

        if self.model is None or self.tokenizer is None or torch is None:
            yield "data: [Model not loaded, falling back to retrieval]\n\n"
            for chunk in top_chunks:
                for word in chunk["text"].split():
                    yield f"data: {word} \n\n"
                    time.sleep(0.02)
                yield "data: \n\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            from transformers import TextIteratorStreamer  # type: ignore[import-not-found]
            context = "\n".join(c["text"] for c in top_chunks)
            prompt = f"### Context:\n{context}\n\n### Instruction:\n{request.query}\n\n### Response:\n"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
            )

            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            for new_text in streamer:
                if new_text:
                    clean_text = new_text.replace("\n", "\\n")
                    yield f"data: {clean_text}\n\n"

            thread.join()
            yield "data: [DONE]\n\n"
        except Exception:
            yield "data: [Fallback retrieval]\n\n"
            for chunk in top_chunks:
                yield f"data: {chunk['text']}\n\n"
            yield "data: [DONE]\n\n"