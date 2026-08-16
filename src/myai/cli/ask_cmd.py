import typer
from pathlib import Path
from ..core.paths import require_project_root
from ..core.config import ProjectConfig
from ..core.console import console
from ..core.home import ensure_home
from ..knowledge.gate import KnowledgeGate
from ..knowledge.embedder import Embedder
from ..models.trained_registry import resolve_adapter

def ask(query: str):
    root = require_project_root()
    cfg = ProjectConfig.load(root)
    home = ensure_home()

    embedder = Embedder()
    gate = KnowledgeGate(root, cfg)
    allowed, score, top = gate.decide(query, embedder)

    console.print(f"\n[bold]Query:[/bold] {query}")
    console.print(f"Gate score: {score:.3f} (threshold {cfg.gate.threshold})")

    if not allowed:
        console.print("Status: [bold red]REFUSED[/bold red] — outside knowledge boundary.\n")
        console.print("[yellow]I can only answer from the project knowledge base.[/yellow]\n")
        return

    console.print("Status: [bold green]ALLOWED[/bold green] — grounded in knowledge base.")
    console.print(f"Model:  [cyan]{cfg.name}[/cyan]")

    adapter_path = resolve_adapter(home, cfg.name, root=root)
    if adapter_path and adapter_path.exists():
        console.print("Adapter: [green]LOADED ✓[/green]")
    else:
        console.print("Adapter: [yellow]NOT FOUND[/yellow] [dim](Run `myai train` & `myai evaluate` first)[/dim]")

    # Resolve base model directory
    base_dir = root / "models" / "base" / cfg.model_id
    if not base_dir.exists():
        base_dir = home / "models" / "base" / cfg.model_id

    # Check for PyTorch / Transformers / PEFT execution
    model_loaded = False
    if adapter_path and adapter_path.exists() and base_dir.exists():
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
            from peft import PeftModel  # type: ignore[import-not-found]

            console.print("\n[cyan]Generating with trained model...[/cyan]")
            tokenizer = AutoTokenizer.from_pretrained(str(base_dir))
            model_kwargs = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto" if torch.cuda.is_available() else None,
            }
            model = AutoModelForCausalLM.from_pretrained(str(base_dir), **model_kwargs)
            model = PeftModel.from_pretrained(model, str(adapter_path))

            context = "\n".join(c["text"] for c in top)
            prompt = f"### Context:\n{context}\n\n### Instruction:\n{query}\n\n### Response:\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
            answer = tokenizer.decode(out[0], skip_special_tokens=True).split("### Response:")[-1].strip()
            console.print(f"\n[bold green]{answer}[/bold green]\n")
            model_loaded = True
            return
        except Exception:
            model_loaded = False

    # Grounded answer synthesis (Lightweight / Fallback mode)
    if not model_loaded:
        console.print("\n[dim]Execution engine: Lightweight Knowledge-Grounded Fallback[/dim]")
        context_texts = [c["text"].strip() for c in top if c.get("text")]
        if context_texts:
            clean_q = query.lower().strip("?.! ")
            candidates = [t for t in context_texts if t.lower().strip("?.! ") != clean_q]
            primary_text = candidates[0] if candidates else context_texts[0]
            console.print(f"\n[bold green]{primary_text}[/bold green]\n")
            other_contexts = [t for t in context_texts if t != primary_text]
            if other_contexts:
                console.print("[dim]Supporting context:[/dim]")
                for c_text in other_contexts[:2]:
                    console.print(f"  [dim]• {c_text[:140]}[/dim]")
            console.print("")
        else:
            console.print("[yellow]No relevant context found in index.[/yellow]\n")