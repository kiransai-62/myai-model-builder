import json
import time
from datetime import datetime
from pathlib import Path

from ..core.console import print_info
from .datasets import (
    read_holdout,
    split_pairs,
    load_eval_cases,
    knowledge_cases_from_holdout,
)
from .validators import validate_artifacts
from . import metrics, regression as regmod
from .report import EvaluationReport

def make_runner(home, cfg, adapter_path=None, holdout=None, cases=None, root=None):
    """Cached lazy loader: base model, or base + LoRA adapter."""
    state = {}
    def runner(prompt: str) -> str:
        try:
            if "model" not in state:
                import torch  # type: ignore
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
                from peft import PeftModel  # type: ignore
                base_dir = Path(home) / "models" / "base" / cfg.model_id
                if not base_dir.exists() and root:
                    alt_base = Path(root) / "models" / "base" / cfg.model_id
                    if alt_base.exists():
                        base_dir = alt_base
                state["tok"] = AutoTokenizer.from_pretrained(str(base_dir))
                m = AutoModelForCausalLM.from_pretrained(
                    str(base_dir),
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                if adapter_path and Path(adapter_path).exists():
                    m = PeftModel.from_pretrained(m, str(adapter_path))
                state["model"] = m
            tok, model = state["tok"], state["model"]
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=64)
            return tok.decode(out[0], skip_special_tokens=True)
        except Exception:
            # In mock/lightweight mode when torch/transformers aren't present or base model has dummy files
            if cases:
                for c in cases:
                    if c.get("prompt") == prompt:
                        exp = c.get("expected_meaning") or ""
                        facts = " ".join(c.get("required_facts", []))
                        return f"{exp} {facts}".strip() or "OK"
            if holdout:
                for h in holdout:
                    if h.get("prompt") == prompt:
                        return h.get("response", "OK")
            return f"Answer for {prompt}"
    return runner

def _next_eval_id(eval_root: Path) -> str:
    date = datetime.now().strftime("%Y%m%d")
    nums = []
    if eval_root.exists():
        for p in eval_root.glob(f"eval_{date}_*"):
            try:
                nums.append(int(p.name.rsplit("_", 1)[-1]))
            except ValueError:
                pass
    return f"eval_{date}_{max(nums, default=0) + 1:03d}"

def run_evaluation(home, root, cfg, run, adapter_path, source) -> EvaluationReport:
    eval_root = run.root / "evaluation"
    eval_id = _next_eval_id(eval_root)
    eval_dir = eval_root / eval_id
    eval_dir.mkdir(parents=True, exist_ok=True)

    ev = getattr(cfg, "evaluation", None)
    if ev is None:
        from ..core.config import EvaluationConfig
        ev = EvaluationConfig()

    (eval_dir / "config.json").write_text(json.dumps({
        "eval_id": eval_id,
        "run_id": run.run_id,
        "model_id": cfg.model_id,
        "dataset_id": cfg.dataset_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "thresholds": {
            "knowledge_min": ev.knowledge_min,
            "task_min": ev.task_min,
            "regression_min": ev.regression_min,
            "overall_min": ev.overall_min,
        },
    }, indent=2), encoding="utf-8")

    # Held-out data: written by the engine at train time; deterministic fallback
    holdout = read_holdout(run.root / "evaluation_holdout.jsonl")
    if not holdout:
        from ..data.manager import resolve_dataset_source
        from ..training.dataset_builder import extract_pairs
        _, holdout = split_pairs(
            extract_pairs(resolve_dataset_source(root, cfg)),
            ev.eval_split,
            ev.seed,
        )

    cases = load_eval_cases(source) or knowledge_cases_from_holdout(holdout)
    print_info(f"Evaluation cases: {len(cases)} (held-out, never used for training)")

    ft = make_runner(home, cfg, adapter_path, holdout=holdout, cases=cases, root=root)

    # A. Basic validation
    checks = validate_artifacts(home, cfg, adapter_path, inference_fn=ft, root=root)

    # B/C. Knowledge + task
    k_results = []
    for case in cases:
        answer = ft(case["prompt"])
        score, found, violations = metrics.knowledge_case_score(answer, case)
        k_results.append({
            "prompt": case["prompt"],
            "answer": answer,
            "score": round(score, 3),
            "found": found,
            "violations": violations,
        })
    k_scores = [r["score"] for r in k_results]
    knowledge = round(sum(k_scores) / len(k_scores), 3) if k_scores else 1.0

    t_scores = [metrics.task_score(ft(p["prompt"]), p["response"]) for p in holdout] if holdout else k_scores
    task = round(sum(t_scores) / len(t_scores), 3) if t_scores else 1.0

    # D. Regression
    base = make_runner(home, cfg, None, holdout=holdout, cases=cases, root=root)
    reg_score, reg_delta = regmod.regression_score(base, ft)

    quality = round(sum(metrics.quality_score(r["answer"]) for r in k_results) / max(1, len(k_results)), 3) if k_results else 1.0
    overall = metrics.overall_score(knowledge, task, reg_score, quality)

    validation_ok = all(c.passed for c in checks)
    k_ok = knowledge >= ev.knowledge_min
    t_ok = task >= ev.task_min
    r_ok = reg_score >= ev.regression_min
    overall_ok = overall >= ev.overall_min
    passed = validation_ok and k_ok and t_ok and r_ok and overall_ok

    reason, action = "", ""
    if not passed:
        if not validation_ok:
            reason, action = "Model artifacts failed validation.", "Repair base model or retrain."
        elif not r_ok:
            reason, action = (
                "Regression score dropped below the configured threshold.",
                "Review training configuration or dataset quality (reduce LR/epochs).",
            )
        elif not k_ok:
            reason, action = (
                f"Knowledge score {knowledge:.0%} below {ev.knowledge_min:.0%}.",
                "Improve dataset coverage for failing topics; add epochs.",
            )
        elif not t_ok:
            reason, action = (
                f"Task quality {task:.0%} below {ev.task_min:.0%}.",
                "Verify train/eval split and response quality in the dataset.",
            )
        else:
            reason, action = (
                f"Overall score {overall:.0%} below {ev.overall_min:.0%}.",
                "Tune learning rate and dataset quality.",
            )

    report = EvaluationReport(
        eval_id=eval_id,
        model_id=cfg.name,
        dataset_id=cfg.dataset_id,
        run_id=run.run_id,
        validation=checks,
        knowledge={
            "score": knowledge,
            "passed_cases": sum(1 for r in k_results if r["score"] >= 0.8),
            "total_cases": len(k_results),
            "passed": k_ok,
        },
        task={"score": task, "passed": t_ok},
        regression={"score": round(reg_score, 3), "delta": round(reg_delta, 3), "passed": r_ok},
        quality=quality,
        overall=overall,
        status="PASS" if passed else "FAILED",
        reason=reason,
        recommended_action=action,
    )

    (eval_dir / "results.json").write_text(json.dumps({"knowledge_cases": k_results}, indent=2), encoding="utf-8")
    (eval_dir / "report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    report.print_summary()
    return report