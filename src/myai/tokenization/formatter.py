"""MYAI Dataset Formatter & Schema Detector.

Identifies training dataset schemas and formats records into the exact
training representations expected by base models and tokenizers:
- Instruction: instruction + input + output
- Prompt/Response: prompt + response / question + answer
- Chat / Conversational: messages: [{role, content}]
- Raw text: text / document content
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FormattedSample:
    input_text: str
    output_text: str
    full_text: str
    schema: str
    char_count: int
    word_count: int


def detect_record_schema(record: Dict[str, Any]) -> str:
    """Detect the schema of a single record dictionary."""
    if not isinstance(record, dict):
        return "text"

    keys = {k.lower().strip() for k in record.keys()}

    if "messages" in keys or "conversations" in keys:
        return "chat"
    elif "instruction" in keys and ("output" in keys or "response" in keys):
        return "instruction"
    elif "prompt" in keys and ("response" in keys or "completion" in keys):
        return "prompt_response"
    elif "question" in keys and ("answer" in keys or "response" in keys):
        return "prompt_response"
    elif "input" in keys and ("output" in keys or "target" in keys):
        return "instruction"
    elif "text" in keys or "content" in keys or "document" in keys:
        return "text"
    
    # Fallback to column matching for CSVs / dicts with prompt-like keys
    if any(k in keys for k in ["prompt", "question", "instruction", "input"]):
        return "prompt_response"

    return "generic"


def format_record(record: Dict[str, Any], tokenizer: Optional[Any] = None) -> FormattedSample:
    """Extract input, output, and complete training representation from a raw record."""
    if not isinstance(record, dict):
        text = str(record)
        return FormattedSample(
            input_text=text,
            output_text="",
            full_text=text,
            schema="text",
            char_count=len(text),
            word_count=len(text.split()),
        )

    schema = detect_record_schema(record)

    input_text = ""
    output_text = ""
    full_text = ""

    if schema == "chat":
        messages = record.get("messages") or record.get("conversations") or []
        if isinstance(messages, list):
            # Extract input (all user/system turns) and output (assistant turns)
            user_parts = [m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") in ("user", "system")]
            asst_parts = [m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
            input_text = "\n".join(user_parts)
            output_text = "\n".join(asst_parts)

            if tokenizer and hasattr(tokenizer, "apply_chat_template"):
                try:
                    full_text = tokenizer.apply_chat_template(messages)
                except Exception:
                    full_text = "\n".join(f"<|im_start|>{m.get('role', 'user')}\n{m.get('content', '')}<|im_end|>" for m in messages if isinstance(m, dict))
            else:
                full_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages if isinstance(m, dict))
        else:
            full_text = str(messages)

    elif schema == "instruction":
        inst = str(record.get("instruction") or "")
        inp = str(record.get("input") or "")
        out = str(record.get("output") or record.get("response") or "")

        if inp.strip():
            input_text = f"Below is an instruction that describes a task, paired with an input that provides further context.\n\n### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
        else:
            input_text = f"Below is an instruction that describes a task.\n\n### Instruction:\n{inst}\n\n### Response:\n"
        
        output_text = out
        full_text = input_text + output_text

    elif schema == "prompt_response":
        p = str(record.get("prompt") or record.get("question") or record.get("input") or "")
        r = str(record.get("response") or record.get("completion") or record.get("answer") or record.get("output") or "")
        
        input_text = p
        output_text = r
        full_text = f"{p}\n{r}" if p and r else (p or r)

    else:
        # Text / Generic
        val = record.get("text") or record.get("content") or record.get("document") or ""
        if not val:
            # Join all string values
            val = " ".join(str(v) for v in record.values() if isinstance(v, (str, int, float)))
        input_text = str(val)
        output_text = ""
        full_text = str(val)

    return FormattedSample(
        input_text=input_text,
        output_text=output_text,
        full_text=full_text,
        schema=schema,
        char_count=len(full_text),
        word_count=len(full_text.split()),
    )
