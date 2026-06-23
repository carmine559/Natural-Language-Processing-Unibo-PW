"""
utils/json_parser.py — extract + validate JSON from raw LLM output.

LLMs often wrap JSON in markdown fences, add preambles, or produce
slightly invalid JSON. This module handles all those cases and feeds
the validation error back to the caller so they can retry with context.
"""
from __future__ import annotations

import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def extract_json_from_text(text: str) -> dict:
    """
    Try increasingly permissive extraction strategies until one works.
    Raises ValueError if nothing produces valid JSON.
    """
    text = text.strip()

    # 1. Direct parse — model returned clean JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fence ```json ... ```  or  ``` ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Find the first { ... } block (greedy — takes the outermost)
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            # 4. Sometimes trailing commas or single quotes break things
            candidate = brace.group(0)
            # Remove trailing commas before } or ]
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            # Replace single-quoted strings (crude but handles simple cases)
            candidate = re.sub(r"'([^']*)'", r'"\1"', candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    raise ValueError(
        f"Could not extract valid JSON from model output. "
        f"First 300 chars: {text[:300]!r}"
    )


def parse_with_schema(
    text: str,
    schema: Type[T],
    max_retries: int = 1,          # retries are done at the caller level
) -> tuple[T, None] | tuple[None, str]:
    """
    Parse text into a Pydantic model.
    Returns (parsed_object, None) on success, (None, error_message) on failure.
    The error_message can be fed back to the LLM in a retry prompt.
    """
    try:
        raw = extract_json_from_text(text)
    except ValueError as e:
        return None, f"JSON extraction failed: {e}"

    try:
        return schema(**raw), None
    except ValidationError as e:
        # Provide a concise summary of what's wrong
        errors = "; ".join(
            f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
            for err in e.errors()
        )
        return None, f"Schema validation failed: {errors}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def build_retry_suffix(error_msg: str, schema: Type[T]) -> str:
    """
    Build a suffix to append to a prompt when asking the model to fix its output.
    """
    return (
        f"\n\nIl tuo output precedente aveva questo errore: {error_msg}\n"
        f"Riprova. Restituisci SOLO JSON valido, nessun testo aggiuntivo."
    )
