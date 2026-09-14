"""Extract a JSON object from unconstrained model text."""

from __future__ import annotations

import json
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first JSON object in fenced or raw model output."""
    stripped = text.strip()
    if not stripped:
        return None
    fenced = _from_fence(stripped)
    if fenced is not None:
        return fenced
    return _from_braces(stripped)


def _from_fence(text: str) -> dict[str, Any] | None:
    start = text.find("```")
    if start < 0:
        return None
    rest = text[start + 3 :]
    if rest.lower().startswith("json"):
        rest = rest[4:]
    end = rest.find("```")
    candidate = rest.strip() if end < 0 else rest[:end].strip()
    return _loads_object(candidate)


def _from_braces(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    return _loads_object(text[start : end + 1])


def _loads_object(payload: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed
