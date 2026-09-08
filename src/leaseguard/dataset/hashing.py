"""Stable SHA-256 helpers for Dataset v1 artifacts."""

import hashlib
import json
from typing import Any


def sha256_json(value: Any) -> str:
    """Hash a JSON-serializable value with sorted keys."""
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
