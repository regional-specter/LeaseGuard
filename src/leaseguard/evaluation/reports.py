"""Write small machine-readable evaluation reports."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def write_json_report(path: Path, model: BaseModel) -> Path:
    """Save a Pydantic evaluation record as stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True)
    path.write_text(f"{payload}\n", encoding="utf-8")
    return path


def write_json_payload(path: Path, payload: dict[str, Any]) -> Path:
    """Save a small JSON object that is not a public schema model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(f"{content}\n", encoding="utf-8")
    return path
