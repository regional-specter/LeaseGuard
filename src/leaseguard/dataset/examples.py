"""Load Dataset v1 example files from a directory of JSON records."""

import json
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from leaseguard.dataset.models import (
    DatasetSubset,
    EvidenceConversationRecord,
    MultiTurnRecord,
    PreferencePairRecord,
    RawDomainRecord,
    RecordBase,
    RefusalRecord,
    StructuredExtractionRecord,
)

RECORD_MODELS: dict[str, type[BaseModel]] = {
    "raw_domain_text": RawDomainRecord,
    "structured_extraction": StructuredExtractionRecord,
    "evidence_conversation": EvidenceConversationRecord,
    "refusal_uncertainty": RefusalRecord,
    "multi_turn": MultiTurnRecord,
    "preference_pairs": PreferencePairRecord,
}


class DatasetExampleError(ValueError):
    """Raised when an example file cannot be loaded."""


def load_example_payload(payload: dict[str, Any]) -> RecordBase:
    """Validate one tagged example object."""
    subset = payload.get("subset")
    if subset not in RECORD_MODELS:
        raise DatasetExampleError(f"unknown or missing subset: {subset!r}")
    return cast(RecordBase, RECORD_MODELS[subset].model_validate(payload))


def load_examples(root: Path) -> list[RecordBase]:
    """Read every JSON example under a directory, in path order."""
    if not root.is_dir():
        raise DatasetExampleError(f"example directory does not exist: {root}")
    records: list[RecordBase] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetExampleError(f"{path}: {error}") from error
        if not isinstance(payload, dict):
            raise DatasetExampleError(f"{path}: example files must contain one JSON object")
        try:
            records.append(load_example_payload(payload))
        except (DatasetExampleError, ValueError) as error:
            raise DatasetExampleError(f"{path}: {error}") from error
    if not records:
        raise DatasetExampleError(f"no JSON examples found under {root}")
    return records


def subset_records(records: list[RecordBase], subset: DatasetSubset) -> list[RecordBase]:
    """Return records belonging to one Dataset v1 group."""
    return [record for record in records if record.subset == subset]
