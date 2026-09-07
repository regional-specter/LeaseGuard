"""Tests that keep committed JSON Schema artifacts reproducible."""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from leaseguard.ontology import DocumentAnswer, LeaseExtraction

SCHEMA_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "schemas"
SAMPLE_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "samples"


@pytest.mark.parametrize("model", [LeaseExtraction, DocumentAnswer])
def test_committed_json_schema_matches_model(model: type[BaseModel]) -> None:
    """Generated JSON Schema must stay synchronized with its source model."""
    schema_path = SCHEMA_DIRECTORY / f"{model.__name__}.schema.json"
    committed_schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert committed_schema == model.model_json_schema()


@pytest.mark.parametrize(
    ("filename", "model"),
    [
        ("lease_extraction.v1.json", LeaseExtraction),
        ("document_answer.v1.json", DocumentAnswer),
    ],
)
def test_sample_validates_against_public_model(
    filename: str,
    model: type[BaseModel],
) -> None:
    """Committed examples must remain valid as schemas evolve."""
    sample = (SAMPLE_DIRECTORY / filename).read_text(encoding="utf-8")

    model.model_validate_json(sample)
