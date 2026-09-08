"""Tests that keep committed JSON Schema artifacts reproducible."""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from leaseguard.dataset import DatasetBundle, DatasetConfig, DatasetReport
from leaseguard.evaluation import (
    BenchmarkRegistry,
    BenchmarkRun,
    RegressionReport,
    RegressionSuiteConfig,
)
from leaseguard.ingestion import BatchReport, PipelineConfig, SourceManifest
from leaseguard.ontology import SCHEMA_VERSION, DocumentAnswer, LeaseExtraction
from leaseguard.preprocessing import ProcessedDocument

SCHEMA_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "schemas"
SAMPLE_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "samples"
VALIDATION_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "validation"


@pytest.mark.parametrize(
    "model",
    [
        LeaseExtraction,
        DocumentAnswer,
        BenchmarkRegistry,
        BenchmarkRun,
        RegressionReport,
        RegressionSuiteConfig,
        SourceManifest,
        PipelineConfig,
        BatchReport,
        ProcessedDocument,
        DatasetConfig,
        DatasetBundle,
        DatasetReport,
    ],
)
def test_committed_json_schema_matches_model(model: type[BaseModel]) -> None:
    """Generated JSON Schema must stay synchronized with its source model."""
    schema_path = SCHEMA_DIRECTORY / f"{model.__name__}.schema.json"
    committed_schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert committed_schema == model.model_json_schema()


def test_frozen_ontology_version_matches_models() -> None:
    """The release marker and Python models must use the same version."""
    version_path = SCHEMA_DIRECTORY / "ontology.version.json"
    version_record = json.loads(version_path.read_text(encoding="utf-8"))

    assert version_record["version"] == SCHEMA_VERSION
    assert version_record["status"] == "frozen"


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


@pytest.mark.parametrize(
    "filename",
    [
        "gsa_office_template.extraction.json",
        "gsa_amendment_template.extraction.json",
        "pandadoc_retail_template.extraction.json",
        "st_joseph_retail.extraction.json",
    ],
)
def test_phase2_validation_record_matches_ontology(filename: str) -> None:
    """Representative office, retail, and amendment records must validate."""
    record = (VALIDATION_DIRECTORY / filename).read_text(encoding="utf-8")

    LeaseExtraction.model_validate_json(record)
