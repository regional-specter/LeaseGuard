"""Export public Pydantic models as versioned JSON Schema files."""

import json
from pathlib import Path

from pydantic import BaseModel

from leaseguard.evaluation import (
    BenchmarkRegistry,
    BenchmarkRun,
    RegressionReport,
    RegressionSuiteConfig,
)
from leaseguard.ontology import DocumentAnswer, LeaseExtraction

SCHEMA_MODELS: tuple[type[BaseModel], ...] = (
    LeaseExtraction,
    DocumentAnswer,
    BenchmarkRegistry,
    BenchmarkRun,
    RegressionReport,
    RegressionSuiteConfig,
)
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "schemas"


def main() -> None:
    """Write a stable JSON Schema file for every public output model."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for model in SCHEMA_MODELS:
        output_path = OUTPUT_DIRECTORY / f"{model.__name__}.schema.json"
        content = json.dumps(model.model_json_schema(), indent=2, sort_keys=True)
        output_path.write_text(f"{content}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
