# Data Schemas

This directory contains versioned machine-readable schemas for LeaseGuard records.

Ontology version `1.0.0` was frozen on September 7, 2026 after the review recorded in `docs/phase2-validation.md`. It currently exposes:

- `LeaseExtraction.schema.json` for a complete document extraction.
- `DocumentAnswer.schema.json` for evidence-based questions and answers.

Evaluation records currently expose:

- `BenchmarkRegistry.schema.json` for the frozen professional benchmark list.
- `BenchmarkRun.schema.json` for official published runs.
- `RegressionSuiteConfig.schema.json` and `RegressionReport.schema.json` for the internal product suite.

Document-pipeline records currently expose:

- `SourceManifest.schema.json` for approved sources.
- `PipelineConfig.schema.json` for parser limits.
- `ProcessedDocument.schema.json` for intermediate parsed records.
- `BatchReport.schema.json` for one pipeline run.

The source models live in `src/leaseguard/ontology` and `src/leaseguard/evaluation`. Regenerate the JSON Schema files after an approved model change:

```bash
uv run python scripts/prepare/export_schemas.py
```

Generated files must be committed with the Python model changes. A breaking change requires a new major ontology version.
