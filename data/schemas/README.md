# Data Schemas

This directory contains versioned machine-readable schemas for LeaseGuard records.

Ontology version `1.0.0` currently exposes:

- `LeaseExtraction.schema.json` for a complete document extraction.
- `DocumentAnswer.schema.json` for evidence-based questions and answers.

The source models live in `src/leaseguard/ontology`. Regenerate the JSON Schema files after an approved model change:

```bash
uv run python scripts/prepare/export_schemas.py
```

Generated files must be committed with the Python model changes. A breaking change requires a new major ontology version.
