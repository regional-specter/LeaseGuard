# Configuration

Versioned configuration files keep experiments reproducible and prevent environment-specific values from being hard-coded.

- `data/` controls collection and preparation.
- `training/` controls model and fine-tuning runs.
- `evaluation/` controls datasets, metrics, and thresholds.
- `inference/` controls local model serving and generation.

Secrets and private paths must be provided through ignored local environment files, not committed configurations.
