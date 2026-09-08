# LeaseGuard Data

This directory stores only small, reviewable data files that belong in Git.

- `manifests/` records approved sources, licenses, checksums, and document metadata.
- `schemas/` contains machine-readable dataset and annotation schemas.
- `samples/` contains small synthetic or clearly reusable examples, including the Dataset v1 seed.
- `datasets/` is ignored except for a README; full Dataset v1 exports belong there or on Drive.

Downloaded documents should use local `data/raw`, `data/interim`, `data/processed`, and `data/annotations` directories. Those directories are ignored by Git.

Never place confidential leases, personal information, or data without a verified reuse license in this repository.
