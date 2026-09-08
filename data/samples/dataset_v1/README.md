# Dataset v1 Seed

This directory is the Git-sized Dataset v1 seed. It is synthetic, CC0-1.0, and small enough to validate in CI.

- `examples/` holds one JSON record per example, grouped by subset.
- `quality_review.json` records the accepted manual sample.

Do not add full lease corpora here. Export those with `scripts/prepare/build_dataset.py --output` into an ignored directory.
