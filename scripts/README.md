# Scripts

Command-line entry scripts are grouped by task:

- `collect/` imports approved source documents.
- `prepare/` processes and validates datasets.
- `train/` launches reproducible model experiments.
- `evaluate/` evaluates models and saved predictions.

Scripts should remain small and call reusable functions from `src/leaseguard`.
