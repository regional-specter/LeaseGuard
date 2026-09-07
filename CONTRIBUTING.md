# Contributing to LeaseGuard

Thank you for helping improve LeaseGuard. Contributions should remain reproducible, evidence-based, and safe for legal documents.

## Development Setup

LeaseGuard supports Python 3.11 and 3.12. Install `uv`, clone the repository, and run:

```bash
uv sync --dev
uv run pre-commit install
```

Run the complete local quality check with:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

To apply automatic formatting:

```bash
uv run ruff format .
uv run ruff check --fix .
```

## Project Conventions

- Put reusable code in `src/leaseguard`.
- Keep scripts small and call package functions from them.
- Use notebooks only for exploration or demonstrations.
- Add tests for new behavior and bug fixes.
- Add type annotations to public and internal functions.
- Keep configuration outside implementation code.
- Record data sources, licenses, and processing decisions.
- Never commit private leases, personal information, secrets, model weights, or large datasets.

## Pull Requests

Keep each pull request focused on one clear change. Explain why the change is needed, how it was tested, and whether it affects data, evaluation, privacy, or model behavior.

All formatting, linting, typing, and test checks must pass before merge.
