"""Locate repository files without downloading datasets."""

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Return the LeaseGuard repository root that contains pyproject.toml."""
    current = start.resolve() if start is not None else Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "leaseguard").is_dir():
            return candidate
    raise FileNotFoundError("LeaseGuard repository root was not found")


def default_registry_path() -> Path:
    """Return the frozen professional benchmark registry."""
    return repo_root() / "configs" / "evaluation" / "benchmarks.v1.json"


def default_regression_path() -> Path:
    """Return the internal lease regression suite configuration."""
    return repo_root() / "configs" / "evaluation" / "regression.v1.json"
