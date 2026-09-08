"""Locate Dataset v1 configuration and seed examples."""

from pathlib import Path

from leaseguard.evaluation.paths import repo_root


def default_dataset_config_path() -> Path:
    """Return the frozen Dataset v1 rule file."""
    return repo_root() / "configs" / "data" / "dataset.v1.json"


def default_regression_path() -> Path:
    """Return the internal regression suite that Dataset v1 must honor."""
    return repo_root() / "configs" / "evaluation" / "regression.v1.json"


def default_seed_examples_path() -> Path:
    """Return the tiny committed Dataset v1 examples directory."""
    return repo_root() / "data" / "samples" / "dataset_v1" / "examples"


def default_quality_review_path() -> Path:
    """Return the committed manual review file for the seed."""
    return repo_root() / "data" / "samples" / "dataset_v1" / "quality_review.json"
