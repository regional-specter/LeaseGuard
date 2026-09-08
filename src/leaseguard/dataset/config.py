"""Load frozen Dataset v1 configuration and keep it aligned with evaluation holdout."""

from pathlib import Path

from leaseguard.dataset.models import DatasetConfig
from leaseguard.dataset.paths import default_dataset_config_path, default_regression_path
from leaseguard.evaluation.regression import load_regression_suite


class DatasetConfigError(ValueError):
    """Raised when Dataset v1 rules conflict with the evaluation holdout."""


def load_dataset_config(path: Path | None = None) -> DatasetConfig:
    """Read and validate the frozen Dataset v1 configuration."""
    config_path = path or default_dataset_config_path()
    return DatasetConfig.model_validate_json(config_path.read_text(encoding="utf-8"))


def assert_holdout_alignment(
    config: DatasetConfig,
    regression_path: Path | None = None,
) -> None:
    """Require every regression document ID to stay blocked from training."""
    suite = load_regression_suite(regression_path or default_regression_path())
    blocked = set(config.exclusion.blocked_from_training_document_ids)
    missing = sorted(set(suite.blocked_from_training_document_ids) - blocked)
    if missing:
        raise DatasetConfigError(
            "Dataset v1 must block every regression document ID: " + ", ".join(missing)
        )
