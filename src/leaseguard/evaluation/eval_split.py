"""Load and verify the frozen Dataset v1 evaluation split."""

from pathlib import Path

from leaseguard.dataset.config import load_dataset_config
from leaseguard.dataset.models import DatasetConfig
from leaseguard.dataset.splits import assign_split
from leaseguard.evaluation.baseline_models import EvalSplitConfig
from leaseguard.evaluation.paths import default_eval_split_path
from leaseguard.evaluation.protocol import DEVELOPMENT_PURPOSES, HoldoutIntegrityError


class EvalSplitError(ValueError):
    """Raised when the frozen evaluation split no longer matches Dataset v1 rules."""


def load_eval_split(path: Path | None = None) -> EvalSplitConfig:
    """Read the frozen Dataset v1 evaluation-split pin."""
    split_path = path or default_eval_split_path()
    return EvalSplitConfig.model_validate_json(split_path.read_text(encoding="utf-8"))


def assert_eval_split_matches_dataset(
    split: EvalSplitConfig,
    config: DatasetConfig | None = None,
) -> None:
    """Require salt, blocked IDs, and seed family assignments to stay aligned."""
    dataset = config or load_dataset_config()
    if split.salt != dataset.splits.salt:
        raise EvalSplitError("evaluation split salt must match Dataset v1")
    if split.method != dataset.splits.method:
        raise EvalSplitError("evaluation split method must match Dataset v1")
    if split.dataset_version != dataset.dataset_version:
        raise EvalSplitError("evaluation split dataset_version must match Dataset v1")
    blocked = set(dataset.exclusion.blocked_from_training_document_ids)
    missing = sorted(set(split.blocked_from_training_document_ids) - blocked)
    if missing:
        raise EvalSplitError(
            "evaluation split is missing Dataset v1 blocked IDs: " + ", ".join(missing)
        )
    for family in split.seed_families:
        expected = assign_split(family.family_id, dataset.splits)
        if family.split != expected:
            raise EvalSplitError(
                f"{family.family_id} is frozen as {family.split} but family_hash assigns {expected}"
            )


def assert_eval_split_not_used_for_development(purpose: str) -> None:
    """Forbid prompt search and training against the frozen evaluation split."""
    if purpose in DEVELOPMENT_PURPOSES:
        raise HoldoutIntegrityError(
            f"the frozen Dataset v1 evaluation split cannot be used for {purpose}"
        )
