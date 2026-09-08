"""Reproducible train, validation, and test assignment by agreement family."""

import hashlib

from leaseguard.dataset.models import DatasetSplit, SplitRules


def family_bucket(family_id: str, salt: str) -> int:
    """Map a family identifier onto a stable 0-99 bucket."""
    digest = hashlib.sha256(f"{salt}:{family_id}".encode()).hexdigest()
    return int(digest[:8], 16) % 100


def assign_split(family_id: str, rules: SplitRules) -> DatasetSplit:
    """Place an entire agreement family into one split."""
    bucket = family_bucket(family_id, rules.salt)
    if bucket < rules.train_percent:
        return "train"
    if bucket < rules.train_percent + rules.validation_percent:
        return "validation"
    return "test"
