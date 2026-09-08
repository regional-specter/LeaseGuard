"""Inclusion, exclusion, and quality gates for Dataset v1."""

import json
from collections.abc import Sequence

from leaseguard.dataset.models import DatasetConfig, RecordBase
from leaseguard.ontology.enums import RightsStatus


class DatasetEligibilityError(ValueError):
    """Raised when a record violates Dataset v1 inclusion or exclusion rules."""


def blocked_document_ids(config: DatasetConfig) -> set[str]:
    """Return document identifiers that cannot enter training or validation."""
    return set(config.exclusion.blocked_from_training_document_ids)


def record_is_training_eligible(record: RecordBase, config: DatasetConfig) -> bool:
    """Return whether a record may be used for model development."""
    blocked = blocked_document_ids(config)
    if any(document_id in blocked for document_id in record.document_ids):
        return False
    if record.rights_status in config.exclusion.blocked_rights:
        return False
    if record.rights_status not in config.inclusion.allowed_rights:
        return False
    return not (record.rights_status is RightsStatus.OPEN_LICENSE and record.source_license is None)


def assert_record_eligible(record: RecordBase, config: DatasetConfig) -> None:
    """Reject holdout, restricted, or undersized records."""
    if not record_is_training_eligible(record, config):
        raise DatasetEligibilityError(
            f"{record.record_id} is excluded by Dataset v1 rights or holdout rules"
        )
    for document_id, text in record.source_texts.items():
        if len(text) < config.quality.min_source_text_chars:
            raise DatasetEligibilityError(
                f"{record.record_id} source text for {document_id} is shorter than "
                f"{config.quality.min_source_text_chars} characters"
            )


def canonical_text(record: RecordBase) -> str:
    """Return the text used for exact and near-duplicate comparison."""
    payload = record.model_dump(mode="json")
    payload.pop("record_id", None)
    payload.pop("split", None)
    return json.dumps(payload, sort_keys=True, default=str)


def exact_duplicate_ids(records: Sequence[RecordBase]) -> dict[str, str]:
    """Map a later record_id to the first record with identical canonical text."""
    seen: dict[str, str] = {}
    duplicates: dict[str, str] = {}
    for record in records:
        original = seen.setdefault(canonical_text(record), record.record_id)
        if original != record.record_id:
            duplicates[record.record_id] = original
    return duplicates


def _word_shingles(text: str, size: int) -> set[str]:
    """Return overlapping word n-grams used for near-duplicate comparison."""
    words = [token.lower() for token in text.split() if token]
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + size]) for index in range(len(words) - size + 1)}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Return set overlap in the unit interval."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union)


def near_duplicate_groups(
    records: Sequence[RecordBase],
    *,
    threshold: float,
    shingle_size: int,
) -> list[tuple[str, str, str, str]]:
    """Return near-duplicate pairs that would leak across families."""
    shingles = [_word_shingles(canonical_text(record), shingle_size) for record in records]
    leaks: list[tuple[str, str, str, str]] = []
    for index, record in enumerate(records):
        for earlier_index in range(index):
            earlier = records[earlier_index]
            if record.family_id == earlier.family_id:
                continue
            score = _jaccard_similarity(shingles[index], shingles[earlier_index])
            if score >= threshold:
                leaks.append(
                    (record.record_id, record.family_id, earlier.record_id, earlier.family_id)
                )
    return leaks
