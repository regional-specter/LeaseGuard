"""Select a deterministic manual-review sample from Dataset v1."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from leaseguard.dataset.models import (
    DatasetConfig,
    DatasetSubset,
    QualityReviewStatus,
    QualitySample,
    RecordBase,
)


def select_quality_samples(
    records: Sequence[RecordBase],
    config: DatasetConfig,
    reviews: dict[str, dict[str, Any]] | None = None,
) -> list[QualitySample]:
    """Take the first N record IDs in each subset, overlaying committed reviews."""
    grouped: dict[DatasetSubset, list[RecordBase]] = defaultdict(list)
    for record in records:
        grouped[record.subset].append(record)

    samples: list[QualitySample] = []
    review_map = reviews or {}
    for subset in config.subsets:
        subset_records = sorted(grouped.get(subset, []), key=lambda item: item.record_id)
        for record in subset_records[: config.quality.quality_sample_per_subset]:
            if record.split is None:
                continue
            overlay = review_map.get(record.record_id, {})
            raw_status = overlay.get("review_status", "pending_review")
            notes = overlay.get("notes")
            review_status: QualityReviewStatus = "pending_review"
            if raw_status in ("pending_review", "accepted", "rejected"):
                review_status = raw_status
            samples.append(
                QualitySample(
                    record_id=record.record_id,
                    subset=subset,
                    split=record.split,
                    review_status=review_status,
                    notes=notes if isinstance(notes, str) and notes else None,
                )
            )
    return samples


def load_quality_reviews(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index committed review rows by record_id."""
    reviews: dict[str, dict[str, Any]] = {}
    for row in payload.get("samples", []):
        record_id = row.get("record_id")
        if isinstance(record_id, str) and record_id:
            reviews[record_id] = row
    return reviews


def reviews_are_complete(samples: Sequence[QualitySample]) -> bool:
    """Return whether every sampled seed record has been accepted."""
    statuses: tuple[QualityReviewStatus, ...] = tuple(sample.review_status for sample in samples)
    return bool(statuses) and all(status == "accepted" for status in statuses)
