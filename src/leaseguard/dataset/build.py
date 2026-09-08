"""Assemble a leakage-safe Dataset v1 bundle from labelled examples."""

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from leaseguard.dataset.config import assert_holdout_alignment, load_dataset_config
from leaseguard.dataset.eligibility import record_is_training_eligible
from leaseguard.dataset.examples import load_examples
from leaseguard.dataset.hashing import sha256_json
from leaseguard.dataset.models import (
    ALL_SUBSETS,
    DatasetBundle,
    DatasetChecksums,
    DatasetConfig,
    DatasetIssue,
    DatasetReport,
    DatasetSplit,
    DatasetStatistics,
    DatasetSubset,
    EvidenceConversationRecord,
    MultiTurnRecord,
    PreferencePairRecord,
    RawDomainRecord,
    RecordBase,
    RefusalRecord,
    SourceLicenseRecord,
    StructuredExtractionRecord,
    SubsetCounts,
)
from leaseguard.dataset.paths import default_quality_review_path, default_seed_examples_path
from leaseguard.dataset.quality import load_quality_reviews, select_quality_samples
from leaseguard.dataset.splits import assign_split
from leaseguard.dataset.validate import collection_issues, record_issues
from leaseguard.evaluation.reports import write_json_report


def assign_record_splits(records: list[RecordBase], config: DatasetConfig) -> None:
    """Write the family-hash split onto every record."""
    for record in records:
        record.split = assign_split(record.family_id, config.splits)


def source_license_records(
    records: Sequence[RecordBase],
    config: DatasetConfig,
) -> list[SourceLicenseRecord]:
    """Deduplicate source metadata by document identifier."""
    sources: dict[str, SourceLicenseRecord] = {}
    for record in records:
        eligible = record_is_training_eligible(record, config)
        for document_id in record.document_ids:
            if document_id in sources:
                continue
            sources[document_id] = SourceLicenseRecord(
                document_id=document_id,
                family_id=record.family_id,
                source_name=record.source_name,
                rights_status=record.rights_status,
                source_license=record.source_license,
                training_eligible=eligible,
                synthetic=record.synthetic,
            )
    return sorted(sources.values(), key=lambda item: item.document_id)


def compute_statistics(records: list[RecordBase]) -> DatasetStatistics:
    """Summarize splits, subsets, and licenses."""
    split_counter: Counter[DatasetSplit] = Counter()
    license_counter: Counter[str] = Counter()
    subset_split: dict[DatasetSubset, Counter[DatasetSplit]] = defaultdict(Counter)
    families: set[str] = set()
    documents: set[str] = set()
    synthetic_count = 0
    for record in records:
        if record.split is None:
            continue
        split_counter[record.split] += 1
        subset_split[record.subset][record.split] += 1
        families.add(record.family_id)
        documents.update(record.document_ids)
        license_key = record.source_license or record.rights_status.value
        license_counter[license_key] += 1
        if record.synthetic:
            synthetic_count += 1
    subset_counts = [
        SubsetCounts(
            subset=subset,
            total=subset_split[subset].total(),
            train=subset_split[subset]["train"],
            validation=subset_split[subset]["validation"],
            test=subset_split[subset]["test"],
        )
        for subset in ALL_SUBSETS
    ]
    return DatasetStatistics(
        record_count=len(records),
        family_count=len(families),
        source_count=len(documents),
        split_counts={
            "train": split_counter["train"],
            "validation": split_counter["validation"],
            "test": split_counter["test"],
        },
        subset_counts=subset_counts,
        license_counts=dict(license_counter),
        synthetic_count=synthetic_count,
    )


def compute_checksums(config: DatasetConfig, records: list[RecordBase]) -> DatasetChecksums:
    """Hash the frozen rules and each subset independently."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record.subset].append(record.model_dump(mode="json"))
    subsets: dict[str, str] = {
        subset: sha256_json(grouped.get(subset, [])) for subset in ALL_SUBSETS
    }
    return DatasetChecksums(
        config_sha256=sha256_json(config.model_dump(mode="json")),
        records_sha256=sha256_json([record.model_dump(mode="json") for record in records]),
        subsets=subsets,
    )


def _typed(records: list[RecordBase], subset: DatasetSubset) -> list[RecordBase]:
    return sorted(
        [record for record in records if record.subset == subset],
        key=lambda item: item.record_id,
    )


def build_bundle(
    records: list[RecordBase],
    config: DatasetConfig,
    *,
    reviews: dict[str, dict[str, Any]] | None = None,
    completed_at: datetime | None = None,
) -> tuple[DatasetBundle | None, DatasetReport]:
    """Validate examples and, when clean, return a checksummed bundle."""
    assign_record_splits(records, config)
    issues: list[DatasetIssue] = []
    for record in records:
        issues.extend(record_issues(record, config))
    issues.extend(collection_issues(records, config))
    finished = completed_at or datetime.now(UTC)
    if issues:
        return None, DatasetReport(
            passed=False,
            record_count=len(records),
            issues=issues,
            completed_at=finished,
        )

    ordered = sorted(records, key=lambda item: (item.subset, item.record_id))
    bundle = DatasetBundle(
        ontology_version=config.ontology_version,
        frozen_on=config.frozen_on,
        purpose=config.purpose,
        sources=source_license_records(ordered, config),
        raw_domain_text=cast(list[RawDomainRecord], _typed(ordered, "raw_domain_text")),
        structured_extraction=cast(
            list[StructuredExtractionRecord], _typed(ordered, "structured_extraction")
        ),
        evidence_conversation=cast(
            list[EvidenceConversationRecord], _typed(ordered, "evidence_conversation")
        ),
        refusal_uncertainty=cast(list[RefusalRecord], _typed(ordered, "refusal_uncertainty")),
        multi_turn=cast(list[MultiTurnRecord], _typed(ordered, "multi_turn")),
        preference_pairs=cast(list[PreferencePairRecord], _typed(ordered, "preference_pairs")),
        statistics=compute_statistics(ordered),
        checksums=compute_checksums(config, ordered),
        quality_samples=select_quality_samples(ordered, config, reviews),
    )
    missing_subsets = [subset for subset in config.subsets if getattr(bundle, subset) == []]
    if missing_subsets:
        return None, DatasetReport(
            passed=False,
            record_count=len(ordered),
            issues=[
                DatasetIssue(
                    code="empty_subset",
                    message="Dataset v1 is missing examples for: " + ", ".join(missing_subsets),
                )
            ],
            completed_at=finished,
        )
    return bundle, DatasetReport(
        passed=True,
        record_count=len(ordered),
        issues=[],
        completed_at=finished,
    )


def verify_bundle_checksums(bundle: DatasetBundle, config: DatasetConfig) -> list[DatasetIssue]:
    """Recompute checksums and report mismatches."""
    records: list[RecordBase] = [
        *bundle.raw_domain_text,
        *bundle.structured_extraction,
        *bundle.evidence_conversation,
        *bundle.refusal_uncertainty,
        *bundle.multi_turn,
        *bundle.preference_pairs,
    ]
    expected = compute_checksums(
        config, sorted(records, key=lambda item: (item.subset, item.record_id))
    )
    issues: list[DatasetIssue] = []
    if expected.config_sha256 != bundle.checksums.config_sha256:
        issues.append(DatasetIssue(code="checksum", message="config checksum mismatch"))
    if expected.records_sha256 != bundle.checksums.records_sha256:
        issues.append(DatasetIssue(code="checksum", message="records checksum mismatch"))
    if expected.subsets != bundle.checksums.subsets:
        issues.append(DatasetIssue(code="checksum", message="subset checksum mismatch"))
    return issues


def build_from_examples(
    examples_root: Path,
    config: DatasetConfig,
    *,
    quality_review_path: Path | None = None,
) -> tuple[DatasetBundle | None, DatasetReport]:
    """Load a directory of examples and build Dataset v1."""
    records = load_examples(examples_root)
    reviews: dict[str, dict[str, Any]] | None = None
    review_path = quality_review_path
    if review_path is not None and review_path.is_file():
        reviews = load_quality_reviews(json.loads(review_path.read_text(encoding="utf-8")))
    return build_bundle(records, config, reviews=reviews)


def build_seed_dataset() -> tuple[DatasetBundle | None, DatasetReport]:
    """Build the committed tiny Dataset v1 seed."""
    config = load_dataset_config()
    assert_holdout_alignment(config)
    return build_from_examples(
        default_seed_examples_path(),
        config,
        quality_review_path=default_quality_review_path(),
    )


def export_bundle(path: Path, bundle: DatasetBundle) -> Path:
    """Write a Dataset v1 bundle without placing it in Git by default."""
    return write_json_report(path, bundle)
