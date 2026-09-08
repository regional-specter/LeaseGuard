"""Check Dataset v1 records for evidence, family leakage, and schema rules."""

from collections import defaultdict
from collections.abc import Sequence

from leaseguard.dataset.eligibility import (
    DatasetEligibilityError,
    assert_record_eligible,
    exact_duplicate_ids,
    near_duplicate_groups,
)
from leaseguard.dataset.evidence import (
    EvidenceAlignmentError,
    assert_answer_in_sources,
    assert_spans_in_sources,
    collect_extraction_evidence,
)
from leaseguard.dataset.models import (
    DatasetConfig,
    DatasetIssue,
    EvidenceConversationRecord,
    MultiTurnRecord,
    PreferencePairRecord,
    RecordBase,
    RefusalRecord,
    StructuredExtractionRecord,
)
from leaseguard.ontology.enums import AnswerStatus


def record_issues(record: RecordBase, config: DatasetConfig) -> list[DatasetIssue]:
    """Return eligibility, evidence, and subset-specific problems for one record."""
    issues: list[DatasetIssue] = []
    try:
        assert_record_eligible(record, config)
    except DatasetEligibilityError as error:
        issues.append(
            DatasetIssue(code="eligibility", message=str(error), record_id=record.record_id)
        )
        return issues

    try:
        _assert_subset_evidence(record)
    except EvidenceAlignmentError as error:
        issues.append(DatasetIssue(code="evidence", message=str(error), record_id=record.record_id))

    if isinstance(record, StructuredExtractionRecord):
        source_text = record.source_texts[record.extraction.source.document_id]
        for clause in record.extraction.clauses:
            if clause.text not in source_text:
                issues.append(
                    DatasetIssue(
                        code="clause_text",
                        message=(
                            f"{record.record_id}: clause {clause.clause_id} text "
                            "was not found in the source"
                        ),
                        record_id=record.record_id,
                    )
                )
        if record.extraction.source.document_type not in config.inclusion.document_types:
            issues.append(
                DatasetIssue(
                    code="document_type",
                    message=f"{record.record_id}: document type is outside Dataset v1 scope",
                    record_id=record.record_id,
                )
            )

    if isinstance(record, RefusalRecord) and record.answer.status is AnswerStatus.ANSWERED:
        issues.append(
            DatasetIssue(
                code="refusal_status",
                message=f"{record.record_id}: refusal subset cannot use answered status",
                record_id=record.record_id,
            )
        )

    if isinstance(record, PreferencePairRecord) and not record.ranking_reason.strip():
        issues.append(
            DatasetIssue(
                code="preference_rank",
                message=f"{record.record_id}: preference pairs require a ranking reason",
                record_id=record.record_id,
            )
        )
    return issues


def _assert_subset_evidence(record: RecordBase) -> None:
    """Dispatch evidence checks by subset."""
    if isinstance(record, StructuredExtractionRecord):
        assert_spans_in_sources(
            collect_extraction_evidence(record.extraction),
            record.source_texts,
            record_id=record.record_id,
        )
        return
    if isinstance(record, EvidenceConversationRecord):
        assert_answer_in_sources(record.answer, record.source_texts, record_id=record.record_id)
        return
    if isinstance(record, RefusalRecord):
        assert_answer_in_sources(record.answer, record.source_texts, record_id=record.record_id)
        return
    if isinstance(record, MultiTurnRecord):
        for index, turn in enumerate(record.turns):
            if turn.answer is not None:
                assert_answer_in_sources(
                    turn.answer,
                    record.source_texts,
                    record_id=f"{record.record_id}:turn-{index}",
                )
        return
    if isinstance(record, PreferencePairRecord):
        assert_answer_in_sources(
            record.preferred.answer,
            record.source_texts,
            record_id=f"{record.record_id}:preferred",
        )
        if not record.rejected.invented_facts:
            assert_answer_in_sources(
                record.rejected.answer,
                record.source_texts,
                record_id=f"{record.record_id}:rejected",
            )


def collection_issues(records: Sequence[RecordBase], config: DatasetConfig) -> list[DatasetIssue]:
    """Return duplicate, identity, and cross-family near-duplicate problems."""
    issues: list[DatasetIssue] = []
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        issues.append(DatasetIssue(code="record_id", message="record_id values must be unique"))

    for record_id, original in exact_duplicate_ids(records).items():
        issues.append(
            DatasetIssue(
                code="exact_duplicate",
                message=f"{record_id} duplicates {original}",
                record_id=record_id,
            )
        )

    for later_id, later_family, earlier_id, earlier_family in near_duplicate_groups(
        records,
        threshold=config.quality.near_duplicate_threshold,
        shingle_size=config.quality.shingle_size,
    ):
        issues.append(
            DatasetIssue(
                code="near_duplicate_leakage",
                message=(
                    f"{later_id} in family {later_family} is a near-duplicate of "
                    f"{earlier_id} in family {earlier_family}"
                ),
                record_id=later_id,
            )
        )

    families: dict[str, set[str]] = defaultdict(set)
    documents: dict[str, set[str]] = defaultdict(set)
    for record in records:
        families[record.family_id].add(record.split or "unassigned")
        for document_id in record.document_ids:
            documents[document_id].add(record.family_id)

    for family_id, splits in families.items():
        if len(splits) > 1:
            issues.append(
                DatasetIssue(
                    code="family_split_leakage",
                    message=f"family {family_id} appears in splits: {', '.join(sorted(splits))}",
                )
            )

    for document_id, family_ids in documents.items():
        if len(family_ids) > 1:
            issues.append(
                DatasetIssue(
                    code="document_family_leakage",
                    message=(
                        f"document {document_id} appears in families: "
                        + ", ".join(sorted(family_ids))
                    ),
                )
            )
    return issues
