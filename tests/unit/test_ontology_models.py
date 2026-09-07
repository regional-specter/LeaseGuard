"""Validation tests for ontology version 1."""

from datetime import date
from typing import Any

import pytest
from pydantic import ValidationError

from leaseguard.ontology import (
    AnswerStatus,
    Clause,
    ClauseType,
    DocumentAnswer,
    DocumentType,
    EvidenceSpan,
    LeaseExtraction,
    Obligation,
    SourceDocument,
    StakeholderPerspective,
    TimingType,
)


def evidence(**changes: Any) -> EvidenceSpan:
    """Build a valid evidence span with optional field changes."""
    values: dict[str, Any] = {
        "text": "Tenant shall pay rent on the first day of each month.",
        "page_number": 12,
        "section": "4.1 Base Rent",
    }
    values.update(changes)
    return EvidenceSpan(**values)


def obligation(**changes: Any) -> Obligation:
    """Build a valid obligation with optional field changes."""
    values: dict[str, Any] = {
        "obligation_id": "obligation-1",
        "responsible_party": "Example Tenant LLC",
        "action": "Pay monthly base rent",
        "timing_type": TimingType.RECURRING,
        "recurrence": "First day of each month",
        "evidence": [evidence()],
    }
    values.update(changes)
    return Obligation(**values)


def test_evidence_accepts_valid_offset_pairs() -> None:
    """Offsets are optional but valid when supplied as a complete pair."""
    assert evidence(start_char=10, end_char=20).end_char == 20
    assert evidence().start_char is None


@pytest.mark.parametrize(
    ("start_char", "end_char"),
    [(0, None), (None, 10), (10, 10), (11, 10)],
)
def test_evidence_rejects_invalid_offsets(
    start_char: int | None,
    end_char: int | None,
) -> None:
    """Incomplete, empty, and reversed offset ranges are invalid."""
    with pytest.raises(ValidationError):
        evidence(start_char=start_char, end_char=end_char)


def test_clause_pages_remain_in_document_order() -> None:
    """A clause cannot finish before it starts."""
    valid = Clause(
        clause_id="clause-1",
        clause_type=ClauseType.BASE_RENT,
        text="Base rent terms",
        page_start=4,
        page_end=5,
    )
    assert valid.page_end == 5

    with pytest.raises(ValidationError, match="page_end"):
        Clause(
            clause_id="clause-2",
            clause_type=ClauseType.BASE_RENT,
            text="Invalid range",
            page_start=5,
            page_end=4,
        )


@pytest.mark.parametrize(
    ("timing_type", "timing_value"),
    [
        (TimingType.FIXED_DATE, {"due_date": date(2027, 1, 1)}),
        (TimingType.RELATIVE_DEADLINE, {"relative_deadline": "10 days after notice"}),
        (TimingType.RECURRING, {"recurrence": "Monthly"}),
        (TimingType.EVENT_BASED, {"trigger": "Receipt of written notice"}),
        (TimingType.UNSTATED, {}),
    ],
)
def test_obligation_timing_matches_its_type(
    timing_type: TimingType,
    timing_value: dict[str, object],
) -> None:
    """Each timing type accepts its matching source detail."""
    changes: dict[str, object] = {"timing_type": timing_type, "recurrence": None}
    changes.update(timing_value)
    result = obligation(**changes)
    assert result.timing_type is timing_type


@pytest.mark.parametrize(
    "timing_type",
    [
        TimingType.FIXED_DATE,
        TimingType.RELATIVE_DEADLINE,
        TimingType.RECURRING,
        TimingType.EVENT_BASED,
    ],
)
def test_obligation_rejects_missing_timing_detail(timing_type: TimingType) -> None:
    """A stated timing type cannot omit its matching value."""
    with pytest.raises(ValidationError, match="matching timing value"):
        obligation(timing_type=timing_type, recurrence=None)


def test_answered_response_requires_evidence() -> None:
    """Successful answers need exact source support."""
    answer = DocumentAnswer(
        question="When is rent due?",
        perspective=StakeholderPerspective.NEUTRAL,
        status=AnswerStatus.ANSWERED,
        explanation="Rent is due on the first day of each month.",
        confidence=0.95,
        evidence=[evidence()],
    )
    assert answer.schema_version == "1.0.0"

    with pytest.raises(ValidationError, match="require at least one evidence"):
        DocumentAnswer(
            question="When is rent due?",
            perspective=StakeholderPerspective.NEUTRAL,
            status=AnswerStatus.ANSWERED,
            explanation="Rent is due monthly.",
            confidence=0.5,
        )


def test_non_answer_status_may_have_no_evidence() -> None:
    """The model may abstain when the document does not support an answer."""
    answer = DocumentAnswer(
        question="May the tenant install roof equipment?",
        perspective=StakeholderPerspective.TENANT,
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        explanation="The supplied document does not answer this question.",
        confidence=0.2,
    )
    assert answer.evidence == []


def test_complete_extraction_keeps_source_family_and_obligations() -> None:
    """The top-level result joins provenance and extracted records."""
    source = SourceDocument(
        document_id="document-1",
        family_id="family-1",
        filename="example-office-lease.pdf",
        sha256="a" * 64,
        document_type=DocumentType.OFFICE_LEASE,
        source_name="Synthetic test fixture",
        source_license="CC0-1.0",
    )
    extraction = LeaseExtraction(source=source, obligations=[obligation()])

    assert extraction.source.family_id == "family-1"
    assert extraction.obligations[0].responsible_party == "Example Tenant LLC"
