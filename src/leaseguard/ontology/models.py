"""Validated records for LeaseGuard ontology version 1."""

from datetime import date
from decimal import Decimal
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from leaseguard.ontology.enums import (
    AmendmentAction,
    AnswerStatus,
    ClauseType,
    DocumentType,
    PartyRole,
    RightsStatus,
    StakeholderPerspective,
    TimingType,
    TimingUnit,
)

SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"


class SchemaModel(BaseModel):
    """Shared strict behavior for public LeaseGuard records."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class EvidenceSpan(SchemaModel):
    """Exact text supporting an extracted fact or answer."""

    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    page_label: str | None = Field(default=None, min_length=1)
    section: str | None = Field(default=None, min_length=1)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        """Require offsets as a valid zero-based, end-exclusive pair."""
        if (self.start_char is None) != (self.end_char is None):
            raise ValueError("start_char and end_char must be provided together")
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char <= self.start_char
        ):
            raise ValueError("end_char must be greater than start_char")
        if (
            self.page_number is None
            and self.page_label is None
            and self.section is None
            and self.start_char is None
        ):
            raise ValueError("evidence requires a page, section, or character offset")
        return self


class SourceDocument(SchemaModel):
    """Identity, provenance, and scope information for one document."""

    document_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    document_type: DocumentType
    source_name: str = Field(min_length=1)
    source_url: str | None = Field(default=None, min_length=1)
    rights_status: RightsStatus
    source_license: str | None = Field(default=None, min_length=1)
    rights_notes: str | None = Field(default=None, min_length=1)
    retrieved_at: date | None = None
    normalized_text_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    is_template: bool = False
    jurisdiction: Literal["US"] = "US"
    state: str | None = Field(default=None, min_length=2)
    language: Literal["en"] = "en"
    governing_law_text: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_rights_record(self) -> Self:
        """Require enough detail to audit reuse decisions."""
        if self.rights_status is RightsStatus.OPEN_LICENSE and self.source_license is None:
            raise ValueError("open-license sources require source_license")
        if self.rights_status is RightsStatus.REVIEW_REQUIRED and self.rights_notes is None:
            raise ValueError("sources awaiting review require rights_notes")
        return self


class Party(SchemaModel):
    """A named party and its role in the lease."""

    name: str = Field(min_length=1)
    role: PartyRole
    aliases: list[str] = Field(default_factory=list)
    evidence: list[EvidenceSpan] = Field(min_length=1)


class Premises(SchemaModel):
    """A physical space included in the lease."""

    description: str = Field(min_length=1)
    property_name: str | None = Field(default=None, min_length=1)
    address: str | None = Field(default=None, min_length=1)
    suite: str | None = Field(default=None, min_length=1)
    floor: str | None = Field(default=None, min_length=1)
    rentable_square_feet: Decimal | None = Field(default=None, ge=0)
    usable_square_feet: Decimal | None = Field(default=None, ge=0)
    permitted_use: str | None = Field(default=None, min_length=1)
    evidence: list[EvidenceSpan] = Field(min_length=1)


class Clause(SchemaModel):
    """A classified section of lease text."""

    clause_id: str = Field(min_length=1)
    clause_types: list[ClauseType] = Field(min_length=1)
    heading: str | None = Field(default=None, min_length=1)
    text: str = Field(min_length=1)
    summary: str | None = Field(default=None, min_length=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_page_range(self) -> Self:
        """Keep page ranges in document order."""
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("page_start and page_end must be provided together")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end must be greater than or equal to page_start")
        return self


class MonetaryTerm(SchemaModel):
    """A source expression and optional normalized monetary value."""

    label: str = Field(min_length=1)
    original_text: str = Field(min_length=1)
    amount: Decimal | None = Field(default=None, ge=0)
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")
    frequency: str | None = Field(default=None, min_length=1)
    period: str | None = Field(default=None, min_length=1)
    rate_basis: str | None = Field(default=None, min_length=1)
    evidence: list[EvidenceSpan] = Field(min_length=1)


class DateTerm(SchemaModel):
    """A source expression and optional normalized calendar date."""

    label: str = Field(min_length=1)
    original_text: str = Field(min_length=1)
    normalized_date: date | None = None
    trigger: str | None = Field(default=None, min_length=1)
    evidence: list[EvidenceSpan] = Field(min_length=1)


class Obligation(SchemaModel):
    """An action a named party is required to perform."""

    obligation_id: str = Field(min_length=1)
    responsible_party: str = Field(min_length=1)
    action: str = Field(min_length=1)
    beneficiary: str | None = Field(default=None, min_length=1)
    timing_type: TimingType
    due_date: date | None = None
    relative_deadline: str | None = Field(default=None, min_length=1)
    timing_value: int | None = Field(default=None, ge=0)
    timing_unit: TimingUnit | None = None
    trigger: str | None = Field(default=None, min_length=1)
    recurrence: str | None = Field(default=None, min_length=1)
    amount: MonetaryTerm | None = None
    condition: str | None = Field(default=None, min_length=1)
    evidence: list[EvidenceSpan] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_timing(self) -> Self:
        """Require the timing detail selected by timing_type."""
        required_fields = {
            TimingType.FIXED_DATE: self.due_date,
            TimingType.RELATIVE_DEADLINE: self.relative_deadline,
            TimingType.RECURRING: self.recurrence,
            TimingType.EVENT_BASED: self.trigger,
        }
        required_value = required_fields.get(self.timing_type)
        if self.timing_type is not TimingType.UNSTATED and required_value is None:
            raise ValueError(f"{self.timing_type.value} requires its matching timing value")
        if (self.timing_value is None) != (self.timing_unit is None):
            raise ValueError("timing_value and timing_unit must be provided together")
        return self


class AmendmentEffect(SchemaModel):
    """One explicit change an amendment makes to an earlier lease."""

    effect_id: str = Field(min_length=1)
    action: AmendmentAction
    target_document_id: str | None = Field(default=None, min_length=1)
    target_clause_reference: str | None = Field(default=None, min_length=1)
    description: str = Field(min_length=1)
    effective_date: date | None = None
    evidence: list[EvidenceSpan] = Field(min_length=1)


class LeaseExtraction(SchemaModel):
    """Complete structured extraction result for one lease document."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    source: SourceDocument
    related_document_ids: list[str] = Field(default_factory=list)
    parties: list[Party] = Field(default_factory=list)
    premises: list[Premises] = Field(default_factory=list)
    clauses: list[Clause] = Field(default_factory=list)
    monetary_terms: list[MonetaryTerm] = Field(default_factory=list)
    date_terms: list[DateTerm] = Field(default_factory=list)
    obligations: list[Obligation] = Field(default_factory=list)
    amendment_effects: list[AmendmentEffect] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DocumentAnswer(SchemaModel):
    """Simple and structured answer grounded in one or more evidence spans."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    question: str = Field(min_length=1)
    perspective: StakeholderPerspective
    status: AnswerStatus
    explanation: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def answered_status_requires_evidence(self) -> Self:
        """Prevent a successful answer without source support."""
        if self.status is AnswerStatus.ANSWERED and not self.evidence:
            raise ValueError("answered responses require at least one evidence span")
        return self
