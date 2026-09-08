"""Versioned Dataset v1 records, rules, and reports."""

from datetime import date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from leaseguard.ontology.enums import DocumentType, RightsStatus, StakeholderPerspective
from leaseguard.ontology.models import DocumentAnswer, LeaseExtraction

DATASET_VERSION: Literal["1.0.0"] = "1.0.0"
DatasetSubset = Literal[
    "raw_domain_text",
    "structured_extraction",
    "evidence_conversation",
    "refusal_uncertainty",
    "multi_turn",
    "preference_pairs",
]
ALL_SUBSETS: tuple[DatasetSubset, ...] = (
    "raw_domain_text",
    "structured_extraction",
    "evidence_conversation",
    "refusal_uncertainty",
    "multi_turn",
    "preference_pairs",
)
DatasetSplit = Literal["train", "validation", "test"]
QualityReviewStatus = Literal["pending_review", "accepted", "rejected"]


class DatasetModel(BaseModel):
    """Shared strict behavior for dataset records."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class InclusionRules(DatasetModel):
    """What may enter Dataset v1."""

    document_types: list[DocumentType] = Field(min_length=1)
    jurisdiction: Literal["US"]
    language: Literal["en"]
    allowed_rights: list[RightsStatus] = Field(min_length=1)
    require_evidence_for_answered: Literal[True] = True
    synthetic_must_not_invent_legal_facts: Literal[True] = True


class ExclusionRules(DatasetModel):
    """What must stay out of Dataset v1 training and validation."""

    blocked_from_training_document_ids: list[str] = Field(min_length=1)
    blocked_rights: list[RightsStatus] = Field(min_length=1)
    official_benchmark_test_splits: Literal[True] = True
    user_uploaded_documents: Literal[True] = True
    exact_duplicates: Literal[True] = True
    near_duplicates_across_splits: Literal[True] = True

    @model_validator(mode="after")
    def blocked_ids_are_unique(self) -> Self:
        """Keep the holdout list unambiguous."""
        blocked = self.blocked_from_training_document_ids
        if len(blocked) != len(set(blocked)):
            raise ValueError("blocked_from_training_document_ids must be unique")
        return self


class QualityRules(DatasetModel):
    """Checks applied to every retained example."""

    min_source_text_chars: int = Field(ge=1)
    near_duplicate_threshold: float = Field(gt=0, le=1)
    shingle_size: int = Field(ge=2, le=16)
    preference_requires_ranking_reason: Literal[True] = True
    evidence_must_occur_in_source: Literal[True] = True
    quality_sample_per_subset: int = Field(ge=1, le=50)


class SplitRules(DatasetModel):
    """Reproducible agreement-family assignment into train, validation, and test."""

    method: Literal["family_hash"]
    salt: str = Field(min_length=1)
    train_percent: int = Field(ge=1, le=98)
    validation_percent: int = Field(ge=1, le=98)
    test_percent: int = Field(ge=1, le=98)

    @model_validator(mode="after")
    def percents_sum_to_one_hundred(self) -> Self:
        """Require a complete partition."""
        total = self.train_percent + self.validation_percent + self.test_percent
        if total != 100:
            raise ValueError("split percents must sum to 100")
        return self


class DatasetConfig(DatasetModel):
    """Frozen Dataset v1 construction rules."""

    dataset_version: Literal["1.0.0"]
    frozen_on: date
    ontology_version: Literal["1.0.0"]
    purpose: str = Field(min_length=1)
    storage_policy: str = Field(min_length=1)
    inclusion: InclusionRules
    exclusion: ExclusionRules
    quality: QualityRules
    splits: SplitRules
    subsets: list[DatasetSubset] = Field(min_length=1)

    @model_validator(mode="after")
    def subsets_are_unique_and_complete(self) -> Self:
        """Require every Dataset v1 group exactly once."""
        if len(self.subsets) != len(set(self.subsets)):
            raise ValueError("subset names must be unique")
        required: set[str] = {
            "raw_domain_text",
            "structured_extraction",
            "evidence_conversation",
            "refusal_uncertainty",
            "multi_turn",
            "preference_pairs",
        }
        if set(self.subsets) != required:
            raise ValueError("Dataset v1 must include every required subset")
        return self


class SourceLicenseRecord(DatasetModel):
    """License and rights metadata for one document used in the dataset."""

    document_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    rights_status: RightsStatus
    source_license: str | None = Field(default=None, min_length=1)
    rights_notes: str | None = Field(default=None, min_length=1)
    training_eligible: bool
    synthetic: bool = False


class RecordBase(DatasetModel):
    """Shared identity for one dataset example."""

    record_id: str = Field(min_length=1)
    subset: DatasetSubset
    family_id: str = Field(min_length=1)
    document_ids: list[str] = Field(min_length=1)
    source_texts: dict[str, str] = Field(min_length=1)
    rights_status: RightsStatus
    source_license: str | None = Field(default=None, min_length=1)
    source_name: str = Field(min_length=1)
    synthetic: bool = False
    split: DatasetSplit | None = None

    @model_validator(mode="after")
    def source_texts_cover_documents(self) -> Self:
        """Require a source string for every cited document."""
        missing = [
            document_id for document_id in self.document_ids if document_id not in self.source_texts
        ]
        if missing:
            raise ValueError("source_texts missing document_id values: " + ", ".join(missing))
        extra = sorted(set(self.source_texts) - set(self.document_ids))
        if extra:
            raise ValueError("source_texts has unused document_id values: " + ", ".join(extra))
        return self


class RawDomainRecord(RecordBase):
    """Lease-domain text used for language exposure, not labelled extraction."""

    subset: Literal["raw_domain_text"] = "raw_domain_text"
    document_type: DocumentType


class StructuredExtractionRecord(RecordBase):
    """Ontology-valid labelled extraction grounded in source text."""

    subset: Literal["structured_extraction"] = "structured_extraction"
    extraction: LeaseExtraction

    @model_validator(mode="after")
    def extraction_matches_record(self) -> Self:
        """Keep labelled extraction identity aligned with the dataset envelope."""
        source = self.extraction.source
        if source.document_id not in self.document_ids:
            raise ValueError("extraction document_id must appear in document_ids")
        if source.family_id != self.family_id:
            raise ValueError("extraction family_id must match the record family_id")
        return self


class EvidenceConversationRecord(RecordBase):
    """Single-turn question and evidence-grounded answer."""

    subset: Literal["evidence_conversation"] = "evidence_conversation"
    perspective: StakeholderPerspective
    answer: DocumentAnswer


class RefusalRecord(RecordBase):
    """Uncertainty, missing evidence, or out-of-scope refusal."""

    subset: Literal["refusal_uncertainty"] = "refusal_uncertainty"
    perspective: StakeholderPerspective
    answer: DocumentAnswer


class ConversationTurn(DatasetModel):
    """One user or assistant turn in a multi-turn lease conversation."""

    role: Literal["user", "assistant"]
    text: str = Field(min_length=1)
    answer: DocumentAnswer | None = None

    @model_validator(mode="after")
    def assistant_turns_include_answers(self) -> Self:
        """Keep assistant replies structured and auditable."""
        if self.role == "assistant" and self.answer is None:
            raise ValueError("assistant turns require an answer record")
        if self.role == "user" and self.answer is not None:
            raise ValueError("user turns cannot include an answer record")
        return self


class MultiTurnRecord(RecordBase):
    """Follow-up conversation that stays on the same agreement family."""

    subset: Literal["multi_turn"] = "multi_turn"
    perspective: StakeholderPerspective
    turns: list[ConversationTurn] = Field(min_length=2)

    @model_validator(mode="after")
    def turns_alternate_from_user(self) -> Self:
        """Require a user-started, alternating conversation."""
        if self.turns[0].role != "user":
            raise ValueError("multi-turn conversations must start with a user turn")
        for index, turn in enumerate(self.turns):
            expected = "user" if index % 2 == 0 else "assistant"
            if turn.role != expected:
                raise ValueError("multi-turn conversations must alternate user and assistant")
        return self


class RankedResponse(DatasetModel):
    """One side of a preference pair."""

    text: str = Field(min_length=1)
    answer: DocumentAnswer
    invented_facts: bool = False


class PreferencePairRecord(RecordBase):
    """Two responses with an explicit human ranking."""

    subset: Literal["preference_pairs"] = "preference_pairs"
    perspective: StakeholderPerspective
    prompt: str = Field(min_length=1)
    preferred: RankedResponse
    rejected: RankedResponse
    ranking_reason: str = Field(min_length=8)

    @model_validator(mode="after")
    def ranking_is_clear(self) -> Self:
        """Reject ties, invented preferred answers, and missing reasons."""
        if self.preferred.invented_facts:
            raise ValueError("the preferred response cannot invent legal facts")
        if self.preferred.text == self.rejected.text:
            raise ValueError("preferred and rejected responses must differ")
        return self


class SubsetCounts(DatasetModel):
    """Example counts for one subset."""

    subset: DatasetSubset
    total: int = Field(ge=0)
    train: int = Field(ge=0)
    validation: int = Field(ge=0)
    test: int = Field(ge=0)


class DatasetStatistics(DatasetModel):
    """Summary counts for a built dataset."""

    record_count: int = Field(ge=0)
    family_count: int = Field(ge=0)
    source_count: int = Field(ge=0)
    split_counts: dict[DatasetSplit, int]
    subset_counts: list[SubsetCounts]
    license_counts: dict[str, int]
    synthetic_count: int = Field(ge=0)


class DatasetChecksums(DatasetModel):
    """SHA-256 digests for the configuration and each subset."""

    config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    records_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    subsets: dict[str, str]


class QualitySample(DatasetModel):
    """One record selected for manual review."""

    record_id: str = Field(min_length=1)
    subset: DatasetSubset
    split: DatasetSplit
    review_status: QualityReviewStatus = "pending_review"
    notes: str | None = Field(default=None, min_length=1)


class DatasetBundle(DatasetModel):
    """Complete Dataset v1 artifact, small enough to validate locally."""

    dataset_version: Literal["1.0.0"] = DATASET_VERSION
    ontology_version: Literal["1.0.0"]
    frozen_on: date
    purpose: str = Field(min_length=1)
    sources: list[SourceLicenseRecord]
    raw_domain_text: list[RawDomainRecord]
    structured_extraction: list[StructuredExtractionRecord]
    evidence_conversation: list[EvidenceConversationRecord]
    refusal_uncertainty: list[RefusalRecord]
    multi_turn: list[MultiTurnRecord]
    preference_pairs: list[PreferencePairRecord]
    statistics: DatasetStatistics
    checksums: DatasetChecksums
    quality_samples: list[QualitySample] = Field(default_factory=list)


class DatasetIssue(DatasetModel):
    """One validation finding."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    record_id: str | None = Field(default=None, min_length=1)


class DatasetReport(DatasetModel):
    """Machine-readable result of building or validating Dataset v1."""

    dataset_version: Literal["1.0.0"] = DATASET_VERSION
    passed: bool
    record_count: int = Field(ge=0)
    issues: list[DatasetIssue] = Field(default_factory=list)
    completed_at: datetime
