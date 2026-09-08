"""Source manifests, pipeline configuration, and processing reports."""

from datetime import date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from leaseguard.ontology.enums import DocumentType, RightsStatus

MediaType = Literal["pdf", "docx", "markdown", "text"]
PipelineStage = Literal[
    "acquired",
    "validated",
    "parsed",
    "normalized",
    "segmented",
    "redacted",
    "exported",
]


class PipelineModel(BaseModel):
    """Shared strict behavior for document-pipeline records."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SourceEntry(PipelineModel):
    """One approved document listed in a source manifest."""

    document_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    document_type: DocumentType
    source_name: str = Field(min_length=1)
    source_url: str | None = Field(default=None, min_length=1)
    local_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    rights_status: RightsStatus
    source_license: str | None = Field(default=None, min_length=1)
    rights_notes: str | None = Field(default=None, min_length=1)
    is_template: bool = False
    media_type: MediaType
    jurisdiction: Literal["US"] = "US"
    language: Literal["en"] = "en"
    related_document_ids: list[str] = Field(default_factory=list)
    retrieved_at: date | None = None
    training_eligible: Literal[False] = False
    validation_role: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_rights_and_url(self) -> Self:
        """Keep reuse metadata and download targets auditable."""
        if self.rights_status is RightsStatus.OPEN_LICENSE and self.source_license is None:
            raise ValueError("open-license sources require source_license")
        if self.rights_status is RightsStatus.REVIEW_REQUIRED and self.rights_notes is None:
            raise ValueError("sources awaiting review require rights_notes")
        if self.source_url is not None and not self.source_url.startswith("https://"):
            raise ValueError("source_url must be an https URL")
        return self


class SourceManifest(PipelineModel):
    """Versioned list of approved source documents."""

    manifest_version: Literal["1.0.0"]
    purpose: str = Field(min_length=1)
    storage_policy: str = Field(min_length=1)
    sources: list[SourceEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def document_ids_are_unique(self) -> Self:
        """Prevent one identifier from selecting two files."""
        document_ids = [source.document_id for source in self.sources]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document_id values must be unique")
        return self


class PipelineConfig(PipelineModel):
    """Limits and policies for document acquisition and parsing."""

    pipeline_version: Literal["1.0.0"]
    allowed_media_types: list[MediaType] = Field(min_length=1)
    max_file_bytes: int = Field(gt=0)
    near_duplicate_threshold: float = Field(gt=0, le=1)
    ocr_enabled: Literal[False] = False
    allow_download_env: str = Field(min_length=1)
    shingle_size: int = Field(default=5, ge=2, le=16)


class PipelineCheckpoint(PipelineModel):
    """Resumable progress for one document in a Colab session."""

    document_id: str = Field(min_length=1)
    completed_stage: PipelineStage
    artifact_path: str = Field(min_length=1)
    updated_at: datetime


class ProcessingIssue(PipelineModel):
    """One warning or error produced while processing a document."""

    document_id: str = Field(min_length=1)
    severity: Literal["warning", "error"]
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class FamilyGroup(PipelineModel):
    """Documents that must stay together across dataset splits."""

    family_id: str = Field(min_length=1)
    document_ids: list[str] = Field(min_length=1)


class ProcessingReport(PipelineModel):
    """Machine-readable result for one manifest entry."""

    document_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    status: Literal["exported", "skipped", "failed"]
    completed_stage: PipelineStage | None = None
    output_path: str | None = Field(default=None, min_length=1)
    raw_path: str | None = Field(default=None, min_length=1)
    duplicate_of: str | None = Field(default=None, min_length=1)
    near_duplicate_of: list[str] = Field(default_factory=list)
    issues: list[ProcessingIssue] = Field(default_factory=list)
    raw_bytes_unchanged: bool = True


class BatchReport(PipelineModel):
    """One pipeline run over a source manifest."""

    pipeline_version: Literal["1.0.0"]
    started_at: datetime
    completed_at: datetime
    reports: list[ProcessingReport]
    families: list[FamilyGroup] = Field(default_factory=list)
