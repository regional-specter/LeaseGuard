"""Parsed and normalized document records produced by the pipeline."""

from typing import Literal

from pydantic import Field

from leaseguard.ingestion.models import MediaType, PipelineModel
from leaseguard.ontology.models import SourceDocument

BlockKind = Literal["heading", "paragraph", "table", "exhibit", "page"]


class PageRecord(PipelineModel):
    """Text extracted from one physical page or equivalent unit."""

    page_number: int = Field(ge=1)
    text: str
    warnings: list[str] = Field(default_factory=list)


class TextBlock(PipelineModel):
    """A heading, paragraph, table, or exhibit with character offsets."""

    block_id: str = Field(min_length=1)
    kind: BlockKind
    text: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    heading: str | None = Field(default=None, min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=1)


class ProcessedDocument(PipelineModel):
    """Validated intermediate record between raw files and labelled ontology output."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    source: SourceDocument
    media_type: MediaType
    raw_path: str = Field(min_length=1)
    pages: list[PageRecord] = Field(min_length=1)
    blocks: list[TextBlock] = Field(default_factory=list)
    normalized_text: str
    related_document_ids: list[str] = Field(default_factory=list)
    duplicate_of: str | None = Field(default=None, min_length=1)
    near_duplicate_of: list[str] = Field(default_factory=list)
    redaction_codes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
