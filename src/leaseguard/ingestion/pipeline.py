"""Turn a manifest entry into a validated intermediate record."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from leaseguard.evaluation.reports import write_json_report
from leaseguard.ingestion.acquire import acquire_source
from leaseguard.ingestion.models import (
    BatchReport,
    FamilyGroup,
    PipelineCheckpoint,
    PipelineConfig,
    PipelineStage,
    ProcessingIssue,
    ProcessingReport,
    SourceEntry,
    SourceManifest,
)
from leaseguard.ingestion.parse import parse_document
from leaseguard.ingestion.validate import sha256_file, validate_acquired_file
from leaseguard.ontology.models import SourceDocument
from leaseguard.preprocessing.duplicates import exact_duplicate_map, near_duplicate_map
from leaseguard.preprocessing.models import PageRecord, ProcessedDocument
from leaseguard.preprocessing.normalize import assign_offsets, sha256_text
from leaseguard.preprocessing.segment import segment_pages
from leaseguard.privacy.redact import redact_personal_identifiers

PIPELINE_VERSION: Literal["1.0.0"] = "1.0.0"


def process_manifest(
    manifest: SourceManifest,
    *,
    config: PipelineConfig,
    raw_root: Path,
    processed_root: Path,
    checkpoint_dir: Path,
    search_roots: list[Path],
    allow_download: bool,
    download_env: str | None,
) -> BatchReport:
    """Acquire, parse, and export every approved source in family-aware batches."""
    started_at = datetime.now(tz=UTC)
    processed_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    documents: list[ProcessedDocument] = []
    reports: list[ProcessingReport] = []

    for source in manifest.sources:
        report, document = process_source(
            source,
            config=config,
            raw_root=raw_root,
            processed_root=processed_root,
            checkpoint_dir=checkpoint_dir,
            search_roots=search_roots,
            allow_download=allow_download,
            download_env=download_env,
        )
        reports.append(report)
        if document is not None:
            documents.append(document)

    duplicates = exact_duplicate_map(documents)
    near_duplicates = near_duplicate_map(
        documents,
        threshold=config.near_duplicate_threshold,
        shingle_size=config.shingle_size,
    )
    updated: list[ProcessedDocument] = []
    for document in documents:
        document_id = document.source.document_id
        patched = document.model_copy(
            update={
                "duplicate_of": duplicates.get(document_id),
                "near_duplicate_of": near_duplicates.get(document_id, []),
            }
        )
        output_path = processed_root / f"{document_id}.json"
        write_json_report(output_path, patched)
        updated.append(patched)
    documents = updated

    by_id = {report.document_id: report for report in reports}
    for document in documents:
        report = by_id[document.source.document_id]
        report.duplicate_of = document.duplicate_of
        report.near_duplicate_of = document.near_duplicate_of

    families = _family_groups(manifest, documents)
    completed_at = datetime.now(tz=UTC)
    return BatchReport(
        pipeline_version=PIPELINE_VERSION,
        started_at=started_at,
        completed_at=completed_at,
        reports=reports,
        families=families,
    )


def process_source(
    source: SourceEntry,
    *,
    config: PipelineConfig,
    raw_root: Path,
    processed_root: Path,
    checkpoint_dir: Path,
    search_roots: list[Path],
    allow_download: bool,
    download_env: str | None,
) -> tuple[ProcessingReport, ProcessedDocument | None]:
    """Process one document, resuming from an exported checkpoint when present."""
    output_path = processed_root / f"{source.document_id}.json"
    checkpoint_path = checkpoint_dir / f"{source.document_id}.json"
    issues: list[ProcessingIssue] = []
    if output_path.is_file() and _exported(checkpoint_path):
        document = ProcessedDocument.model_validate_json(output_path.read_text(encoding="utf-8"))
        return (
            ProcessingReport(
                document_id=source.document_id,
                family_id=source.family_id,
                status="skipped",
                completed_stage="exported",
                output_path=str(output_path),
                raw_path=document.raw_path,
                issues=[
                    ProcessingIssue(
                        document_id=source.document_id,
                        severity="warning",
                        code="resumed",
                        message="existing exported record was reused",
                    )
                ],
                raw_bytes_unchanged=True,
            ),
            document,
        )

    try:
        raw_path = acquire_source(
            source,
            raw_root=raw_root,
            search_roots=search_roots,
            config=config,
            allow_download=allow_download,
            download_env=download_env,
        )
        _write_checkpoint(checkpoint_path, source.document_id, "acquired", raw_path)
        media_type = validate_acquired_file(raw_path, source, config)
        _write_checkpoint(checkpoint_path, source.document_id, "validated", raw_path)
        pages = parse_document(raw_path, media_type)
        pages, redaction_codes, parse_warnings = _redact_pages(source.document_id, pages)
        issues.extend(parse_warnings)
        _write_checkpoint(checkpoint_path, source.document_id, "parsed", raw_path)
        if not any(page.text.strip() for page in pages):
            issues.append(
                ProcessingIssue(
                    document_id=source.document_id,
                    severity="warning",
                    code="unreadable_source",
                    message="no extractable digital text; OCR is deferred",
                )
            )
        normalized_text, blocks = segment_pages(pages)
        if blocks:
            assign_offsets(blocks, normalized_text)
        _write_checkpoint(checkpoint_path, source.document_id, "segmented", raw_path)
        if sha256_file(raw_path) != source.sha256:
            raise ValueError("raw file changed during processing")
        source_record = _source_document(source, normalized_text)
        export_pages = pages if pages else [PageRecord(page_number=1, text="", warnings=[])]
        document = ProcessedDocument(
            source=source_record,
            media_type=media_type,
            raw_path=str(raw_path),
            pages=export_pages,
            blocks=blocks,
            normalized_text=normalized_text,
            related_document_ids=list(source.related_document_ids),
            redaction_codes=list(redaction_codes),
            warnings=[issue.message for issue in issues if issue.severity == "warning"],
        )
        write_json_report(output_path, document)
        _write_checkpoint(checkpoint_path, source.document_id, "exported", output_path)
        return (
            ProcessingReport(
                document_id=source.document_id,
                family_id=source.family_id,
                status="exported",
                completed_stage="exported",
                output_path=str(output_path),
                raw_path=str(raw_path),
                issues=issues,
                raw_bytes_unchanged=True,
            ),
            document,
        )
    except Exception as error:
        issues.append(
            ProcessingIssue(
                document_id=source.document_id,
                severity="error",
                code=error.__class__.__name__,
                message=str(error),
            )
        )
        return (
            ProcessingReport(
                document_id=source.document_id,
                family_id=source.family_id,
                status="failed",
                issues=issues,
                raw_bytes_unchanged=True,
            ),
            None,
        )


def _redact_pages(
    document_id: str,
    pages: list[PageRecord],
) -> tuple[list[PageRecord], tuple[str, ...], list[ProcessingIssue]]:
    codes: list[str] = []
    issues: list[ProcessingIssue] = []
    redacted_pages: list[PageRecord] = []
    for page in pages:
        result = redact_personal_identifiers(page.text)
        codes.extend(result.codes)
        warnings = list(page.warnings)
        if result.codes:
            warnings.append("personal identifiers redacted in processed text only")
        redacted_pages.append(
            PageRecord(page_number=page.page_number, text=result.text, warnings=warnings)
        )
        for warning in page.warnings:
            issues.append(
                ProcessingIssue(
                    document_id=document_id,
                    severity="warning",
                    code="parse",
                    message=warning,
                )
            )
    unique_codes = tuple(dict.fromkeys(codes))
    return redacted_pages, unique_codes, issues


def _source_document(source: SourceEntry, normalized_text: str) -> SourceDocument:
    return SourceDocument(
        document_id=source.document_id,
        family_id=source.family_id,
        filename=Path(source.local_path).name,
        sha256=source.sha256,
        document_type=source.document_type,
        source_name=source.source_name,
        source_url=source.source_url,
        rights_status=source.rights_status,
        source_license=source.source_license,
        rights_notes=source.rights_notes,
        retrieved_at=source.retrieved_at,
        normalized_text_sha256=sha256_text(normalized_text),
        is_template=source.is_template,
        jurisdiction=source.jurisdiction,
        language=source.language,
    )


def _write_checkpoint(
    path: Path,
    document_id: str,
    stage: PipelineStage,
    artifact: Path,
) -> None:
    write_json_report(
        path,
        PipelineCheckpoint(
            document_id=document_id,
            completed_stage=stage,
            artifact_path=str(artifact),
            updated_at=datetime.now(tz=UTC),
        ),
    )


def _exported(checkpoint_path: Path) -> bool:
    if not checkpoint_path.is_file():
        return False
    checkpoint = PipelineCheckpoint.model_validate_json(checkpoint_path.read_text(encoding="utf-8"))
    return checkpoint.completed_stage == "exported"


def _family_groups(
    manifest: SourceManifest,
    documents: list[ProcessedDocument],
) -> list[FamilyGroup]:
    grouped: dict[str, list[str]] = {}
    for source in manifest.sources:
        grouped.setdefault(source.family_id, []).append(source.document_id)
    for document in documents:
        for related in document.related_document_ids:
            if related not in grouped[document.source.family_id]:
                grouped[document.source.family_id].append(related)
    return [
        FamilyGroup(family_id=family_id, document_ids=document_ids)
        for family_id, document_ids in grouped.items()
    ]
