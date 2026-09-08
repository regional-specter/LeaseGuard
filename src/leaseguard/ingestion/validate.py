"""File-type, size, and checksum checks for acquired documents."""

import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from leaseguard.ingestion.models import MediaType, PipelineConfig, SourceEntry

PDF_MAGIC = b"%PDF"
DOCX_CONTENT_TYPES = "[Content_Types].xml"


class DocumentValidationError(ValueError):
    """Raised when a file does not match its manifest identity."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def detect_media_type(path: Path, payload: bytes | None = None) -> MediaType:
    """Identify PDF, DOCX, markdown, or plain text from magic bytes and suffix."""
    suffix = path.suffix.lower()
    header = payload[:8] if payload is not None else path.read_bytes()[:8]
    if header.startswith(PDF_MAGIC):
        return "pdf"
    if header.startswith(b"PK") and _is_docx(path, payload):
        return "docx"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix in {".txt"}:
        return "text"
    raise DocumentValidationError(f"unsupported file type for {path.name}")


def _is_docx(path: Path, payload: bytes | None) -> bool:
    source = BytesIO(payload) if payload is not None else path
    try:
        with ZipFile(source) as archive:
            names = set(archive.namelist())
    except (OSError, ValueError):
        return False
    return DOCX_CONTENT_TYPES in names and "word/document.xml" in names


def validate_acquired_file(
    path: Path,
    source: SourceEntry,
    config: PipelineConfig,
) -> MediaType:
    """Confirm size, checksum, and media type before parsing."""
    size = path.stat().st_size
    if size <= 0:
        raise DocumentValidationError(f"{source.document_id} is empty")
    if size > config.max_file_bytes:
        raise DocumentValidationError(
            f"{source.document_id} exceeds max_file_bytes ({config.max_file_bytes})"
        )
    digest = sha256_file(path)
    if digest != source.sha256:
        raise DocumentValidationError(
            f"{source.document_id} checksum {digest} does not match manifest {source.sha256}"
        )
    media_type = detect_media_type(path)
    if media_type != source.media_type:
        raise DocumentValidationError(
            f"{source.document_id} detected {media_type} but manifest lists {source.media_type}"
        )
    if media_type not in config.allowed_media_types:
        raise DocumentValidationError(f"{media_type} is not allowed by the pipeline config")
    return media_type
