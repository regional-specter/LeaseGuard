"""Document loading and source validation."""

from leaseguard.ingestion.acquire import (
    ALLOW_DOWNLOAD_ENV,
    DocumentAcquisitionError,
    DocumentDownloadError,
    acquire_source,
)
from leaseguard.ingestion.cli import main as ingestion_cli
from leaseguard.ingestion.manifest import (
    get_source,
    load_pipeline_config,
    load_source_manifest,
)
from leaseguard.ingestion.models import (
    BatchReport,
    PipelineConfig,
    ProcessingReport,
    SourceEntry,
    SourceManifest,
)
from leaseguard.ingestion.parse import parse_document
from leaseguard.ingestion.pipeline import process_manifest, process_source
from leaseguard.ingestion.validate import (
    DocumentValidationError,
    detect_media_type,
    sha256_file,
    validate_acquired_file,
)

__all__ = [
    "ALLOW_DOWNLOAD_ENV",
    "BatchReport",
    "DocumentAcquisitionError",
    "DocumentDownloadError",
    "DocumentValidationError",
    "PipelineConfig",
    "ProcessingReport",
    "SourceEntry",
    "SourceManifest",
    "acquire_source",
    "detect_media_type",
    "get_source",
    "ingestion_cli",
    "load_pipeline_config",
    "load_source_manifest",
    "parse_document",
    "process_manifest",
    "process_source",
    "sha256_file",
    "validate_acquired_file",
]
