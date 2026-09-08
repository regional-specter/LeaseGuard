"""Load and query approved source manifests."""

from pathlib import Path

from leaseguard.ingestion.models import PipelineConfig, SourceEntry, SourceManifest


def load_source_manifest(path: Path) -> SourceManifest:
    """Read and validate a source-manifest file."""
    return SourceManifest.model_validate_json(path.read_text(encoding="utf-8"))


def get_source(manifest: SourceManifest, document_id: str) -> SourceEntry:
    """Return one approved source by its stable identifier."""
    for source in manifest.sources:
        if source.document_id == document_id:
            return source
    raise KeyError(f"Unknown document_id: {document_id}")


def load_pipeline_config(path: Path) -> PipelineConfig:
    """Read pipeline size, media-type, and download limits."""
    return PipelineConfig.model_validate_json(path.read_text(encoding="utf-8"))
