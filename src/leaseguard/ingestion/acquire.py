"""Copy approved local files and optionally download them in Colab."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

from leaseguard.ingestion.models import PipelineConfig, SourceEntry
from leaseguard.ingestion.validate import sha256_file

ALLOW_DOWNLOAD_ENV = "LEASEGUARD_ALLOW_DOCUMENT_DOWNLOAD"


class DocumentDownloadError(PermissionError):
    """Raised when a local machine tries to download a source document."""


class DocumentAcquisitionError(FileNotFoundError):
    """Raised when a source file cannot be imported."""


def raw_destination(raw_root: Path, source: SourceEntry) -> Path:
    """Return the immutable raw path for one document."""
    filename = Path(source.local_path).name
    return raw_root / source.document_id / filename


def acquire_source(
    source: SourceEntry,
    *,
    raw_root: Path,
    search_roots: list[Path],
    config: PipelineConfig,
    allow_download: bool,
    download_env: str | None,
) -> Path:
    """Place the raw file under raw_root without overwriting a verified copy."""
    destination = raw_destination(raw_root, source)
    if destination.is_file() and sha256_file(destination) == source.sha256:
        return destination
    if destination.is_file():
        raise DocumentAcquisitionError(
            f"raw file for {source.document_id} exists but does not match the manifest checksum"
        )

    local = _find_local_copy(source, search_roots)
    if local is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, destination)
        return destination

    if source.source_url is None:
        raise DocumentAcquisitionError(f"{source.document_id} has no local file and no source_url")
    if not allow_download or download_env != "1":
        raise DocumentDownloadError(
            "document downloads are blocked on the local machine. "
            f"Set {config.allow_download_env}=1 in Colab and pass --allow-download."
        )
    return _download(source, destination)


def current_download_env() -> str | None:
    """Read the Colab download gate from the process environment."""
    return os.environ.get(ALLOW_DOWNLOAD_ENV)


def _find_local_copy(source: SourceEntry, search_roots: list[Path]) -> Path | None:
    candidates = [Path(source.local_path)]
    candidates.extend(root / source.local_path for root in search_roots)
    candidates.extend(root / Path(source.local_path).name for root in search_roots)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _download(source: SourceEntry, destination: Path) -> Path:
    if source.source_url is None:
        raise DocumentAcquisitionError(f"{source.document_id} has no source_url")
    request = Request(
        source.source_url,
        headers={"User-Agent": "LeaseGuard/0.1 (document pipeline)"},
        method="GET",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != source.sha256:
        raise DocumentAcquisitionError(
            f"downloaded {source.document_id} does not match the manifest checksum"
        )
    partial.write_bytes(payload)
    partial.replace(destination)
    return destination
