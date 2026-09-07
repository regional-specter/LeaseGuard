"""Pin and verify official evaluators. Full corpora stay off the local machine."""

from pathlib import Path

from leaseguard.evaluation.models import BenchmarkSpec

REVISION_MARKER = ".leaseguard-evaluator-revision"
ALLOW_DOWNLOAD_ENV = "LEASEGUARD_ALLOW_BENCHMARK_DOWNLOAD"


class OfficialEvaluatorNotAvailableError(FileNotFoundError):
    """Raised when the pinned official evaluator is not present."""


class OfficialEvaluatorRevisionError(ValueError):
    """Raised when a checkout does not match the frozen evaluator revision."""


class OfficialEvaluatorDownloadError(PermissionError):
    """Raised when a local machine tries to clone a full benchmark repository."""


def evaluator_checkout_path(checkout_root: Path, spec: BenchmarkSpec) -> Path:
    """Return the expected directory for one pinned official evaluator."""
    return checkout_root / spec.benchmark_id


def build_clone_command(spec: BenchmarkSpec, destination: Path) -> tuple[str, ...]:
    """Return the git command that Colab should run for one official evaluator."""
    return (
        "git",
        "clone",
        "--filter=blob:none",
        spec.evaluator_repository_url,
        str(destination),
    )


def build_checkout_command(spec: BenchmarkSpec, destination: Path) -> tuple[str, ...]:
    """Return the git command that pins the cloned evaluator to the frozen SHA."""
    return ("git", "-C", str(destination), "checkout", spec.evaluator_revision)


def write_revision_marker(checkout: Path, revision: str) -> Path:
    """Record the evaluator SHA that a Colab checkout actually used."""
    checkout.mkdir(parents=True, exist_ok=True)
    marker = checkout / REVISION_MARKER
    marker.write_text(f"{revision}\n", encoding="utf-8")
    return marker


def verify_official_checkout(checkout: Path, spec: BenchmarkSpec) -> Path:
    """Confirm that a checkout exists and matches the frozen evaluator SHA."""
    marker = checkout / REVISION_MARKER
    if not marker.is_file():
        raise OfficialEvaluatorNotAvailableError(
            f"official evaluator for {spec.benchmark_id} is not present at {checkout}. "
            "Clone it only in Colab after setting "
            f"{ALLOW_DOWNLOAD_ENV}=1. Local machines must not download full benchmarks."
        )
    actual = marker.read_text(encoding="utf-8").strip()
    if actual != spec.evaluator_revision:
        raise OfficialEvaluatorRevisionError(
            f"{spec.benchmark_id} checkout {actual} does not match frozen "
            f"evaluator revision {spec.evaluator_revision}"
        )
    return checkout


def prepare_official_checkout(
    spec: BenchmarkSpec,
    checkout_root: Path,
    *,
    allow_download: bool,
    download_env: str | None,
) -> tuple[str, ...]:
    """Return clone arguments only when Colab explicitly allows downloads.

    This function does not clone. The Colab notebook must run the returned
    command, then write the revision marker after a successful checkout.
    """
    if not allow_download or download_env != "1":
        raise OfficialEvaluatorDownloadError(
            "full benchmark downloads are blocked on the local machine. "
            f"Set {ALLOW_DOWNLOAD_ENV}=1 in Colab and pass --allow-download."
        )
    destination = evaluator_checkout_path(checkout_root, spec)
    return build_clone_command(spec, destination)


def official_evaluate_command(
    spec: BenchmarkSpec,
    checkout: Path,
    predictions_path: Path,
    output_path: Path,
) -> tuple[str, ...]:
    """Return the command that should invoke the pinned official evaluator."""
    verify_official_checkout(checkout, spec)
    return (
        "python",
        spec.evaluator_entrypoint,
        "--predictions",
        str(predictions_path),
        "--output",
        str(output_path),
    )
