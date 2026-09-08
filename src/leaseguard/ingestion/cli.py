"""Command-line entry point for the document pipeline."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from leaseguard.evaluation.paths import repo_root
from leaseguard.evaluation.reports import write_json_report
from leaseguard.ingestion.acquire import current_download_env
from leaseguard.ingestion.manifest import load_pipeline_config, load_source_manifest
from leaseguard.ingestion.pipeline import process_manifest


def default_manifest_path() -> Path:
    """Return the frozen Phase 4 source manifest."""
    return repo_root() / "data" / "manifests" / "phase2_validation_sources.json"


def default_pipeline_config_path() -> Path:
    """Return the frozen pipeline configuration."""
    return repo_root() / "configs" / "data" / "pipeline.v1.json"


def build_parser() -> argparse.ArgumentParser:
    """Create the document-pipeline command parser."""
    parser = argparse.ArgumentParser(
        description="Acquire and parse approved LeaseGuard source documents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-manifest", help="Validate the source manifest.")
    process = subparsers.add_parser(
        "process",
        help="Import, parse, and export intermediate records from the manifest.",
    )
    process.add_argument("--manifest", type=Path)
    process.add_argument("--config", type=Path)
    process.add_argument("--raw-root", type=Path, required=True)
    process.add_argument("--processed-root", type=Path, required=True)
    process.add_argument("--checkpoint-dir", type=Path, required=True)
    process.add_argument("--search-root", type=Path, action="append", default=[])
    process.add_argument("--output", type=Path)
    process.add_argument("--allow-download", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one document-pipeline subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest_path = default_manifest_path()
    if args.command == "validate-manifest":
        try:
            manifest = load_source_manifest(manifest_path)
        except (OSError, ValidationError) as error:
            sys.stderr.write(f"{error}\n")
            return 1
        sys.stdout.write(
            f"manifest {manifest.manifest_version} with {len(manifest.sources)} sources\n"
        )
        return 0

    if args.command == "process":
        manifest = load_source_manifest(args.manifest or manifest_path)
        config = load_pipeline_config(args.config or default_pipeline_config_path())
        search_roots = list(args.search_root) if args.search_root else [repo_root()]
        report = process_manifest(
            manifest,
            config=config,
            raw_root=args.raw_root,
            processed_root=args.processed_root,
            checkpoint_dir=args.checkpoint_dir,
            search_roots=search_roots,
            allow_download=args.allow_download,
            download_env=current_download_env(),
        )
        if args.output is not None:
            write_json_report(args.output, report)
        failed = [item.document_id for item in report.reports if item.status == "failed"]
        sys.stdout.write(f"processed {len(report.reports)} documents; {len(failed)} failed\n")
        if failed:
            sys.stderr.write("failed: " + ", ".join(failed) + "\n")
            return 1
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
