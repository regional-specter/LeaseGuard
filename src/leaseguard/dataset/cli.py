"""Command-line entry point for Dataset v1 construction and validation."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from leaseguard.dataset.build import (
    build_from_examples,
    build_seed_dataset,
    export_bundle,
    verify_bundle_checksums,
)
from leaseguard.dataset.config import (
    DatasetConfigError,
    assert_holdout_alignment,
    load_dataset_config,
)
from leaseguard.dataset.models import DatasetBundle, DatasetReport
from leaseguard.dataset.paths import (
    default_dataset_config_path,
    default_quality_review_path,
    default_seed_examples_path,
)
from leaseguard.dataset.quality import reviews_are_complete
from leaseguard.evaluation.reports import write_json_report


def build_parser() -> argparse.ArgumentParser:
    """Create the Dataset v1 command parser."""
    parser = argparse.ArgumentParser(
        description="Build and validate LeaseGuard Dataset v1.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Dataset v1 rule file. Defaults to configs/data/dataset.v1.json.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-config", help="Validate Dataset v1 rules against the holdout.")
    subparsers.add_parser("validate-seed", help="Build and validate the committed seed examples.")
    build = subparsers.add_parser(
        "build", help="Build a dataset bundle from an examples directory."
    )
    build.add_argument("--examples", type=Path, default=default_seed_examples_path())
    build.add_argument("--quality-review", type=Path)
    build.add_argument("--output", type=Path)
    stats = subparsers.add_parser("stats", help="Print dataset statistics.")
    stats.add_argument("--bundle", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one Dataset v1 subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_dataset_config(args.config)
        assert_holdout_alignment(config)
    except (OSError, ValidationError, DatasetConfigError) as error:
        sys.stderr.write(f"{error}\n")
        return 1

    if args.command == "validate-config":
        sys.stdout.write(
            f"dataset {config.dataset_version} config={default_dataset_config_path().name} "
            f"subsets={len(config.subsets)}\n"
        )
        return 0

    if args.command == "validate-seed":
        bundle, report = build_from_examples(
            default_seed_examples_path(),
            config,
            quality_review_path=default_quality_review_path(),
        )
        return _finish_build(bundle, report, config, require_reviews=True)

    if args.command == "build":
        bundle, report = build_from_examples(
            args.examples,
            config,
            quality_review_path=args.quality_review,
        )
        if bundle is not None and args.output is not None:
            export_bundle(args.output, bundle)
        return _finish_build(bundle, report, config, require_reviews=False)

    if args.command == "stats":
        if args.bundle is not None:
            try:
                bundle = DatasetBundle.model_validate_json(args.bundle.read_text(encoding="utf-8"))
            except (OSError, ValidationError) as error:
                sys.stderr.write(f"{error}\n")
                return 1
            issues = verify_bundle_checksums(bundle, config)
            if issues:
                sys.stderr.write(issues[0].message + "\n")
                return 1
        else:
            bundle, report = build_seed_dataset()
            status = _finish_build(bundle, report, config, require_reviews=True)
            if status != 0 or bundle is None:
                return status
        sys.stdout.write(
            json.dumps(bundle.statistics.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _finish_build(
    bundle: DatasetBundle | None,
    report: DatasetReport,
    config: object,
    *,
    require_reviews: bool,
) -> int:
    from leaseguard.dataset.models import DatasetConfig

    assert isinstance(config, DatasetConfig)
    if not report.passed or bundle is None:
        for issue in report.issues:
            sys.stderr.write(f"{issue.code}: {issue.message}\n")
        return 1
    checksum_issues = verify_bundle_checksums(bundle, config)
    if checksum_issues:
        for issue in checksum_issues:
            sys.stderr.write(f"{issue.code}: {issue.message}\n")
        return 1
    if require_reviews and not reviews_are_complete(bundle.quality_samples):
        sys.stderr.write("seed quality samples must be accepted after manual review\n")
        return 1
    sys.stdout.write(
        f"dataset {bundle.dataset_version} records={bundle.statistics.record_count} "
        f"families={bundle.statistics.family_count}\n"
    )
    return 0


def write_report(path: Path, report: DatasetReport) -> Path:
    """Save a dataset report next to other small artifacts."""
    return write_json_report(path, report)
