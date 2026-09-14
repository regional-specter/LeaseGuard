"""Command-line entry point for professional benchmarks and local regression."""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from leaseguard.evaluation.metrics import (
    area_under_pr_curve,
    precision_at_recall,
    ranked_precision_recall,
)
from leaseguard.evaluation.models import BenchmarkRun, MetricResult
from leaseguard.evaluation.official import (
    ALLOW_DOWNLOAD_ENV,
    OfficialEvaluatorDownloadError,
    OfficialEvaluatorNotAvailableError,
    OfficialEvaluatorRevisionError,
    evaluator_checkout_path,
    official_evaluate_command,
    prepare_official_checkout,
    verify_official_checkout,
)
from leaseguard.evaluation.paths import (
    default_registry_path,
    default_regression_path,
    repo_root,
)
from leaseguard.evaluation.protocol import (
    HeadlineClaimError,
    HoldoutIntegrityError,
    assert_split_allowed_for_purpose,
    assert_training_ids_are_clean,
    bind_run_to_spec,
)
from leaseguard.evaluation.registry import get_benchmark, load_benchmark_registry
from leaseguard.evaluation.regression import load_regression_suite, run_regression_suite
from leaseguard.evaluation.reports import write_json_payload, write_json_report

HELP_EPILOG = (
    "Published claims require LegalBench, CUAD, ContractNLI, or LegalBench-RAG "
    "with that benchmark's official evaluator. The local lease suite is regression "
    "only and cannot support a breakthrough claim."
)


def build_parser() -> argparse.ArgumentParser:
    """Create the evaluation command parser."""
    parser = argparse.ArgumentParser(
        description="Run LeaseGuard evaluation commands.",
        epilog=HELP_EPILOG,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List frozen professional benchmarks.")
    subparsers.add_parser("validate-registry", help="Validate the frozen benchmark registry.")

    regression = subparsers.add_parser(
        "regression",
        help="Run the internal lease regression suite. Not a professional benchmark.",
    )
    regression.add_argument("--output", type=Path)

    ranked = subparsers.add_parser(
        "score-ranked",
        help="Score a tiny synthetic ranked list. Not an official CUAD result.",
    )
    ranked.add_argument("fixture", type=Path)
    ranked.add_argument("--output", type=Path)

    integrity = subparsers.add_parser(
        "check-integrity",
        help="Reject training IDs that overlap the evaluation holdout.",
    )
    integrity.add_argument("--training-ids-file", type=Path, required=True)

    record = subparsers.add_parser(
        "record-run",
        help="Validate a professional BenchmarkRun against the frozen registry.",
    )
    record.add_argument("run_file", type=Path)

    official = subparsers.add_parser(
        "official-status",
        help="Check whether a pinned official evaluator checkout is present.",
    )
    official.add_argument("--benchmark", required=True)
    official.add_argument("--checkout-root", type=Path, required=True)

    prepare = subparsers.add_parser(
        "prepare-official",
        help="Print the Colab git clone command. Downloads stay blocked locally.",
    )
    prepare.add_argument("--benchmark", required=True)
    prepare.add_argument("--checkout-root", type=Path, required=True)
    prepare.add_argument("--allow-download", action="store_true")

    evaluate = subparsers.add_parser(
        "official-evaluate-command",
        help="Print the official evaluator command after verifying the checkout pin.",
    )
    evaluate.add_argument("--benchmark", required=True)
    evaluate.add_argument("--checkout-root", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    subparsers.add_parser(
        "list-candidates",
        help="List frozen unmodified base-model candidates.",
    )
    subparsers.add_parser(
        "validate-baselines",
        help="Validate baseline configs against the frozen Dataset v1 eval split.",
    )
    subparsers.add_parser(
        "show-eval-split",
        help="Print the frozen Dataset v1 evaluation split.",
    )

    baseline = subparsers.add_parser(
        "baseline",
        help="Run one unmodified base-model comparison. Product-task scores are not a lab result.",
    )
    baseline.add_argument("--candidate", required=True)
    baseline.add_argument("--comparison", required=True)
    baseline.add_argument("--examples", type=Path)
    baseline.add_argument("--output-dir", type=Path)
    baseline.add_argument("--checkpoint-dir", type=Path)
    baseline.add_argument("--backend", choices=("scripted", "transformers"), default="scripted")
    baseline.add_argument("--script", type=Path)
    baseline.add_argument("--allow-download", action="store_true")
    baseline.add_argument("--hardware", default="Google Colab T4")
    baseline.add_argument("--limit", type=int)

    compare = subparsers.add_parser(
        "compare-baselines",
        help="Select the strongest practical candidate from saved run records.",
    )
    compare.add_argument("runs", nargs="+", type=Path)
    compare.add_argument("--output", type=Path)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one evaluation subcommand and return an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    root = repo_root()
    registry = load_benchmark_registry(default_registry_path())

    if args.command == "list":
        for benchmark in registry.benchmarks:
            sys.stdout.write(
                f"{benchmark.benchmark_id}\t{benchmark.name}\t{benchmark.venue}\t{benchmark.role}\n"
            )
        return 0

    if args.command == "validate-registry":
        sys.stdout.write(
            f"registry {registry.registry_version} frozen {registry.frozen_on} "
            f"with {len(registry.benchmarks)} professional benchmarks\n"
        )
        return 0

    if args.command == "regression":
        suite = load_regression_suite(default_regression_path())
        report = run_regression_suite(suite, root)
        if args.output is not None:
            write_json_report(args.output, report)
        failed = [check.name for check in report.checks if not check.passed]
        if failed:
            sys.stderr.write("regression failed: " + ", ".join(failed) + "\n")
            return 1
        sys.stdout.write("internal lease regression passed; this is not a professional benchmark\n")
        return 0

    if args.command == "score-ranked":
        payload = json.loads(args.fixture.read_text(encoding="utf-8"))
        relevances = [bool(item["relevant"]) for item in payload["predictions"]]
        precisions, recalls = ranked_precision_recall(relevances)
        metrics = [
            MetricResult(
                name="aupr",
                value=area_under_pr_curve(precisions, recalls),
                official=False,
            ),
            MetricResult(
                name="precision_at_80_recall",
                value=precision_at_recall(precisions, recalls, 0.8),
                official=False,
            ),
            MetricResult(
                name="precision_at_90_recall",
                value=precision_at_recall(precisions, recalls, 0.9),
                official=False,
            ),
        ]
        result = {
            "official": False,
            "warning": "Local ranked scoring is not the CUAD official evaluator.",
            "metrics": [metric.model_dump() for metric in metrics],
        }
        if args.output is not None:
            write_json_payload(args.output, result)
        sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return 0

    if args.command == "check-integrity":
        training_ids = [
            line.strip()
            for line in args.training_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        suite = load_regression_suite(default_regression_path())
        try:
            assert_training_ids_are_clean(
                training_ids,
                suite.blocked_from_training_document_ids,
            )
            assert_split_allowed_for_purpose("train", "fine-tuning")
        except HoldoutIntegrityError as error:
            sys.stderr.write(f"{error}\n")
            return 1
        sys.stdout.write("training IDs do not overlap the evaluation holdout\n")
        return 0

    if args.command == "record-run":
        try:
            run = BenchmarkRun.model_validate_json(args.run_file.read_text(encoding="utf-8"))
            spec = get_benchmark(registry, run.benchmark_id)
            bind_run_to_spec(run, spec)
        except (KeyError, HeadlineClaimError, ValidationError) as error:
            sys.stderr.write(f"{error}\n")
            return 1
        sys.stdout.write(f"run {run.run_id} matches frozen {spec.benchmark_id} protocol\n")
        return 0

    if args.command == "official-status":
        try:
            spec = get_benchmark(registry, args.benchmark)
        except KeyError as error:
            sys.stderr.write(f"{error}\n")
            return 1
        checkout = evaluator_checkout_path(args.checkout_root, spec)
        try:
            verify_official_checkout(checkout, spec)
        except OfficialEvaluatorNotAvailableError as error:
            sys.stderr.write(f"{error}\n")
            return 2
        except OfficialEvaluatorRevisionError as error:
            sys.stderr.write(f"{error}\n")
            return 1
        sys.stdout.write(
            f"{spec.benchmark_id} official evaluator {spec.evaluator_revision} is present\n"
        )
        return 0

    if args.command == "prepare-official":
        try:
            spec = get_benchmark(registry, args.benchmark)
            command = prepare_official_checkout(
                spec,
                args.checkout_root,
                allow_download=args.allow_download,
                download_env=os.environ.get(ALLOW_DOWNLOAD_ENV),
            )
        except KeyError as error:
            sys.stderr.write(f"{error}\n")
            return 1
        except OfficialEvaluatorDownloadError as error:
            sys.stderr.write(f"{error}\n")
            return 2
        sys.stdout.write(" ".join(command) + "\n")
        return 0

    if args.command == "official-evaluate-command":
        try:
            spec = get_benchmark(registry, args.benchmark)
            checkout = evaluator_checkout_path(args.checkout_root, spec)
            command = official_evaluate_command(
                spec,
                checkout,
                args.predictions,
                args.output,
            )
        except KeyError as error:
            sys.stderr.write(f"{error}\n")
            return 1
        except OfficialEvaluatorNotAvailableError as error:
            sys.stderr.write(f"{error}\n")
            return 2
        except OfficialEvaluatorRevisionError as error:
            sys.stderr.write(f"{error}\n")
            return 1
        sys.stdout.write(" ".join(command) + "\n")
        return 0

    if args.command == "list-candidates":
        from leaseguard.evaluation.baseline import load_baseline_registry

        baselines = load_baseline_registry()
        for candidate in baselines.candidates:
            sys.stdout.write(
                f"{candidate.candidate_id}\t{candidate.model_id}\t"
                f"{candidate.practical_context_length}\t{','.join(candidate.quantizations)}\n"
            )
        return 0

    if args.command == "validate-baselines":
        from leaseguard.evaluation.baseline import load_baseline_registry
        from leaseguard.evaluation.eval_split import (
            EvalSplitError,
            assert_eval_split_matches_dataset,
            load_eval_split,
        )

        try:
            baselines = load_baseline_registry()
            split = load_eval_split()
            assert_eval_split_matches_dataset(split)
        except (OSError, ValidationError, EvalSplitError) as error:
            sys.stderr.write(f"{error}\n")
            return 1
        sys.stdout.write(
            f"baselines {baselines.registry_version} candidates={len(baselines.candidates)} "
            f"comparisons={len(baselines.comparisons)} eval_split={split.split_version}\n"
        )
        return 0

    if args.command == "show-eval-split":
        from leaseguard.evaluation.eval_split import (
            EvalSplitError,
            assert_eval_split_matches_dataset,
            load_eval_split,
        )

        try:
            split = load_eval_split()
            assert_eval_split_matches_dataset(split)
        except (OSError, ValidationError, EvalSplitError) as error:
            sys.stderr.write(f"{error}\n")
            return 1
        sys.stdout.write(
            f"split {split.split_version} salt={split.salt} "
            f"seed_eval_records={len(split.seed_eval_record_ids)}\n"
        )
        return 0

    if args.command == "baseline":
        from leaseguard.evaluation.baseline import (
            BaselineConfigError,
            get_candidate,
            get_comparison,
            load_baseline_registry,
            run_baseline,
            write_run_manifest,
        )
        from leaseguard.inference.backend import (
            ModelBackendNotAvailableError,
            ModelDownloadError,
            create_backend,
            current_model_download_env,
            load_scripted_outputs,
        )

        try:
            baselines = load_baseline_registry()
            candidate = get_candidate(baselines, args.candidate)
            comparison = get_comparison(baselines, args.comparison)
            scripted_outputs = (
                load_scripted_outputs(args.script) if args.script is not None else None
            )
            backend = create_backend(
                candidate,
                comparison.quantization,
                backend=args.backend,
                allow_download=args.allow_download,
                download_env=current_model_download_env(),
                scripted_outputs=scripted_outputs,
                context_length_limit=candidate.practical_context_length,
            )
            baseline_run = run_baseline(
                candidate_id=args.candidate,
                comparison_id=args.comparison,
                backend=backend,
                examples_root=args.examples,
                output_dir=args.output_dir,
                checkpoint_dir=args.checkpoint_dir,
                registry=baselines,
                hardware=args.hardware,
                limit=args.limit,
            )
            backend.close()
        except (
            OSError,
            ValidationError,
            BaselineConfigError,
            ModelDownloadError,
            ModelBackendNotAvailableError,
            ValueError,
        ) as error:
            sys.stderr.write(f"{error}\n")
            return 1 if not isinstance(error, ModelDownloadError) else 2
        if args.output_dir is not None:
            write_run_manifest(
                args.output_dir / f"{baseline_run.run_id}.manifest.json", baseline_run
            )
        sys.stdout.write(
            f"run {baseline_run.run_id} backend={baseline_run.backend_kind} "
            f"oom={baseline_run.oom} records={len(baseline_run.predictions)}\n"
        )
        if baseline_run.backend_kind == "scripted":
            sys.stdout.write("scripted backend results cannot be published as baselines\n")
        return 0

    if args.command == "compare-baselines":
        from leaseguard.evaluation.baseline import load_baseline_registry
        from leaseguard.evaluation.compare import (
            compare_baseline_runs,
            load_baseline_runs,
            write_comparison_report,
        )

        try:
            baselines = load_baseline_registry()
            runs = load_baseline_runs(args.runs)
            comparison_report = compare_baseline_runs(runs, baselines)
        except (OSError, ValidationError) as error:
            sys.stderr.write(f"{error}\n")
            return 1
        if args.output is not None:
            write_comparison_report(args.output, comparison_report)
        selected = comparison_report.selected_candidate_id or "none"
        sys.stdout.write(
            f"selected {selected} incomplete={str(comparison_report.incomplete).lower()}\n"
        )
        sys.stdout.write(comparison_report.fine_tuning_hypothesis + "\n")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
