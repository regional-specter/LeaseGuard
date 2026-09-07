"""Tests for professional-benchmark protocol, official checkout pins, and CLI."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from leaseguard.evaluation.cli import main
from leaseguard.evaluation.models import (
    BenchmarkRun,
    MetricResult,
    RegressionCheck,
    RegressionReport,
    RegressionSuiteConfig,
)
from leaseguard.evaluation.official import (
    OfficialEvaluatorDownloadError,
    OfficialEvaluatorNotAvailableError,
    OfficialEvaluatorRevisionError,
    build_checkout_command,
    evaluator_checkout_path,
    official_evaluate_command,
    prepare_official_checkout,
    verify_official_checkout,
    write_revision_marker,
)
from leaseguard.evaluation.paths import default_registry_path, repo_root
from leaseguard.evaluation.protocol import (
    HeadlineClaimError,
    HoldoutIntegrityError,
    assert_not_headline_benchmark,
    assert_split_allowed_for_purpose,
    assert_training_ids_are_clean,
    bind_run_to_spec,
)
from leaseguard.evaluation.registry import get_benchmark, load_benchmark_registry
from leaseguard.evaluation.regression import load_regression_suite, run_regression_suite
from leaseguard.evaluation.reports import write_json_report

ROOT = repo_root()
REGISTRY = load_benchmark_registry(default_registry_path())
CUAD = get_benchmark(REGISTRY, "cuad")


def benchmark_run(**changes: Any) -> BenchmarkRun:
    """Build a valid external benchmark run record."""
    started_at = datetime(2026, 9, 7, 8, 0, tzinfo=UTC)
    values: dict[str, Any] = {
        "run_id": "cuad-qwen-base-001",
        "benchmark_id": "cuad",
        "benchmark_registry_version": "1.0.0",
        "repository_revision": "67faa0e6023b04fcaae6cc09497ab00e5d63a2a2",
        "model_id": "Qwen/Qwen3.5-9B",
        "model_revision": "model-revision",
        "dataset_revision": "dataset-revision",
        "evaluator_revision": "67faa0e6023b04fcaae6cc09497ab00e5d63a2a2",
        "split": "test",
        "hardware": "Google Colab T4",
        "quantization": "4-bit",
        "prompt_sha256": "b" * 64,
        "random_seed": 3407,
        "started_at": started_at,
        "completed_at": started_at + timedelta(minutes=20),
        "metrics": [MetricResult(name="aupr", value=0.5, official=True)],
        "raw_predictions_uri": "drive://leaseguard/runs/cuad-qwen-base-001/predictions.json",
        "used_official_evaluator": True,
        "contamination_notes": "Public benchmark exposure during base pretraining is unknown.",
    }
    values.update(changes)
    return BenchmarkRun(**values)


def test_repo_root_requires_package_layout(tmp_path: Path) -> None:
    """A random directory is not treated as the LeaseGuard repository."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='other'\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="repository root"):
        repo_root(tmp_path)


def test_test_split_cannot_be_used_for_development() -> None:
    """Official test examples stay out of training and prompt search."""
    assert_split_allowed_for_purpose("train", "fine-tuning")
    with pytest.raises(HoldoutIntegrityError, match="fine-tuning"):
        assert_split_allowed_for_purpose("test", "fine-tuning")
    with pytest.raises(HoldoutIntegrityError, match="prompt-selection"):
        assert_split_allowed_for_purpose("official_test", "prompt-selection")


def test_evaluation_document_ids_cannot_enter_training() -> None:
    """The frozen holdout list blocks leakage into a training manifest."""
    assert_training_ids_are_clean(["train-lease-1"], ["gsa-l100-2026"])
    with pytest.raises(HoldoutIntegrityError, match="gsa-l100-2026"):
        assert_training_ids_are_clean(["gsa-l100-2026"], ["gsa-l100-2026"])


def test_published_run_must_match_frozen_evaluator_pins() -> None:
    """A headline CUAD claim is rejected if the evaluator SHA drifted."""
    run = benchmark_run(
        evaluator_revision=CUAD.evaluator_revision,
        metrics=[MetricResult(name="aupr", value=0.5, official=True)],
    )
    bind_run_to_spec(run, CUAD)
    with pytest.raises(HeadlineClaimError, match="evaluator revision"):
        bind_run_to_spec(
            benchmark_run(
                evaluator_revision="0" * 40,
                metrics=[MetricResult(name="aupr", value=0.5, official=True)],
            ),
            CUAD,
        )
    with pytest.raises(HeadlineClaimError, match="does not match"):
        bind_run_to_spec(run, get_benchmark(REGISTRY, "legalbench"))
    with pytest.raises(HeadlineClaimError, match="dataset revision"):
        bind_run_to_spec(
            benchmark_run(
                repository_revision="0" * 40,
                evaluator_revision=CUAD.evaluator_revision,
                metrics=[MetricResult(name="aupr", value=0.5, official=True)],
            ),
            CUAD,
        )
    unofficial_evaluator = benchmark_run(
        evaluator_revision=CUAD.evaluator_revision,
        metrics=[MetricResult(name="aupr", value=0.5, official=True)],
    )
    unofficial_evaluator.used_official_evaluator = False  # type: ignore[assignment]
    with pytest.raises(HeadlineClaimError, match="official evaluator"):
        bind_run_to_spec(unofficial_evaluator, CUAD)
    unofficial_metric = benchmark_run(
        evaluator_revision=CUAD.evaluator_revision,
        metrics=[MetricResult(name="aupr", value=0.5, official=True)],
    )
    unofficial_metric.metrics[0].official = False
    with pytest.raises(HeadlineClaimError, match="unofficial metrics"):
        bind_run_to_spec(unofficial_metric, CUAD)


def test_internal_regression_cannot_be_a_headline_benchmark() -> None:
    """The product suite is typed and checked as non-professional."""
    suite = load_regression_suite(ROOT / "configs" / "evaluation" / "regression.v1.json")
    report = run_regression_suite(suite, ROOT)
    assert_not_headline_benchmark(report)
    assert report.is_professional_benchmark is False
    assert report.headline_claim_allowed is False
    assert all(check.passed for check in report.checks)
    assert report.metrics[0].official is False

    fake = RegressionReport.model_construct(
        report_id="fake",
        suite_version="1.0.0",
        is_professional_benchmark=True,
        headline_claim_allowed=True,
        records_scored=0,
        checks=report.checks,
        metrics=report.metrics,
        completed_at=report.completed_at,
    )
    with pytest.raises(HeadlineClaimError, match="not a professional benchmark"):
        assert_not_headline_benchmark(fake)
    dumped = report.model_dump()
    dumped["metrics"] = [{"name": "aupr", "value": 0.1, "official": True}]
    with pytest.raises(ValidationError, match="cannot be marked official"):
        RegressionReport.model_validate(dumped)


def test_regression_detects_family_and_holdout_failures() -> None:
    """Broken family membership or missing holdout IDs fail the suite."""
    suite = load_regression_suite(ROOT / "configs" / "evaluation" / "regression.v1.json")
    family_broken = suite.model_copy(deep=True)
    family_broken.required_families[0].member_document_ids.append("missing-doc")
    family_report = run_regression_suite(family_broken, ROOT)
    assert not next(
        check for check in family_report.checks if check.name == "agreement_family_splits"
    ).passed

    holdout_broken = suite.model_copy(deep=True)
    holdout_broken.blocked_from_training_document_ids = ["sample-office-lease"]
    holdout_report = run_regression_suite(holdout_broken, ROOT)
    assert not next(
        check for check in holdout_report.checks if check.name == "evaluation_holdout"
    ).passed

    mismatched_family = suite.model_copy(deep=True)
    mismatched_family.records[0].family_id = "wrong-family"
    mismatched_report = run_regression_suite(mismatched_family, ROOT)
    assert not next(
        check for check in mismatched_report.checks if check.name == "schema_validity"
    ).passed


def test_regression_detects_missing_records_and_answers() -> None:
    """Missing extraction or answer files fail the corresponding checks."""
    suite = load_regression_suite(ROOT / "configs" / "evaluation" / "regression.v1.json")
    missing_record = suite.model_copy(deep=True)
    missing_record.records[0].path = "does-not-exist.json"
    record_report = run_regression_suite(missing_record, ROOT)
    assert not next(
        check for check in record_report.checks if check.name == "schema_validity"
    ).passed

    missing_answer = suite.model_copy(deep=True)
    missing_answer.answer_records[0].path = "does-not-exist.json"
    answer_report = run_regression_suite(missing_answer, ROOT)
    assert not next(
        check for check in answer_report.checks if check.name == "grounded_answers_and_abstention"
    ).passed

    wrong_status = suite.model_copy(deep=True)
    wrong_status.answer_records[0].expected_status = "insufficient_evidence"
    status_report = run_regression_suite(wrong_status, ROOT)
    assert not next(
        check for check in status_report.checks if check.name == "grounded_answers_and_abstention"
    ).passed


def test_regression_config_rejects_professional_status() -> None:
    """The suite file cannot silently claim to be a lab benchmark."""
    payload = json.loads(
        (ROOT / "configs" / "evaluation" / "regression.v1.json").read_text(encoding="utf-8")
    )
    payload["is_professional_benchmark"] = True
    with pytest.raises(ValidationError):
        RegressionSuiteConfig.model_validate(payload)
    payload["is_professional_benchmark"] = False
    payload["blocked_from_training_document_ids"].append("gsa-l100-2026")
    with pytest.raises(ValidationError, match="must be unique"):
        RegressionSuiteConfig.model_validate(payload)


def test_official_checkout_is_pinned_and_not_downloaded_locally(
    tmp_path: Path,
) -> None:
    """Local machines cannot clone evaluators; Colab must match the frozen SHA."""
    checkout = evaluator_checkout_path(tmp_path, CUAD)
    with pytest.raises(OfficialEvaluatorNotAvailableError, match="not present"):
        verify_official_checkout(checkout, CUAD)
    with pytest.raises(OfficialEvaluatorDownloadError, match="blocked"):
        prepare_official_checkout(
            CUAD,
            tmp_path,
            allow_download=False,
            download_env=None,
        )
    command = prepare_official_checkout(
        CUAD,
        tmp_path,
        allow_download=True,
        download_env="1",
    )
    assert command[0] == "git"
    assert CUAD.evaluator_repository_url in command
    assert "checkout" in build_checkout_command(CUAD, checkout)

    write_revision_marker(checkout, "0" * 40)
    with pytest.raises(OfficialEvaluatorRevisionError, match="does not match"):
        verify_official_checkout(checkout, CUAD)
    write_revision_marker(checkout, CUAD.evaluator_revision)
    verify_official_checkout(checkout, CUAD)
    evaluate = official_evaluate_command(
        CUAD,
        checkout,
        tmp_path / "predictions.json",
        tmp_path / "metrics.json",
    )
    assert evaluate[1] == "evaluate.py"


def test_cli_regression_failure_is_not_a_benchmark_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failed internal suite still cannot be described as a professional result."""
    suite = load_regression_suite(ROOT / "configs" / "evaluation" / "regression.v1.json")
    report = run_regression_suite(suite, ROOT)
    report.checks[0] = RegressionCheck(
        name="schema_validity",
        passed=False,
        detail="forced failure",
    )

    def fake_run(_suite: RegressionSuiteConfig, _root: Path) -> RegressionReport:
        return report

    monkeypatch.setattr("leaseguard.evaluation.cli.run_regression_suite", fake_run)
    assert main(["regression"]) == 1
    assert "regression failed" in capsys.readouterr().err


def test_cli_lists_and_validates_professional_registry(capsys: pytest.CaptureFixture[str]) -> None:
    """The repeatable command exposes only the frozen lab benchmarks."""
    assert main(["list"]) == 0
    listed = capsys.readouterr().out
    assert "legalbench" in listed
    assert "cuad" in listed
    assert "contractnli" in listed
    assert "legalbench-rag" in listed
    assert "NeurIPS 2023" in listed
    assert main(["validate-registry"]) == 0
    assert "4 professional benchmarks" in capsys.readouterr().out


def test_cli_runs_internal_regression_and_ranked_fixture(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Local commands score product regression and synthetic ranked lists only."""
    report_path = tmp_path / "regression.json"
    assert main(["regression", "--output", str(report_path)]) == 0
    assert "not a professional benchmark" in capsys.readouterr().out
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["is_professional_benchmark"] is False

    fixture = ROOT / "tests" / "fixtures" / "evaluation" / "ranked_predictions.json"
    ranked_path = tmp_path / "ranked.json"
    assert main(["score-ranked", str(fixture), "--output", str(ranked_path)]) == 0
    ranked = json.loads(ranked_path.read_text(encoding="utf-8"))
    assert ranked["official"] is False
    assert "not the CUAD official evaluator" in ranked["warning"]


def test_cli_integrity_and_official_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integrity fails on holdout overlap, and official scoring stays Colab-gated."""
    clean = tmp_path / "clean.txt"
    clean.write_text("train-lease-1\n", encoding="utf-8")
    assert main(["check-integrity", "--training-ids-file", str(clean)]) == 0
    dirty = tmp_path / "dirty.txt"
    dirty.write_text("gsa-l100-2026\n", encoding="utf-8")
    assert main(["check-integrity", "--training-ids-file", str(dirty)]) == 1
    assert "cannot enter training" in capsys.readouterr().err

    assert (
        main(
            [
                "official-status",
                "--benchmark",
                "cuad",
                "--checkout-root",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert (
        main(["official-status", "--benchmark", "invented", "--checkout-root", str(tmp_path)]) == 1
    )
    assert (
        main(
            [
                "official-evaluate-command",
                "--benchmark",
                "invented",
                "--checkout-root",
                str(tmp_path),
                "--predictions",
                str(tmp_path / "preds.json"),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
        == 1
    )
    assert (
        main(
            [
                "official-evaluate-command",
                "--benchmark",
                "cuad",
                "--checkout-root",
                str(tmp_path),
                "--predictions",
                str(tmp_path / "preds.json"),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "prepare-official",
                "--benchmark",
                "invented",
                "--checkout-root",
                str(tmp_path),
                "--allow-download",
            ]
        )
        == 1
    )
    assert (
        main(
            [
                "prepare-official",
                "--benchmark",
                "cuad",
                "--checkout-root",
                str(tmp_path),
            ]
        )
        == 2
    )
    monkeypatch.setenv("LEASEGUARD_ALLOW_BENCHMARK_DOWNLOAD", "1")
    write_revision_marker(tmp_path / "cuad", "0" * 40)
    assert (
        main(
            [
                "official-evaluate-command",
                "--benchmark",
                "cuad",
                "--checkout-root",
                str(tmp_path),
                "--predictions",
                str(tmp_path / "preds.json"),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
        == 1
    )
    assert (
        main(
            [
                "prepare-official",
                "--benchmark",
                "cuad",
                "--checkout-root",
                str(tmp_path),
                "--allow-download",
            ]
        )
        == 0
    )
    assert "git clone" in capsys.readouterr().out

    write_revision_marker(tmp_path / "cuad", CUAD.evaluator_revision)
    assert (
        main(
            [
                "official-status",
                "--benchmark",
                "cuad",
                "--checkout-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "official-evaluate-command",
                "--benchmark",
                "cuad",
                "--checkout-root",
                str(tmp_path),
                "--predictions",
                str(tmp_path / "preds.json"),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
        == 0
    )


def test_cli_record_run_accepts_matching_official_audit(tmp_path: Path) -> None:
    """A CUAD run record is accepted only when registry pins match."""
    started_at = datetime(2026, 9, 7, 8, 0, tzinfo=UTC)
    payload = benchmark_run(
        repository_revision=CUAD.repository_revision,
        evaluator_revision=CUAD.evaluator_revision,
        metrics=[MetricResult(name="aupr", value=0.41, official=True)],
        started_at=started_at,
        completed_at=started_at + timedelta(minutes=30),
    )
    run_path = write_json_report(tmp_path / "run.json", payload)
    assert main(["record-run", str(run_path)]) == 0
    mismatched = payload.model_copy(update={"evaluator_revision": "0" * 40})
    bad_path = write_json_report(tmp_path / "bad.json", mismatched)
    assert main(["record-run", str(bad_path)]) == 1
    unknown = payload.model_copy(update={"benchmark_id": "invented"})
    unknown_path = tmp_path / "unknown.json"
    unknown_path.write_text(
        json.dumps(unknown.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    assert main(["record-run", str(unknown_path)]) == 1
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{}\n", encoding="utf-8")
    assert main(["record-run", str(invalid_path)]) == 1
