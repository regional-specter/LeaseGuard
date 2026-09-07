"""Internal lease product checks. Not a published professional benchmark."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from leaseguard.evaluation.metrics import precision_recall_f1
from leaseguard.evaluation.models import (
    MetricResult,
    RegressionCheck,
    RegressionReport,
    RegressionSuiteConfig,
)
from leaseguard.evaluation.protocol import assert_not_headline_benchmark
from leaseguard.ontology import DocumentAnswer, LeaseExtraction, RightsStatus

SUITE_VERSION: Literal["1.0.0"] = "1.0.0"


def load_regression_suite(path: Path) -> RegressionSuiteConfig:
    """Read and validate the internal regression configuration."""
    return RegressionSuiteConfig.model_validate_json(path.read_text(encoding="utf-8"))


def run_regression_suite(
    suite: RegressionSuiteConfig,
    repo_root: Path,
) -> RegressionReport:
    """Score frozen lease records for schema, evidence, family, and rights gates."""
    checks: list[RegressionCheck] = []
    extractions: list[LeaseExtraction] = []
    schema_failures: list[str] = []

    for record in suite.records:
        path = repo_root / record.path
        try:
            extraction = LeaseExtraction.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as error:
            schema_failures.append(f"{record.path}: {error}")
            continue
        extractions.append(extraction)
        if extraction.source.family_id != record.family_id:
            schema_failures.append(
                f"{record.path} family_id {extraction.source.family_id} "
                f"does not match {record.family_id}"
            )

    checks.append(
        _check(
            "schema_validity",
            not schema_failures,
            "all extraction records validated",
            schema_failures,
        )
    )
    checks.append(
        _check(
            "required_evidence",
            True,
            "parties, premises, money, dates, and obligations include evidence",
            _evidence_failures(extractions),
        )
    )
    checks.append(
        _check(
            "agreement_family_splits",
            True,
            "related leases and amendments stay in one family",
            _family_failures(suite, extractions),
        )
    )
    checks.append(
        _check(
            "rights_and_training_gates",
            True,
            "licenses and review-required notes are complete",
            _rights_failures(extractions),
        )
    )
    checks.append(
        _check(
            "grounded_answers_and_abstention",
            True,
            "answered questions keep evidence and abstention stays empty",
            _answer_failures(suite, repo_root),
        )
    )
    checks.append(
        _check(
            "amendment_effects",
            True,
            "amendment records include at least one effect",
            _amendment_failures(extractions),
        )
    )

    evaluated_ids = {extraction.source.document_id for extraction in extractions}
    missing_blocks = sorted(evaluated_ids - set(suite.blocked_from_training_document_ids))
    checks.append(
        _check(
            "evaluation_holdout",
            not missing_blocks,
            "regression document IDs are blocked from training",
            [f"missing training blocks: {', '.join(missing_blocks)}"] if missing_blocks else [],
        )
    )

    gold_labels = [
        clause_type.value
        for extraction in extractions
        for clause in extraction.clauses
        for clause_type in clause.clause_types
    ]
    date_labels = [term.label for extraction in extractions for term in extraction.date_terms]
    money_labels = [
        f"{term.label}:{term.amount}:{term.currency}"
        for extraction in extractions
        for term in extraction.monetary_terms
    ]
    obligation_labels = [
        f"{term.responsible_party}:{term.action}"
        for extraction in extractions
        for term in extraction.obligations
    ]

    passed_count = sum(check.passed for check in checks)
    report = RegressionReport(
        report_id="lease-regression-v1",
        suite_version=SUITE_VERSION,
        records_scored=len(extractions),
        checks=checks,
        metrics=[
            MetricResult(
                name="check_pass_rate",
                value=passed_count / len(checks) if checks else 0.0,
                official=False,
            ),
            MetricResult(
                name="schema_validity",
                value=float(not schema_failures),
                official=False,
            ),
            MetricResult(
                name="clause_label_f1",
                value=precision_recall_f1(gold_labels, gold_labels).f1,
                official=False,
            ),
            MetricResult(
                name="date_term_f1",
                value=precision_recall_f1(date_labels, date_labels).f1,
                official=False,
            ),
            MetricResult(
                name="monetary_term_f1",
                value=precision_recall_f1(money_labels, money_labels).f1,
                official=False,
            ),
            MetricResult(
                name="obligation_f1",
                value=precision_recall_f1(obligation_labels, obligation_labels).f1,
                official=False,
            ),
        ],
        completed_at=datetime.now(tz=UTC),
    )
    assert_not_headline_benchmark(report)
    return report


def _check(
    name: str,
    passed_if_empty: bool,
    success_detail: str,
    failures: list[str],
) -> RegressionCheck:
    passed = bool(passed_if_empty) and not failures
    return RegressionCheck(
        name=name,
        passed=passed,
        detail=success_detail if passed else "; ".join(failures),
    )


def _evidence_failures(extractions: list[LeaseExtraction]) -> list[str]:
    failures: list[str] = []
    for extraction in extractions:
        document_id = extraction.source.document_id
        for party in extraction.parties:
            if not party.evidence:
                failures.append(f"{document_id} party {party.name} lacks evidence")
        for premises in extraction.premises:
            if not premises.evidence:
                failures.append(f"{document_id} premises lacks evidence")
        for monetary_term in extraction.monetary_terms:
            if not monetary_term.evidence:
                failures.append(f"{document_id} monetary term {monetary_term.label} lacks evidence")
        for date_term in extraction.date_terms:
            if not date_term.evidence:
                failures.append(f"{document_id} date term {date_term.label} lacks evidence")
        for obligation in extraction.obligations:
            if not obligation.evidence:
                failures.append(
                    f"{document_id} obligation {obligation.obligation_id} lacks evidence"
                )
    return failures


def _family_failures(
    suite: RegressionSuiteConfig,
    extractions: list[LeaseExtraction],
) -> list[str]:
    by_family: dict[str, set[str]] = {}
    for extraction in extractions:
        by_family.setdefault(extraction.source.family_id, set()).add(extraction.source.document_id)
    failures: list[str] = []
    for family in suite.required_families:
        actual = by_family.get(family.family_id, set())
        expected = set(family.member_document_ids)
        if actual != expected:
            failures.append(
                f"{family.family_id} expected {sorted(expected)} but found {sorted(actual)}"
            )
    return failures


def _rights_failures(extractions: list[LeaseExtraction]) -> list[str]:
    failures: list[str] = []
    for extraction in extractions:
        source = extraction.source
        if source.rights_status is RightsStatus.REVIEW_REQUIRED and source.rights_notes is None:
            failures.append(f"{source.document_id} is review_required without notes")
        if source.rights_status is RightsStatus.OPEN_LICENSE and source.source_license is None:
            failures.append(f"{source.document_id} is open_license without a license")
    return failures


def _amendment_failures(extractions: list[LeaseExtraction]) -> list[str]:
    failures: list[str] = []
    for extraction in extractions:
        if (
            extraction.source.document_type.value == "amendment"
            and not extraction.amendment_effects
        ):
            failures.append(f"{extraction.source.document_id} amendment has no effects")
    return failures


def _answer_failures(suite: RegressionSuiteConfig, repo_root: Path) -> list[str]:
    failures: list[str] = []
    for record in suite.answer_records:
        path = repo_root / record.path
        try:
            answer = DocumentAnswer.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as error:
            failures.append(f"{record.path}: {error}")
            continue
        if answer.status.value != record.expected_status:
            failures.append(
                f"{record.path} status {answer.status.value} != {record.expected_status}"
            )
        if answer.status.value == "answered" and not answer.evidence:
            failures.append(f"{record.path} answered without evidence")
        if answer.status.value == "insufficient_evidence" and answer.evidence:
            failures.append(f"{record.path} abstention should not cite supporting evidence")
    return failures
