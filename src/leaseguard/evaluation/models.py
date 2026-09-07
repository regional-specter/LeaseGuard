"""Reproducible records for external benchmark runs."""

from datetime import date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluationModel(BaseModel):
    """Shared strict behavior for evaluation records."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class BenchmarkSpec(EvaluationModel):
    """Pinned identity and official protocol for one published benchmark."""

    benchmark_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    paper_url: str = Field(min_length=1)
    repository_url: str = Field(min_length=1)
    repository_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    evaluator_repository_url: str = Field(min_length=1)
    evaluator_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    evaluator_entrypoint: str = Field(min_length=1)
    license: str = Field(min_length=1)
    test_split: str = Field(min_length=1)
    official_metrics: list[str] = Field(min_length=1)
    development_variant: str | None = Field(default=None, min_length=1)
    final_run_required: bool
    headline_claim_allowed: Literal[True] = True


class BenchmarkRegistry(EvaluationModel):
    """Versioned set of approved professional benchmarks."""

    registry_version: Literal["1.0.0"]
    frozen_on: date
    policy: str = Field(min_length=1)
    benchmarks: list[BenchmarkSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def benchmark_ids_are_unique(self) -> Self:
        """Prevent one benchmark identifier from selecting two protocols."""
        benchmark_ids = [benchmark.benchmark_id for benchmark in self.benchmarks]
        if len(benchmark_ids) != len(set(benchmark_ids)):
            raise ValueError("benchmark_id values must be unique")
        return self


class MetricResult(EvaluationModel):
    """One score normalized to a zero-to-one scale."""

    name: str = Field(min_length=1)
    value: float = Field(ge=0, le=1)
    official: bool = False


class BenchmarkRun(EvaluationModel):
    """Audit record for one model evaluated on one benchmark."""

    run_id: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    benchmark_registry_version: Literal["1.0.0"]
    repository_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    adapter_revision: str | None = Field(default=None, min_length=1)
    dataset_revision: str = Field(min_length=1)
    evaluator_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    split: str = Field(min_length=1)
    hardware: str = Field(min_length=1)
    quantization: str | None = Field(default=None, min_length=1)
    prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    random_seed: int
    started_at: datetime
    completed_at: datetime
    metrics: list[MetricResult] = Field(min_length=1)
    raw_predictions_uri: str = Field(min_length=1)
    used_official_evaluator: Literal[True]
    paid_api_used: Literal[False] = False
    test_data_used_for_development: Literal[False] = False
    contamination_notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        """Reject invalid times and duplicate metric names."""
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be earlier than started_at")
        metric_names = [metric.name for metric in self.metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("metric names must be unique")
        unofficial = [metric.name for metric in self.metrics if not metric.official]
        if unofficial:
            raise ValueError(
                "external benchmark runs cannot include unofficial metrics: "
                + ", ".join(unofficial)
            )
        return self


class RegressionCheck(EvaluationModel):
    """One pass or fail result from the internal lease regression suite."""

    name: str = Field(min_length=1)
    passed: bool
    detail: str = Field(min_length=1)


class RegressionReport(EvaluationModel):
    """Internal product checks. This is not a published professional benchmark."""

    report_id: str = Field(min_length=1)
    suite_version: Literal["1.0.0"]
    is_professional_benchmark: Literal[False] = False
    headline_claim_allowed: Literal[False] = False
    records_scored: int = Field(ge=0)
    checks: list[RegressionCheck] = Field(min_length=1)
    metrics: list[MetricResult] = Field(min_length=1)
    completed_at: datetime

    @model_validator(mode="after")
    def reject_professional_claim(self) -> Self:
        """Keep the internal suite from being described as a lab benchmark."""
        if any(metric.official for metric in self.metrics):
            raise ValueError("regression metrics cannot be marked official")
        return self


class RegressionRecordSpec(EvaluationModel):
    """One internal lease record used only for product regression."""

    path: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    training_eligible: Literal[False] = False


class RegressionFamilySpec(EvaluationModel):
    """Documents that must stay together in the same split."""

    family_id: str = Field(min_length=1)
    member_document_ids: list[str] = Field(min_length=1)


class RegressionAnswerSpec(EvaluationModel):
    """An evidence-grounded answer used to check abstention behavior."""

    path: str = Field(min_length=1)
    expected_status: str = Field(min_length=1)


class RegressionSuiteConfig(EvaluationModel):
    """Frozen internal regression suite. Never a professional benchmark."""

    suite_version: Literal["1.0.0"]
    frozen_on: date
    is_professional_benchmark: Literal[False] = False
    purpose: str = Field(min_length=1)
    records: list[RegressionRecordSpec] = Field(min_length=1)
    required_families: list[RegressionFamilySpec] = Field(min_length=1)
    blocked_from_training_document_ids: list[str] = Field(min_length=1)
    answer_records: list[RegressionAnswerSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def blocked_ids_are_unique(self) -> Self:
        """Keep the holdout list unambiguous."""
        blocked = self.blocked_from_training_document_ids
        if len(blocked) != len(set(blocked)):
            raise ValueError("blocked_from_training_document_ids must be unique")
        return self
