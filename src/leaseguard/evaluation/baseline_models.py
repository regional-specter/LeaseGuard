"""Frozen baseline configurations and product-task run records."""

from datetime import date, datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from leaseguard.evaluation.models import EvaluationModel, MetricResult

PromptMode = Literal["direct", "schema_constrained"]
ContextMode = Literal["full_document", "clause_level", "retrieval_assisted"]
QuantizationName = Literal["4bit", "8bit"]
BackendKind = Literal["scripted", "transformers"]
ErrorClass = Literal[
    "correct",
    "model_error",
    "parse_error",
    "validation_error",
    "prompt_error",
    "retrieval_error",
]
BaselineTaskKind = Literal["structured_extraction", "evidence_conversation", "refusal_uncertainty"]


class SeedFamilySplit(EvaluationModel):
    """Pinned family-hash outcome for one Dataset v1 seed family."""

    family_id: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]


class EvalSplitConfig(EvaluationModel):
    """Frozen Dataset v1 evaluation split used for unmodified base-model runs."""

    split_version: Literal["1.0.0"]
    frozen_on: date
    dataset_version: Literal["1.0.0"]
    dataset_config: str = Field(min_length=1)
    method: Literal["family_hash"]
    salt: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    headline_claim_allowed: Literal[False] = False
    scored_subsets: list[BaselineTaskKind] = Field(min_length=1)
    seed_families: list[SeedFamilySplit] = Field(min_length=1)
    seed_eval_record_ids: list[str] = Field(min_length=1)
    professional_benchmarks: list[str] = Field(min_length=1)
    blocked_from_training_document_ids: list[str] = Field(min_length=1)
    forbidden_uses: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def identifiers_are_unique(self) -> Self:
        """Keep family and record lists unambiguous."""
        families = [item.family_id for item in self.seed_families]
        if len(families) != len(set(families)):
            raise ValueError("seed family_id values must be unique")
        if len(self.seed_eval_record_ids) != len(set(self.seed_eval_record_ids)):
            raise ValueError("seed_eval_record_ids must be unique")
        if len(self.scored_subsets) != len(set(self.scored_subsets)):
            raise ValueError("scored_subsets must be unique")
        return self


class CandidateConfig(EvaluationModel):
    """One unmodified publisher checkpoint evaluated on Colab T4."""

    candidate_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    license: str = Field(min_length=1)
    parameter_count: str = Field(min_length=1)
    native_context_length: int = Field(ge=1024)
    practical_context_length: int = Field(ge=512)
    quantizations: list[QuantizationName] = Field(min_length=1)
    gated: bool
    notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def practical_context_fits_native(self) -> Self:
        """Refuse a T4 window larger than the published native context."""
        if self.practical_context_length > self.native_context_length:
            raise ValueError("practical_context_length cannot exceed native_context_length")
        if len(self.quantizations) != len(set(self.quantizations)):
            raise ValueError("quantizations must be unique")
        return self


class ComparisonSpec(EvaluationModel):
    """One prompt, context, and quantization combination."""

    comparison_id: str = Field(min_length=1)
    prompt_mode: PromptMode
    context_mode: ContextMode
    quantization: QuantizationName
    candidate_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def candidate_ids_are_unique(self) -> Self:
        """Keep the comparison matrix unambiguous."""
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("comparison candidate_ids must be unique")
        return self


class GenerationSettings(EvaluationModel):
    """Deterministic generation limits shared by every candidate."""

    max_new_tokens: int = Field(ge=16, le=4096)
    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(gt=0, le=1)
    do_sample: bool = False


class RetrievalSettings(EvaluationModel):
    """Keyword-retrieval limits for retrieval-assisted context."""

    top_k: int = Field(ge=1, le=32)
    min_score: float = Field(ge=0, le=1)


class BaselineRegistry(EvaluationModel):
    """Versioned set of unmodified base-model experiments."""

    registry_version: Literal["1.0.0"]
    frozen_on: date
    hardware: Literal["Google Colab T4"]
    purpose: str = Field(min_length=1)
    headline_claim_allowed: Literal[False] = False
    random_seed: int
    required_comparison_id: str = Field(min_length=1)
    max_practical_vram_mb: int = Field(ge=1024)
    candidates: list[CandidateConfig] = Field(min_length=1)
    comparisons: list[ComparisonSpec] = Field(min_length=1)
    generation: GenerationSettings
    retrieval: RetrievalSettings

    @model_validator(mode="after")
    def matrix_is_internally_consistent(self) -> Self:
        """Require unique IDs and comparisons that only name known candidates."""
        candidate_ids = [item.candidate_id for item in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_id values must be unique")
        comparison_ids = [item.comparison_id for item in self.comparisons]
        if len(comparison_ids) != len(set(comparison_ids)):
            raise ValueError("comparison_id values must be unique")
        if self.required_comparison_id not in comparison_ids:
            raise ValueError("required_comparison_id must name a comparison")
        known = set(candidate_ids)
        by_candidate = {item.candidate_id: item for item in self.candidates}
        for comparison in self.comparisons:
            unknown = [item for item in comparison.candidate_ids if item not in known]
            if unknown:
                raise ValueError("unknown comparison candidate_ids: " + ", ".join(unknown))
            for candidate_id in comparison.candidate_ids:
                allowed = by_candidate[candidate_id].quantizations
                if comparison.quantization not in allowed:
                    raise ValueError(
                        f"{comparison.comparison_id} quantization {comparison.quantization} "
                        f"is not allowed for {candidate_id}"
                    )
        return self


class PromptCatalog(EvaluationModel):
    """Frozen prompt templates used by every baseline comparison."""

    prompt_version: Literal["1.0.0"]
    frozen_on: date
    templates: dict[str, str] = Field(min_length=1)


class ResourceMetrics(EvaluationModel):
    """Runtime cost of one generation or one completed run."""

    hardware: str = Field(min_length=1)
    peak_vram_mb: float = Field(ge=0)
    latency_seconds: float = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    tokens_per_second: float = Field(ge=0)
    context_length_limit: int = Field(ge=1)


class BaselinePrediction(EvaluationModel):
    """One product-task example scored for an unmodified base model."""

    record_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    subset: BaselineTaskKind
    prompt_mode: PromptMode
    context_mode: ContextMode
    quantization: QuantizationName
    error_class: ErrorClass
    schema_valid: bool
    content_score: float = Field(ge=0, le=1)
    gold_evidence_in_context: bool
    source_identity_injected: bool = False
    raw_output: str
    parsed_payload: dict[str, object] | None = None
    detail: str = Field(min_length=1)
    resources: ResourceMetrics


class BaselineRun(EvaluationModel):
    """Audit record for one candidate on one comparison using Dataset v1 eval."""

    run_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    comparison_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    resolved_revision: str | None = Field(default=None, min_length=1)
    dataset_version: Literal["1.0.0"]
    eval_split_version: Literal["1.0.0"]
    prompt_version: Literal["1.0.0"]
    prompt_template_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    backend_kind: BackendKind
    hardware: str = Field(min_length=1)
    quantization: QuantizationName
    prompt_mode: PromptMode
    context_mode: ContextMode
    random_seed: int
    started_at: datetime
    completed_at: datetime
    oom: bool = False
    headline_claim_allowed: Literal[False] = False
    paid_api_used: Literal[False] = False
    contamination_notes: str = Field(min_length=1)
    metrics: list[MetricResult] = Field(min_length=1)
    error_counts: dict[str, int]
    resources: ResourceMetrics
    predictions: list[BaselinePrediction] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        """Reject impossible times, unofficial headline metrics, and duplicate IDs."""
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be earlier than started_at")
        record_ids = [item.record_id for item in self.predictions]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("prediction record_id values must be unique")
        metric_names = [metric.name for metric in self.metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("metric names must be unique")
        if any(metric.official for metric in self.metrics):
            raise ValueError("baseline product-task metrics cannot be marked official")
        return self


class CandidateRanking(EvaluationModel):
    """Lexicographic rank for one practical candidate. Not a blended score."""

    candidate_id: str = Field(min_length=1)
    practical: bool
    schema_validity: float = Field(ge=0, le=1)
    extraction_f1: float = Field(ge=0, le=1)
    answer_status_accuracy: float = Field(ge=0, le=1)
    evidence_recall: float = Field(ge=0, le=1)
    tokens_per_second: float = Field(ge=0)
    peak_vram_mb: float = Field(ge=0)
    dominant_error_class: ErrorClass
    notes: str = Field(min_length=1)


class BaselineComparisonReport(EvaluationModel):
    """Measured choice of the strongest practical unmodified base model."""

    report_id: str = Field(min_length=1)
    eval_split_version: Literal["1.0.0"]
    baseline_registry_version: Literal["1.0.0"]
    headline_claim_allowed: Literal[False] = False
    completed_at: datetime
    required_comparison_id: str = Field(min_length=1)
    selected_candidate_id: str | None = Field(default=None, min_length=1)
    incomplete: bool
    rankings: list[CandidateRanking]
    fine_tuning_hypothesis: str = Field(min_length=1)
    notes: str = Field(min_length=1)
