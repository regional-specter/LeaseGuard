"""Choose the strongest practical unmodified base model from measured runs."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from leaseguard.evaluation.baseline_models import (
    BaselineComparisonReport,
    BaselineRegistry,
    BaselineRun,
    CandidateRanking,
    ErrorClass,
)
from leaseguard.evaluation.reports import write_json_report


def metric_value(run: BaselineRun, name: str) -> float:
    """Return one unofficial product-task metric from a baseline run."""
    for metric in run.metrics:
        if metric.name == name:
            return metric.value
    return 0.0


def dominant_error_class(run: BaselineRun) -> ErrorClass:
    """Return the most common error class, preferring failures over correct."""
    counts = dict(run.error_counts)
    counts.pop("correct", None)
    if not counts:
        return "correct"
    return cast(ErrorClass, max(counts, key=lambda name: (counts[name], name)))


def is_practical(run: BaselineRun, registry: BaselineRegistry) -> bool:
    """True when the run finished on T4 without exceeding the VRAM budget."""
    if run.oom or run.backend_kind == "scripted":
        return False
    return run.resources.peak_vram_mb <= registry.max_practical_vram_mb


def ranking_key(rank: CandidateRanking) -> tuple[object, ...]:
    """Lexicographic key: quality first, then speed, then lower memory."""
    return (
        rank.practical,
        rank.schema_validity,
        rank.extraction_f1,
        rank.answer_status_accuracy,
        rank.evidence_recall,
        rank.tokens_per_second,
        -rank.peak_vram_mb,
        rank.candidate_id,
    )


def rank_candidate(run: BaselineRun, registry: BaselineRegistry) -> CandidateRanking:
    """Summarize one required comparison run for selection."""
    practical = is_practical(run, registry)
    notes = "scripted backend; not eligible for selection"
    if run.backend_kind != "scripted":
        if run.oom:
            notes = "out of memory on the configured hardware"
        elif not practical:
            notes = "peak VRAM exceeds the practical T4 budget"
        else:
            notes = "completed the required comparison on Colab T4"
    return CandidateRanking(
        candidate_id=run.candidate_id,
        practical=practical,
        schema_validity=metric_value(run, "schema_validity"),
        extraction_f1=metric_value(run, "extraction_f1"),
        answer_status_accuracy=metric_value(run, "answer_status_accuracy"),
        evidence_recall=metric_value(run, "evidence_recall"),
        tokens_per_second=run.resources.tokens_per_second,
        peak_vram_mb=run.resources.peak_vram_mb,
        dominant_error_class=dominant_error_class(run),
        notes=notes,
    )


def fine_tuning_hypothesis(rankings: Sequence[CandidateRanking]) -> str:
    """Describe the measured weakness fine-tuning must address."""
    practical = [item for item in rankings if item.practical]
    if not practical:
        return (
            "No practical unmodified candidate completed the required T4 comparison. "
            "Fine-tuning should wait until a candidate fits Colab T4 4-bit inference, "
            "or until a smaller quantization and context window are measured."
        )
    winner = practical[0]
    if winner.dominant_error_class == "correct":
        return (
            f"{winner.candidate_id} already matches scored gold labels on the seed holdout. "
            "Fine-tuning should still target schema-valid JSON, evidence quotes, and "
            "refusals on a larger Dataset v1 corpus before any professional-benchmark claim."
        )
    mapping = {
        "parse_error": (
            "Fine-tuning should teach schema-constrained JSON so the model stops returning prose."
        ),
        "prompt_error": (
            "Direct prompting does not reliably produce JSON. Fine-tuning should lock in "
            "schema-constrained extraction and answer formats."
        ),
        "validation_error": (
            "Fine-tuning should teach ontology-valid fields, evidence spans, and enumerations."
        ),
        "retrieval_error": (
            "Missing gold evidence is a context problem. Improve clause retrieval in Phase 8 "
            "rather than asking fine-tuning to invent unseen quotes."
        ),
        "model_error": (
            "The model returns valid JSON with the wrong parties, money, dates, or answer status. "
            "Supervised fine-tuning should start with structured extraction, grounded answers, "
            "and refusal examples."
        ),
    }
    return mapping.get(
        winner.dominant_error_class,
        "Fine-tuning should target the measured dominant error class on the frozen eval split.",
    )


def compare_baseline_runs(
    runs: Sequence[BaselineRun],
    registry: BaselineRegistry,
    *,
    completed_at: datetime | None = None,
) -> BaselineComparisonReport:
    """Select the strongest practical candidate using a documented lexicographic order."""
    required = [run for run in runs if run.comparison_id == registry.required_comparison_id]
    by_candidate: dict[str, BaselineRun] = {}
    for run in required:
        previous = by_candidate.get(run.candidate_id)
        if previous is None or run.completed_at > previous.completed_at:
            by_candidate[run.candidate_id] = run
    expected = {candidate.candidate_id for candidate in registry.candidates}
    missing = sorted(expected - set(by_candidate))
    rankings = [rank_candidate(run, registry) for run in by_candidate.values()]
    rankings.sort(key=ranking_key, reverse=True)
    practical = [item for item in rankings if item.practical]
    incomplete = bool(missing) or not practical
    selected = practical[0].candidate_id if practical and not missing else None
    if missing:
        notes = "Missing required comparison runs: " + ", ".join(missing)
    elif not practical:
        notes = "Required runs exist, but no candidate is practical on Colab T4."
    else:
        notes = (
            f"Selected {selected} by lexicographic order of schema validity, extraction F1, "
            "answer status accuracy, evidence recall, speed, and lower peak VRAM. "
            "This is not a blended LeaseGuard score and not a professional-benchmark result."
        )
    return BaselineComparisonReport(
        report_id="phase6-baseline-comparison",
        eval_split_version="1.0.0",
        baseline_registry_version=registry.registry_version,
        completed_at=completed_at or datetime.now(tz=UTC),
        required_comparison_id=registry.required_comparison_id,
        selected_candidate_id=selected,
        incomplete=incomplete,
        rankings=rankings,
        fine_tuning_hypothesis=fine_tuning_hypothesis(rankings),
        notes=notes,
    )


def load_baseline_runs(paths: Sequence[Path]) -> list[BaselineRun]:
    """Read saved baseline run records from disk."""
    return [BaselineRun.model_validate_json(path.read_text(encoding="utf-8")) for path in paths]


def write_comparison_report(path: Path, report: BaselineComparisonReport) -> Path:
    """Save the Phase 6 selection report."""
    return write_json_report(path, report)
