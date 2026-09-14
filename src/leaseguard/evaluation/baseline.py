"""Run one frozen unmodified base-model comparison on Dataset v1 eval."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from leaseguard.dataset.config import load_dataset_config
from leaseguard.dataset.hashing import sha256_json
from leaseguard.dataset.paths import default_seed_examples_path
from leaseguard.evaluation.baseline_models import (
    BaselinePrediction,
    BaselineRegistry,
    BaselineRun,
    CandidateConfig,
    ComparisonSpec,
    EvalSplitConfig,
)
from leaseguard.evaluation.errors import (
    classify_error,
    validate_answer_payload,
    validate_extraction_payload,
)
from leaseguard.evaluation.eval_split import assert_eval_split_matches_dataset, load_eval_split
from leaseguard.evaluation.paths import default_baselines_path
from leaseguard.evaluation.reports import write_json_report
from leaseguard.evaluation.scoring import (
    answer_status_correct,
    content_is_correct,
    extraction_label_f1,
    summarize_metrics,
)
from leaseguard.evaluation.tasks import BaselineTask, load_baseline_tasks
from leaseguard.inference.backend import GenerationBackend
from leaseguard.inference.context import build_context, gold_evidence_in_context
from leaseguard.inference.parse import extract_json_object
from leaseguard.inference.prompts import (
    load_prompt_catalog,
    render_answer_prompt,
    render_extraction_prompt,
    template_sha256,
)
from leaseguard.inference.resources import (
    aggregate_resources,
    empty_resources,
    resources_from_generation,
)
from leaseguard.ontology.models import DocumentAnswer, LeaseExtraction


class BaselineConfigError(ValueError):
    """Raised when a candidate or comparison is missing from the frozen registry."""


def load_baseline_registry(path: Path | None = None) -> BaselineRegistry:
    """Read the frozen unmodified base-model registry."""
    registry_path = path or default_baselines_path()
    return BaselineRegistry.model_validate_json(registry_path.read_text(encoding="utf-8"))


def get_candidate(registry: BaselineRegistry, candidate_id: str) -> CandidateConfig:
    """Return one candidate by its stable identifier."""
    for candidate in registry.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise BaselineConfigError(f"unknown candidate_id: {candidate_id}")


def get_comparison(registry: BaselineRegistry, comparison_id: str) -> ComparisonSpec:
    """Return one comparison by its stable identifier."""
    for comparison in registry.comparisons:
        if comparison.comparison_id == comparison_id:
            return comparison
    raise BaselineConfigError(f"unknown comparison_id: {comparison_id}")


def run_id_for(candidate_id: str, comparison_id: str) -> str:
    """Build a stable run identifier from the frozen matrix."""
    return f"{candidate_id}-{comparison_id}"


def prediction_checkpoint_path(checkpoint_dir: Path, run_id: str) -> Path:
    """Return the resumable JSONL file for one baseline run."""
    return checkpoint_dir / f"{run_id}.predictions.jsonl"


def load_prediction_checkpoint(path: Path) -> dict[str, BaselinePrediction]:
    """Reload completed predictions so a Colab session can resume."""
    if not path.is_file():
        return {}
    loaded: dict[str, BaselinePrediction] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        prediction = BaselinePrediction.model_validate_json(line)
        loaded[prediction.record_id] = prediction
    return loaded


def append_prediction_checkpoint(path: Path, prediction: BaselinePrediction) -> None:
    """Append one finished example so a disconnect does not lose GPU work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(prediction.model_dump_json() + "\n")


def run_baseline(
    *,
    candidate_id: str,
    comparison_id: str,
    backend: GenerationBackend,
    examples_root: Path | None = None,
    output_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    registry: BaselineRegistry | None = None,
    split: EvalSplitConfig | None = None,
    hardware: str | None = None,
    limit: int | None = None,
) -> BaselineRun:
    """Evaluate one candidate on one frozen comparison and write a run record."""
    registry = registry or load_baseline_registry()
    split = split or load_eval_split()
    assert_eval_split_matches_dataset(split, load_dataset_config())
    candidate = get_candidate(registry, candidate_id)
    comparison = get_comparison(registry, comparison_id)
    if candidate_id not in comparison.candidate_ids:
        raise BaselineConfigError(f"{candidate_id} is not part of comparison {comparison_id}")
    catalog = load_prompt_catalog()
    tasks = load_baseline_tasks(examples_root or default_seed_examples_path(), split)
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise BaselineConfigError("no Dataset v1 evaluation tasks were found")

    run_id = run_id_for(candidate_id, comparison_id)
    checkpoint_path = (
        prediction_checkpoint_path(checkpoint_dir, run_id) if checkpoint_dir is not None else None
    )
    completed = load_prediction_checkpoint(checkpoint_path) if checkpoint_path is not None else {}
    hardware_name = hardware or registry.hardware
    started_at = datetime.now(tz=UTC)
    predictions: list[BaselinePrediction] = []
    oom = False
    for task in tasks:
        if task.record_id in completed:
            predictions.append(completed[task.record_id])
            continue
        try:
            prediction = _predict_one(
                task,
                candidate=candidate,
                comparison=comparison,
                backend=backend,
                catalog=catalog,
                retrieval=registry.retrieval,
                generation=registry.generation,
                hardware=hardware_name,
            )
        except MemoryError:
            oom = True
            prediction = _oom_prediction(task, comparison, hardware_name, candidate)
        predictions.append(prediction)
        if checkpoint_path is not None:
            append_prediction_checkpoint(checkpoint_path, prediction)
        if oom:
            break

    completed_at = datetime.now(tz=UTC)
    run = _build_run(
        run_id=run_id,
        candidate=candidate,
        comparison=comparison,
        backend=backend,
        split=split,
        catalog_sha=template_sha256(
            catalog,
            ("direct_extraction", "schema_extraction", "direct_answer", "schema_answer"),
        ),
        hardware=hardware_name,
        seed=registry.random_seed,
        started_at=started_at,
        completed_at=completed_at,
        oom=oom,
        predictions=predictions,
    )
    if output_dir is not None:
        write_json_report(output_dir / f"{run_id}.json", run)
    return run


def _predict_one(
    task: BaselineTask,
    *,
    candidate: CandidateConfig,
    comparison: ComparisonSpec,
    backend: GenerationBackend,
    catalog: Any,
    retrieval: Any,
    generation: Any,
    hardware: str,
) -> BaselinePrediction:
    context = build_context(
        task.source_texts,
        mode=comparison.context_mode,
        query=task.query,
        retrieval=retrieval,
    )
    evidence_present = gold_evidence_in_context(list(task.gold_evidence_texts), context.text)
    if task.subset == "structured_extraction":
        prompt = render_extraction_prompt(
            catalog,
            prompt_mode=comparison.prompt_mode,
            record_id=task.record_id,
            document_id=task.document_id,
            family_id=task.family_id,
            context=context.text,
        )
    else:
        prompt = render_answer_prompt(
            catalog,
            prompt_mode=comparison.prompt_mode,
            record_id=task.record_id,
            perspective=task.perspective,
            question=task.query,
            context=context.text,
        )
    result = backend.generate(prompt, generation)
    resources = resources_from_generation(
        result,
        hardware=hardware,
        context_length_limit=candidate.practical_context_length,
    )
    parsed = extract_json_object(result.text)
    predicted_extraction: LeaseExtraction | None = None
    predicted_answer: DocumentAnswer | None = None
    injected = False
    working: dict[str, Any] | None = parsed
    schema_valid = False
    if parsed is not None and task.subset == "structured_extraction":
        predicted_extraction, injected, working = validate_extraction_payload(
            parsed,
            document_id=task.document_id,
            family_id=task.family_id,
        )
        schema_valid = predicted_extraction is not None
    elif parsed is not None:
        predicted_answer, working = validate_answer_payload(
            parsed,
            question=task.query,
            perspective=task.perspective,
        )
        schema_valid = predicted_answer is not None
    content_score = 0.0
    if predicted_extraction is not None and task.gold_extraction is not None:
        content_score = extraction_label_f1(predicted_extraction, task.gold_extraction)
    elif predicted_answer is not None and task.gold_answer is not None:
        content_score = float(answer_status_correct(predicted_answer, task.gold_answer))
    correct = content_is_correct(
        predicted_extraction=predicted_extraction,
        gold_extraction=task.gold_extraction,
        predicted_answer=predicted_answer,
        gold_answer=task.gold_answer,
    )
    error_class = classify_error(
        prompt_mode=comparison.prompt_mode,
        parsed=parsed,
        schema_valid=schema_valid,
        gold_evidence_in_context=evidence_present,
        content_correct=correct,
    )
    return BaselinePrediction(
        record_id=task.record_id,
        family_id=task.family_id,
        subset=task.subset,
        prompt_mode=comparison.prompt_mode,
        context_mode=comparison.context_mode,
        quantization=comparison.quantization,
        error_class=error_class,
        schema_valid=schema_valid,
        content_score=content_score,
        gold_evidence_in_context=evidence_present,
        source_identity_injected=injected,
        raw_output=result.text,
        parsed_payload=working,
        detail=_detail(error_class, schema_valid, evidence_present, correct),
        resources=resources,
    )


def _oom_prediction(
    task: BaselineTask,
    comparison: ComparisonSpec,
    hardware: str,
    candidate: CandidateConfig,
) -> BaselinePrediction:
    return BaselinePrediction(
        record_id=task.record_id,
        family_id=task.family_id,
        subset=task.subset,
        prompt_mode=comparison.prompt_mode,
        context_mode=comparison.context_mode,
        quantization=comparison.quantization,
        error_class="model_error",
        schema_valid=False,
        content_score=0.0,
        gold_evidence_in_context=False,
        raw_output="",
        parsed_payload=None,
        detail="generation raised MemoryError; candidate is not practical on this hardware",
        resources=empty_resources(hardware, candidate.practical_context_length),
    )


def _detail(
    error_class: str,
    schema_valid: bool,
    evidence_present: bool,
    correct: bool,
) -> str:
    if error_class == "correct":
        return "schema-valid output matched the scored gold labels"
    if error_class == "prompt_error":
        return "direct prompting did not return a JSON object"
    if error_class == "parse_error":
        return "schema-constrained prompting did not return a JSON object"
    if error_class == "retrieval_error":
        return "gold evidence was missing from the supplied context"
    if error_class == "validation_error":
        return "JSON was returned but failed ontology validation"
    if not evidence_present:
        return "context missed gold evidence and the labels were wrong"
    if not schema_valid:
        return "the model returned invalid structured output"
    if not correct:
        return "schema-valid output disagreed with the gold labels"
    return "unclassified product-task error"


def _build_run(
    *,
    run_id: str,
    candidate: CandidateConfig,
    comparison: ComparisonSpec,
    backend: GenerationBackend,
    split: EvalSplitConfig,
    catalog_sha: str,
    hardware: str,
    seed: int,
    started_at: datetime,
    completed_at: datetime,
    oom: bool,
    predictions: list[BaselinePrediction],
) -> BaselineRun:
    schema_valid_count = sum(item.schema_valid for item in predictions)
    extraction_predictions = [
        item for item in predictions if item.subset == "structured_extraction"
    ]
    answer_predictions = [
        item
        for item in predictions
        if item.subset in {"evidence_conversation", "refusal_uncertainty"}
    ]
    extraction_scores = [item.content_score for item in extraction_predictions]
    answer_count = len(answer_predictions)
    answer_correct = sum(item.error_class == "correct" for item in answer_predictions)
    evidence_scores = [1.0 if item.gold_evidence_in_context else 0.0 for item in predictions]
    metrics = summarize_metrics(
        schema_valid_count=schema_valid_count,
        extraction_scores=extraction_scores,
        answer_correct_count=answer_correct,
        answer_count=answer_count,
        evidence_scores=evidence_scores,
        total=len(predictions),
    )
    error_counts: dict[str, int] = {}
    for prediction in predictions:
        error_counts[prediction.error_class] = error_counts.get(prediction.error_class, 0) + 1
    resources = aggregate_resources(
        [item.resources for item in predictions],
        hardware=hardware,
        context_length_limit=candidate.practical_context_length,
    )
    backend_kind: Literal["scripted", "transformers"] = (
        "transformers" if backend.kind == "transformers" else "scripted"
    )
    return BaselineRun(
        run_id=run_id,
        candidate_id=candidate.candidate_id,
        comparison_id=comparison.comparison_id,
        model_id=candidate.model_id,
        model_revision=candidate.revision,
        resolved_revision=backend.resolved_revision,
        dataset_version="1.0.0",
        eval_split_version=split.split_version,
        prompt_version="1.0.0",
        prompt_template_sha256=catalog_sha,
        backend_kind=backend_kind,
        hardware=hardware,
        quantization=comparison.quantization,
        prompt_mode=comparison.prompt_mode,
        context_mode=comparison.context_mode,
        random_seed=seed,
        started_at=started_at,
        completed_at=completed_at,
        oom=oom,
        contamination_notes=(
            "Public benchmark and lease-template exposure during publisher pretraining is unknown. "
            "Dataset v1 product-task scores are not professional-benchmark results."
        ),
        metrics=metrics,
        error_counts=error_counts,
        resources=resources,
        predictions=predictions,
    )


def write_run_manifest(path: Path, run: BaselineRun) -> Path:
    """Write a tiny companion manifest next to the full run record."""
    payload = {
        "run_id": run.run_id,
        "candidate_id": run.candidate_id,
        "comparison_id": run.comparison_id,
        "backend_kind": run.backend_kind,
        "oom": run.oom,
        "metrics": [metric.model_dump() for metric in run.metrics],
        "checksum": sha256_json(run.model_dump(mode="json")),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
