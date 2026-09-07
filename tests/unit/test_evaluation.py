"""Tests for benchmark integrity records and local metric helpers."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from leaseguard.evaluation import (
    BenchmarkRegistry,
    BenchmarkRun,
    CharacterSpan,
    MetricResult,
    character_retrieval_metrics,
    get_benchmark,
    load_benchmark_registry,
    merge_character_spans,
    precision_recall_f1,
)
from leaseguard.evaluation.metrics import (
    area_under_pr_curve,
    precision_at_recall,
    ranked_precision_recall,
)

REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "evaluation" / "benchmarks.v1.json"
)


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


def test_professional_benchmark_registry_is_pinned() -> None:
    """The registry selects fixed external protocols rather than local inventions."""
    registry = load_benchmark_registry(REGISTRY_PATH)

    assert {benchmark.benchmark_id for benchmark in registry.benchmarks} == {
        "legalbench",
        "cuad",
        "contractnli",
        "legalbench-rag",
    }
    assert get_benchmark(registry, "cuad").final_run_required is True
    assert get_benchmark(registry, "legalbench").venue == "NeurIPS 2023 Datasets and Benchmarks"
    assert get_benchmark(registry, "cuad").evaluator_entrypoint == "evaluate.py"
    assert (
        get_benchmark(registry, "contractnli").evaluator_repository_url
        == "https://github.com/stanfordnlp/contract-nli-bert"
    )
    with pytest.raises(KeyError, match="Unknown benchmark_id"):
        get_benchmark(registry, "random-private-benchmark")


def test_registry_rejects_duplicate_benchmark_ids() -> None:
    """One stable identifier cannot point to two benchmark protocols."""
    registry = load_benchmark_registry(REGISTRY_PATH)
    data = registry.model_dump()
    data["benchmarks"].append(data["benchmarks"][0])

    with pytest.raises(ValidationError, match="must be unique"):
        BenchmarkRegistry.model_validate(data)


def test_label_metrics_cover_matches_misses_and_empty_sets() -> None:
    """Local label metrics use explicit and predictable empty-set behavior."""
    partial = precision_recall_f1(["rent", "term"], ["rent", "notice"])
    assert partial.precision == 0.5
    assert partial.recall == 0.5
    assert partial.f1 == 0.5
    assert precision_recall_f1([], []).f1 == 1.0
    assert precision_recall_f1([], ["rent"]).f1 == 0.0


def test_character_spans_validate_and_merge() -> None:
    """Overlapping characters are counted only once."""
    merged = merge_character_spans(
        [
            CharacterSpan(8, 15),
            CharacterSpan(0, 10),
            CharacterSpan(20, 25),
            CharacterSpan(25, 30),
        ]
    )
    assert merged == (CharacterSpan(0, 15), CharacterSpan(20, 30))
    assert merge_character_spans([]) == ()
    assert CharacterSpan(2, 5).length == 3

    with pytest.raises(ValueError, match="cannot be negative"):
        CharacterSpan(-1, 2)
    with pytest.raises(ValueError, match="greater than"):
        CharacterSpan(2, 2)


def test_character_retrieval_metrics_measure_exact_overlap() -> None:
    """Retrieval precision and recall operate on exact character coverage."""
    result = character_retrieval_metrics(
        [CharacterSpan(0, 10), CharacterSpan(8, 15)],
        [CharacterSpan(5, 20)],
    )
    assert result.predicted_characters == 15
    assert result.reference_characters == 15
    assert result.overlapping_characters == 10
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)

    empty = character_retrieval_metrics([], [])
    assert empty.precision == 1.0
    assert empty.recall == 1.0


def test_benchmark_run_enforces_integrity_fields() -> None:
    """A valid run records official evaluation and the no-paid-API policy."""
    run = benchmark_run()

    assert run.used_official_evaluator is True
    assert run.paid_api_used is False
    assert run.test_data_used_for_development is False


def test_benchmark_run_rejects_invalid_time_and_duplicate_metrics() -> None:
    """Run records cannot hide impossible times or duplicate score names."""
    started_at = datetime(2026, 9, 7, 8, 0, tzinfo=UTC)
    with pytest.raises(ValidationError, match="cannot be earlier"):
        benchmark_run(
            started_at=started_at,
            completed_at=started_at - timedelta(seconds=1),
        )
    with pytest.raises(ValidationError, match="metric names must be unique"):
        benchmark_run(
            metrics=[
                MetricResult(name="aupr", value=0.5, official=True),
                MetricResult(name="aupr", value=0.6, official=True),
            ]
        )
    with pytest.raises(ValidationError, match="unofficial metrics"):
        benchmark_run(metrics=[MetricResult(name="aupr", value=0.5, official=False)])


def test_ranked_retrieval_helpers_match_expected_curve() -> None:
    """Local AUPR helpers are deterministic and refuse invalid inputs."""
    precisions, recalls = ranked_precision_recall([True, False, True])
    assert precisions == [1.0, 0.5, pytest.approx(2 / 3)]
    assert recalls == [0.5, 0.5, 1.0]
    assert precision_at_recall(precisions, recalls, 0.8) == pytest.approx(2 / 3)
    assert precision_at_recall(precisions, recalls, 1.0) == pytest.approx(2 / 3)
    assert area_under_pr_curve(precisions, recalls) == pytest.approx(5 / 12)
    assert ranked_precision_recall([]) == ([], [])
    assert area_under_pr_curve([], []) == 0.0
    assert precision_at_recall([1.0], [0.5], 0.9) == 0.0

    with pytest.raises(ValueError, match="target_recall"):
        precision_at_recall([1.0], [1.0], 0.0)
    with pytest.raises(ValueError, match="same length"):
        precision_at_recall([1.0], [1.0, 0.5], 0.5)
    with pytest.raises(ValueError, match="same length"):
        area_under_pr_curve([1.0], [1.0, 0.5])
