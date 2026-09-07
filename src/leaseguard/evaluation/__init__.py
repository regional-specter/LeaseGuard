"""Repeatable quality, grounding, and safety evaluation."""

from leaseguard.evaluation.cli import main as evaluation_cli
from leaseguard.evaluation.metrics import (
    CharacterSpan,
    ClassificationMetrics,
    RetrievalMetrics,
    area_under_pr_curve,
    character_retrieval_metrics,
    merge_character_spans,
    precision_at_recall,
    precision_recall_f1,
    ranked_precision_recall,
)
from leaseguard.evaluation.models import (
    BenchmarkRegistry,
    BenchmarkRun,
    BenchmarkSpec,
    MetricResult,
    RegressionReport,
    RegressionSuiteConfig,
)
from leaseguard.evaluation.official import (
    OfficialEvaluatorDownloadError,
    OfficialEvaluatorNotAvailableError,
    verify_official_checkout,
)
from leaseguard.evaluation.protocol import (
    HeadlineClaimError,
    HoldoutIntegrityError,
    bind_run_to_spec,
)
from leaseguard.evaluation.registry import get_benchmark, load_benchmark_registry
from leaseguard.evaluation.regression import load_regression_suite, run_regression_suite

__all__ = [
    "BenchmarkRegistry",
    "BenchmarkRun",
    "BenchmarkSpec",
    "CharacterSpan",
    "ClassificationMetrics",
    "HeadlineClaimError",
    "HoldoutIntegrityError",
    "MetricResult",
    "OfficialEvaluatorDownloadError",
    "OfficialEvaluatorNotAvailableError",
    "RegressionReport",
    "RegressionSuiteConfig",
    "RetrievalMetrics",
    "area_under_pr_curve",
    "bind_run_to_spec",
    "character_retrieval_metrics",
    "evaluation_cli",
    "get_benchmark",
    "load_benchmark_registry",
    "load_regression_suite",
    "merge_character_spans",
    "precision_at_recall",
    "precision_recall_f1",
    "ranked_precision_recall",
    "run_regression_suite",
    "verify_official_checkout",
]
