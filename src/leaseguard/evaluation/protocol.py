"""Rules that keep published claims tied to official professional benchmarks."""

from collections.abc import Iterable

from leaseguard.evaluation.models import BenchmarkRun, BenchmarkSpec, RegressionReport

DEVELOPMENT_PURPOSES = frozenset(
    {
        "fine-tuning",
        "continued-pretraining",
        "preference-data",
        "synthetic-prompts",
        "prompt-selection",
        "hyperparameter-selection",
        "error-driven-training",
    }
)

TEST_SPLIT_ALIASES = frozenset({"test", "official test", "official_test", "held-out-test"})


class HoldoutIntegrityError(ValueError):
    """Raised when official test data would leak into development."""


class HeadlineClaimError(ValueError):
    """Raised when an internal score is treated as a professional result."""


def normalize_split(split: str) -> str:
    """Compare split names without punctuation differences."""
    return split.strip().lower().replace("_", " ").replace("-", " ")


def assert_split_allowed_for_purpose(split: str, purpose: str) -> None:
    """Forbid using official test examples for training or prompt search."""
    if (
        normalize_split(split) in {normalize_split(alias) for alias in TEST_SPLIT_ALIASES}
        and purpose in DEVELOPMENT_PURPOSES
    ):
        raise HoldoutIntegrityError(f"the official test split cannot be used for {purpose}")


def assert_training_ids_are_clean(
    training_document_ids: Iterable[str],
    blocked_document_ids: Iterable[str],
) -> None:
    """Reject training records that belong to the frozen evaluation set."""
    blocked = set(blocked_document_ids)
    overlap = sorted(set(training_document_ids) & blocked)
    if overlap:
        raise HoldoutIntegrityError(
            "evaluation documents cannot enter training: " + ", ".join(overlap)
        )


def bind_run_to_spec(run: BenchmarkRun, spec: BenchmarkSpec) -> None:
    """Require a published run to use the frozen dataset and evaluator pins."""
    if run.benchmark_id != spec.benchmark_id:
        raise HeadlineClaimError(f"run {run.run_id} does not match benchmark {spec.benchmark_id}")
    if run.repository_revision != spec.repository_revision:
        raise HeadlineClaimError("dataset revision must match the frozen registry")
    if run.evaluator_revision != spec.evaluator_revision:
        raise HeadlineClaimError("evaluator revision must match the frozen registry")
    if not run.used_official_evaluator:
        raise HeadlineClaimError("published runs must use the official evaluator")
    unofficial = [metric.name for metric in run.metrics if not metric.official]
    if unofficial:
        raise HeadlineClaimError(
            "published runs cannot include unofficial metrics: " + ", ".join(unofficial)
        )


def assert_not_headline_benchmark(report: RegressionReport) -> None:
    """Keep the internal lease suite from being advertised as a lab benchmark."""
    if report.is_professional_benchmark or report.headline_claim_allowed:
        raise HeadlineClaimError(
            "the internal lease regression suite is not a professional benchmark"
        )
