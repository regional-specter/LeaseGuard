"""Load the pinned external benchmark registry."""

from pathlib import Path

from leaseguard.evaluation.models import BenchmarkRegistry, BenchmarkSpec


def load_benchmark_registry(path: Path) -> BenchmarkRegistry:
    """Read and validate a benchmark registry file."""
    return BenchmarkRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def get_benchmark(registry: BenchmarkRegistry, benchmark_id: str) -> BenchmarkSpec:
    """Return one approved benchmark by its stable identifier."""
    for benchmark in registry.benchmarks:
        if benchmark.benchmark_id == benchmark_id:
            return benchmark
    raise KeyError(f"Unknown benchmark_id: {benchmark_id}")
