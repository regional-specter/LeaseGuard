"""Measure generation cost without requiring CUDA in local tests."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from leaseguard.evaluation.baseline_models import ResourceMetrics
from leaseguard.inference.models import GenerationResult

T = TypeVar("T")


def empty_resources(hardware: str, context_length_limit: int) -> ResourceMetrics:
    """Return a zeroed resource record for a failed or skipped generation."""
    return ResourceMetrics(
        hardware=hardware,
        peak_vram_mb=0.0,
        latency_seconds=0.0,
        prompt_tokens=0,
        completion_tokens=0,
        tokens_per_second=0.0,
        context_length_limit=context_length_limit,
    )


def resources_from_generation(
    result: GenerationResult,
    *,
    hardware: str,
    context_length_limit: int,
) -> ResourceMetrics:
    """Convert one generation result into a stored resource record."""
    tokens_per_second = 0.0
    if result.latency_seconds > 0 and result.completion_tokens:
        tokens_per_second = result.completion_tokens / result.latency_seconds
    return ResourceMetrics(
        hardware=hardware,
        peak_vram_mb=result.peak_vram_mb,
        latency_seconds=result.latency_seconds,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        tokens_per_second=tokens_per_second,
        context_length_limit=context_length_limit,
    )


def aggregate_resources(
    rows: list[ResourceMetrics],
    *,
    hardware: str,
    context_length_limit: int,
) -> ResourceMetrics:
    """Sum latency and tokens, and keep the peak VRAM across predictions."""
    if not rows:
        return empty_resources(hardware, context_length_limit)
    latency = sum(row.latency_seconds for row in rows)
    completion_tokens = sum(row.completion_tokens for row in rows)
    tokens_per_second = completion_tokens / latency if latency > 0 and completion_tokens else 0.0
    return ResourceMetrics(
        hardware=hardware,
        peak_vram_mb=max(row.peak_vram_mb for row in rows),
        latency_seconds=latency,
        prompt_tokens=sum(row.prompt_tokens for row in rows),
        completion_tokens=completion_tokens,
        tokens_per_second=tokens_per_second,
        context_length_limit=context_length_limit,
    )


def timed_call(func: Callable[[], T]) -> tuple[T, float]:
    """Run a callable and return its wall-clock duration in seconds."""
    started = time.perf_counter()
    value = func()
    return value, time.perf_counter() - started
