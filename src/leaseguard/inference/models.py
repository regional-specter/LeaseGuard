"""Shared inference request and generation result records."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Raw model text plus the runtime cost of producing it."""

    text: str
    peak_vram_mb: float
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    resolved_revision: str | None = None
