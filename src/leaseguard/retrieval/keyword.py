"""Exact-keyword overlap retrieval used by Phase 6 retrieval-assisted context."""

from __future__ import annotations

import re
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class RankedBlock:
    """One retrieved text block and its overlap score."""

    block_id: str
    text: str
    score: float


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def overlap_score(query: str, block: str) -> float:
    """Return token-overlap recall of the query against one block."""
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    block_tokens = set(tokenize(block))
    score = len(query_tokens & block_tokens) / len(query_tokens)
    lowered_block = block.lower()
    for phrase in _phrases(query):
        if phrase in lowered_block:
            score = min(1.0, score + 0.25)
    return min(1.0, score)


def rank_blocks(
    query: str,
    blocks: list[tuple[str, str]],
    *,
    top_k: int,
    min_score: float,
) -> list[RankedBlock]:
    """Return the highest-scoring blocks, preserving original order on ties."""
    ranked = [
        RankedBlock(block_id=block_id, text=text, score=overlap_score(query, text))
        for block_id, text in blocks
    ]
    ranked.sort(key=lambda item: (-item.score, item.block_id))
    return [item for item in ranked if item.score >= min_score][:top_k]


def _phrases(query: str) -> list[str]:
    tokens = tokenize(query)
    if len(tokens) < 2:
        return []
    return [" ".join(tokens[index : index + 2]) for index in range(len(tokens) - 1)]
