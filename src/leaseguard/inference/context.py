"""Build the document context shown to an unmodified base model."""

from dataclasses import dataclass

from leaseguard.evaluation.baseline_models import ContextMode, RetrievalSettings
from leaseguard.retrieval.keyword import rank_blocks


@dataclass(frozen=True, slots=True)
class ContextBundle:
    """Text actually sent to the model plus the blocks it was built from."""

    text: str
    block_ids: tuple[str, ...]
    mode: ContextMode


def combined_source_text(source_texts: dict[str, str]) -> str:
    """Join registered source documents in stable document-id order."""
    parts = [
        f"[{document_id}]\n{source_texts[document_id].strip()}"
        for document_id in sorted(source_texts)
    ]
    return "\n\n".join(parts)


def segment_source_blocks(source_texts: dict[str, str]) -> list[tuple[str, str]]:
    """Split each source document into heading, clause, table, and exhibit blocks."""
    from leaseguard.preprocessing.models import PageRecord
    from leaseguard.preprocessing.segment import segment_pages

    blocks: list[tuple[str, str]] = []
    for document_id in sorted(source_texts):
        pages = [PageRecord(page_number=1, text=source_texts[document_id])]
        _normalized, segmented = segment_pages(pages)
        if not segmented:
            blocks.append((f"{document_id}:full", source_texts[document_id].strip()))
            continue
        for block in segmented:
            blocks.append((f"{document_id}:{block.block_id}", f"[{document_id}] {block.text}"))
    return blocks


def build_context(
    source_texts: dict[str, str],
    *,
    mode: ContextMode,
    query: str,
    retrieval: RetrievalSettings,
) -> ContextBundle:
    """Select full text, clause blocks, or retrieved blocks for one prompt."""
    if mode == "full_document":
        text = combined_source_text(source_texts)
        return ContextBundle(text=text, block_ids=("full_document",), mode=mode)

    blocks = segment_source_blocks(source_texts)
    if mode == "clause_level":
        text = "\n".join(block_text for _block_id, block_text in blocks)
        return ContextBundle(
            text=text,
            block_ids=tuple(block_id for block_id, _text in blocks),
            mode=mode,
        )

    ranked = rank_blocks(query, blocks, top_k=retrieval.top_k, min_score=retrieval.min_score)
    if not ranked:
        fallback = combined_source_text(source_texts)
        return ContextBundle(text=fallback, block_ids=("retrieval_fallback",), mode=mode)
    text = "\n".join(item.text for item in ranked)
    return ContextBundle(
        text=text,
        block_ids=tuple(item.block_id for item in ranked),
        mode=mode,
    )


def gold_evidence_in_context(evidence_texts: list[str], context: str) -> bool:
    """True when every non-empty gold quote occurs in the supplied context."""
    if not evidence_texts:
        return True
    return all(quote in context for quote in evidence_texts if quote)
