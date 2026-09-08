"""Normalize extracted text and compute character offsets."""

import hashlib
import unicodedata

from leaseguard.preprocessing.models import PageRecord, TextBlock


def normalize_whitespace(text: str) -> str:
    """Apply NFKC and collapse incidental wrapping space without deleting words."""
    normalized = unicodedata.normalize("NFKC", text)
    lines = [" ".join(line.split()) for line in normalized.splitlines()]
    return "\n".join(lines).strip()


def join_pages(pages: list[PageRecord]) -> str:
    """Join page texts in document order with a stable page separator."""
    return "\n\n".join(normalize_whitespace(page.text) for page in pages).strip()


def sha256_text(text: str) -> str:
    """Return the SHA-256 digest of normalized UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_offsets(blocks: list[TextBlock], normalized_text: str) -> list[TextBlock]:
    """Verify each block's offsets against the normalized document text."""
    assigned: list[TextBlock] = []
    for block in blocks:
        excerpt = normalized_text[block.start_char : block.end_char]
        if excerpt != block.text:
            raise ValueError(
                f"{block.block_id} offsets do not match normalized text "
                f"({block.start_char}:{block.end_char})"
            )
        assigned.append(block)
    return assigned
