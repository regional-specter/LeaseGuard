"""Exact and near-duplicate detection for processed documents."""

from collections.abc import Sequence

from leaseguard.preprocessing.models import ProcessedDocument


def exact_duplicate_map(documents: Sequence[ProcessedDocument]) -> dict[str, str]:
    """Map a later document_id to the first document with the same raw checksum."""
    seen: dict[str, str] = {}
    duplicates: dict[str, str] = {}
    for document in documents:
        original = seen.setdefault(document.source.sha256, document.source.document_id)
        if original != document.source.document_id:
            duplicates[document.source.document_id] = original
    return duplicates


def word_shingles(text: str, size: int) -> set[str]:
    """Return overlapping word n-grams used for near-duplicate comparison."""
    words = [token.lower() for token in text.split() if token]
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + size]) for index in range(len(words) - size + 1)}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Return set overlap in the unit interval."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union)


def near_duplicate_map(
    documents: Sequence[ProcessedDocument],
    *,
    threshold: float,
    shingle_size: int,
) -> dict[str, list[str]]:
    """Map each document to earlier near-duplicates at or above the threshold."""
    shingles = [word_shingles(document.normalized_text, shingle_size) for document in documents]
    matches: dict[str, list[str]] = {document.source.document_id: [] for document in documents}
    for index, document in enumerate(documents):
        for earlier_index in range(index):
            score = jaccard_similarity(shingles[index], shingles[earlier_index])
            if score >= threshold:
                matches[document.source.document_id].append(
                    documents[earlier_index].source.document_id
                )
    return matches
