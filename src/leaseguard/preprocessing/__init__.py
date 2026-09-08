"""Document cleaning, OCR, layout, and clause segmentation."""

from leaseguard.preprocessing.duplicates import (
    exact_duplicate_map,
    jaccard_similarity,
    near_duplicate_map,
    word_shingles,
)
from leaseguard.preprocessing.models import ProcessedDocument, TextBlock
from leaseguard.preprocessing.normalize import assign_offsets, join_pages, normalize_whitespace
from leaseguard.preprocessing.segment import segment_pages

__all__ = [
    "ProcessedDocument",
    "TextBlock",
    "assign_offsets",
    "exact_duplicate_map",
    "jaccard_similarity",
    "join_pages",
    "near_duplicate_map",
    "normalize_whitespace",
    "segment_pages",
    "word_shingles",
]
