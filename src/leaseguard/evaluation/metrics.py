"""Small deterministic metrics for local regression checks.

Published benchmark results must use each benchmark's official evaluator.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Precision, recall, and F1 with their underlying counts."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, order=True, slots=True)
class CharacterSpan:
    """Zero-based, end-exclusive character range."""

    start: int
    end: int

    def __post_init__(self) -> None:
        """Reject negative, empty, and reversed spans."""
        if self.start < 0:
            raise ValueError("span start cannot be negative")
        if self.end <= self.start:
            raise ValueError("span end must be greater than span start")

    @property
    def length(self) -> int:
        """Return the number of covered characters."""
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Character-level retrieval precision and recall."""

    predicted_characters: int
    reference_characters: int
    overlapping_characters: int
    precision: float
    recall: float


def precision_recall_f1(
    predicted_labels: Iterable[str],
    reference_labels: Iterable[str],
) -> ClassificationMetrics:
    """Score one set of labels against its reference set."""
    predicted = set(predicted_labels)
    reference = set(reference_labels)
    true_positives = len(predicted & reference)
    false_positives = len(predicted - reference)
    false_negatives = len(reference - predicted)

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = (
        true_positives / precision_denominator if precision_denominator else float(not reference)
    )
    recall = true_positives / recall_denominator if recall_denominator else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return ClassificationMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def merge_character_spans(spans: Iterable[CharacterSpan]) -> tuple[CharacterSpan, ...]:
    """Merge overlapping or touching spans so characters are counted once."""
    ordered = sorted(spans)
    if not ordered:
        return ()

    merged = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start <= previous.end:
            merged[-1] = CharacterSpan(previous.start, max(previous.end, current.end))
        else:
            merged.append(current)
    return tuple(merged)


def character_retrieval_metrics(
    predicted_spans: Sequence[CharacterSpan],
    reference_spans: Sequence[CharacterSpan],
) -> RetrievalMetrics:
    """Measure exact character coverage without double-counting overlaps."""
    predicted = merge_character_spans(predicted_spans)
    reference = merge_character_spans(reference_spans)
    predicted_characters = sum(span.length for span in predicted)
    reference_characters = sum(span.length for span in reference)
    overlapping_characters = _overlap_length(predicted, reference)

    precision = (
        overlapping_characters / predicted_characters
        if predicted_characters
        else float(not reference)
    )
    recall = overlapping_characters / reference_characters if reference_characters else 1.0
    return RetrievalMetrics(
        predicted_characters=predicted_characters,
        reference_characters=reference_characters,
        overlapping_characters=overlapping_characters,
        precision=precision,
        recall=recall,
    )


def ranked_precision_recall(relevances: Sequence[bool]) -> tuple[list[float], list[float]]:
    """Build a precision-recall curve from predictions ordered by decreasing score.

    This helper is for local protocol tests. Published CUAD numbers must come from
    the official `evaluate.py` in the pinned Atticus repository.
    """
    total_relevant = sum(relevances)
    true_positives = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for index, is_relevant in enumerate(relevances, start=1):
        if is_relevant:
            true_positives += 1
        precision = true_positives / index
        recall = true_positives / total_relevant if total_relevant else 1.0
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls


def precision_at_recall(
    precisions: Sequence[float],
    recalls: Sequence[float],
    target_recall: float,
) -> float:
    """Return the highest precision among points that reach the target recall."""
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be greater than 0 and at most 1")
    if len(precisions) != len(recalls):
        raise ValueError("precisions and recalls must have the same length")
    eligible = [
        precision
        for precision, recall in zip(precisions, recalls, strict=True)
        if recall >= target_recall
    ]
    return max(eligible) if eligible else 0.0


def area_under_pr_curve(precisions: Sequence[float], recalls: Sequence[float]) -> float:
    """Trapezoidal area under a precision-recall curve, clipped to [0, 1]."""
    if len(precisions) != len(recalls):
        raise ValueError("precisions and recalls must have the same length")
    if not precisions:
        return 0.0

    ordered = sorted(zip(recalls, precisions, strict=True))
    area = 0.0
    for index in range(1, len(ordered)):
        previous_recall, previous_precision = ordered[index - 1]
        current_recall, current_precision = ordered[index]
        area += (current_recall - previous_recall) * (previous_precision + current_precision) / 2
    return max(0.0, min(1.0, area))


def _overlap_length(
    left_spans: Sequence[CharacterSpan],
    right_spans: Sequence[CharacterSpan],
) -> int:
    """Return the intersection length of two sorted, disjoint span lists."""
    left_index = 0
    right_index = 0
    overlap = 0
    while left_index < len(left_spans) and right_index < len(right_spans):
        left = left_spans[left_index]
        right = right_spans[right_index]
        overlap += max(0, min(left.end, right.end) - max(left.start, right.start))
        if left.end <= right.end:
            left_index += 1
        else:
            right_index += 1
    return overlap
