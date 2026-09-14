"""Score unmodified base-model predictions against Dataset v1 gold labels."""

from leaseguard.evaluation.metrics import precision_recall_f1
from leaseguard.evaluation.models import MetricResult
from leaseguard.ontology.models import DocumentAnswer, LeaseExtraction


def extraction_label_f1(predicted: LeaseExtraction, gold: LeaseExtraction) -> float:
    """Compare parties, clause types, money, dates, and obligations as label sets."""
    return precision_recall_f1(_extraction_labels(predicted), _extraction_labels(gold)).f1


def answer_status_correct(predicted: DocumentAnswer, gold: DocumentAnswer) -> bool:
    """True when the model chose the same answer status as the gold record."""
    return predicted.status == gold.status


def evidence_recall(predicted_quotes: list[str], gold_quotes: list[str]) -> float:
    """Measure how many gold quotes appear verbatim in the model output quotes."""
    return precision_recall_f1(predicted_quotes, gold_quotes).recall


def content_is_correct(
    *,
    predicted_extraction: LeaseExtraction | None,
    gold_extraction: LeaseExtraction | None,
    predicted_answer: DocumentAnswer | None,
    gold_answer: DocumentAnswer | None,
) -> bool:
    """True when scored fields match well enough to count as a correct example."""
    if gold_extraction is not None:
        if predicted_extraction is None:
            return False
        return extraction_label_f1(predicted_extraction, gold_extraction) == 1.0
    if gold_answer is not None:
        if predicted_answer is None:
            return False
        return answer_status_correct(predicted_answer, gold_answer)
    return False


def summarize_metrics(
    *,
    schema_valid_count: int,
    extraction_scores: list[float],
    answer_correct_count: int,
    answer_count: int,
    evidence_scores: list[float],
    total: int,
) -> list[MetricResult]:
    """Build unofficial product-task metrics for one baseline run."""
    return [
        MetricResult(
            name="schema_validity",
            value=(schema_valid_count / total) if total else 0.0,
            official=False,
        ),
        MetricResult(
            name="extraction_f1",
            value=(sum(extraction_scores) / len(extraction_scores) if extraction_scores else 0.0),
            official=False,
        ),
        MetricResult(
            name="answer_status_accuracy",
            value=(answer_correct_count / answer_count) if answer_count else 0.0,
            official=False,
        ),
        MetricResult(
            name="evidence_recall",
            value=(sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.0),
            official=False,
        ),
    ]


def _extraction_labels(extraction: LeaseExtraction) -> list[str]:
    labels = [f"party:{party.role.value}:{party.name}" for party in extraction.parties]
    labels.extend(f"premises:{item.description}" for item in extraction.premises)
    labels.extend(
        f"clause:{clause_type.value}"
        for clause in extraction.clauses
        for clause_type in clause.clause_types
    )
    labels.extend(
        f"money:{term.amount}:{term.currency}"
        for term in extraction.monetary_terms
        if term.amount is not None
    )
    labels.extend(
        f"date:{term.normalized_date.isoformat()}"
        for term in extraction.date_terms
        if term.normalized_date is not None
    )
    labels.extend(
        f"obligation:{item.responsible_party}:{item.action}" for item in extraction.obligations
    )
    return labels


def predicted_quotes(
    extraction: LeaseExtraction | None,
    answer: DocumentAnswer | None,
) -> list[str]:
    """Collect evidence quotes from a validated prediction."""
    quotes: list[str] = []
    if extraction is not None:
        quotes.extend(span.text for party in extraction.parties for span in party.evidence)
        quotes.extend(span.text for item in extraction.premises for span in item.evidence)
        quotes.extend(span.text for term in extraction.monetary_terms for span in term.evidence)
        quotes.extend(span.text for term in extraction.date_terms for span in term.evidence)
        quotes.extend(span.text for item in extraction.obligations for span in item.evidence)
    if answer is not None:
        quotes.extend(span.text for span in answer.evidence)
    return quotes
