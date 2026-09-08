"""Verify that quoted evidence exists in registered source text."""

from collections.abc import Mapping, Sequence

from leaseguard.ontology.models import DocumentAnswer, EvidenceSpan, LeaseExtraction


class EvidenceAlignmentError(ValueError):
    """Raised when a quotation cannot be found in its source document."""


def evidence_failures(span: EvidenceSpan, source_texts: Mapping[str, str]) -> list[str]:
    """Return problems for one evidence span against registered source text."""
    failures: list[str] = []
    source_text = source_texts.get(span.document_id)
    if source_text is None:
        return [f"evidence document_id {span.document_id} has no registered source text"]
    if span.text not in source_text:
        failures.append(f"quoted evidence was not found in source text for {span.document_id}")
    if span.start_char is not None and span.end_char is not None:
        excerpt = source_text[span.start_char : span.end_char]
        if excerpt != span.text:
            failures.append(
                f"character offsets for {span.document_id} do not match the quoted text"
            )
    return failures


def collect_extraction_evidence(extraction: LeaseExtraction) -> list[EvidenceSpan]:
    """Return every evidence span attached to a labelled extraction."""
    spans: list[EvidenceSpan] = []
    for party in extraction.parties:
        spans.extend(party.evidence)
    for premises in extraction.premises:
        spans.extend(premises.evidence)
    for money_term in extraction.monetary_terms:
        spans.extend(money_term.evidence)
    for date_term in extraction.date_terms:
        spans.extend(date_term.evidence)
    for obligation in extraction.obligations:
        spans.extend(obligation.evidence)
    for effect in extraction.amendment_effects:
        spans.extend(effect.evidence)
    return spans


def assert_spans_in_sources(
    spans: Sequence[EvidenceSpan],
    source_texts: Mapping[str, str],
    *,
    record_id: str,
) -> None:
    """Reject a record when any quotation is missing from its source."""
    failures: list[str] = []
    for span in spans:
        failures.extend(evidence_failures(span, source_texts))
    if failures:
        raise EvidenceAlignmentError(f"{record_id}: " + "; ".join(failures))


def assert_answer_in_sources(
    answer: DocumentAnswer,
    source_texts: Mapping[str, str],
    *,
    record_id: str,
) -> None:
    """Check answer evidence the same way as extraction evidence."""
    assert_spans_in_sources(answer.evidence, source_texts, record_id=record_id)
