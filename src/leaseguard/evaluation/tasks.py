"""Turn Dataset v1 evaluation records into baseline generation tasks."""

from dataclasses import dataclass
from pathlib import Path

from leaseguard.dataset.build import assign_record_splits
from leaseguard.dataset.config import load_dataset_config
from leaseguard.dataset.examples import load_examples
from leaseguard.dataset.models import (
    DatasetConfig,
    EvidenceConversationRecord,
    RecordBase,
    RefusalRecord,
    StructuredExtractionRecord,
)
from leaseguard.evaluation.baseline_models import BaselineTaskKind, EvalSplitConfig
from leaseguard.ontology.models import DocumentAnswer, EvidenceSpan, LeaseExtraction


@dataclass(frozen=True, slots=True)
class BaselineTask:
    """One frozen evaluation example the unmodified model must answer."""

    record_id: str
    family_id: str
    subset: BaselineTaskKind
    document_id: str
    source_texts: dict[str, str]
    query: str
    perspective: str
    gold_extraction: LeaseExtraction | None
    gold_answer: DocumentAnswer | None
    gold_evidence_texts: tuple[str, ...]


def load_baseline_tasks(
    examples_root: Path,
    split: EvalSplitConfig,
    config: DatasetConfig | None = None,
) -> list[BaselineTask]:
    """Load Dataset v1 examples and keep only the frozen evaluation split."""
    dataset = config or load_dataset_config()
    records = load_examples(examples_root)
    assign_record_splits(records, dataset)
    tasks = [task for record in records if (task := task_from_record(record, split)) is not None]
    return sorted(tasks, key=lambda item: item.record_id)


def task_from_record(record: RecordBase, split: EvalSplitConfig) -> BaselineTask | None:
    """Return a generation task when the record belongs to the frozen eval split."""
    if record.split != "test":
        return None
    if record.subset not in split.scored_subsets:
        return None
    if record.subset == "structured_extraction":
        assert isinstance(record, StructuredExtractionRecord)
        evidence = _evidence_texts(
            [span for party in record.extraction.parties for span in party.evidence]
            + [span for premises in record.extraction.premises for span in premises.evidence]
            + [span for term in record.extraction.monetary_terms for span in term.evidence]
            + [span for term in record.extraction.date_terms for span in term.evidence]
            + [span for item in record.extraction.obligations for span in item.evidence]
        )
        query = "Extract parties, premises, clauses, money, dates, and obligations."
        return BaselineTask(
            record_id=record.record_id,
            family_id=record.family_id,
            subset="structured_extraction",
            document_id=record.extraction.source.document_id,
            source_texts=dict(record.source_texts),
            query=query,
            perspective="neutral",
            gold_extraction=record.extraction,
            gold_answer=None,
            gold_evidence_texts=evidence,
        )
    if record.subset in {"evidence_conversation", "refusal_uncertainty"}:
        assert isinstance(record, EvidenceConversationRecord | RefusalRecord)
        subset: BaselineTaskKind = record.subset
        evidence = _evidence_texts(record.answer.evidence)
        return BaselineTask(
            record_id=record.record_id,
            family_id=record.family_id,
            subset=subset,
            document_id=record.document_ids[0],
            source_texts=dict(record.source_texts),
            query=record.answer.question,
            perspective=record.perspective.value,
            gold_extraction=None,
            gold_answer=record.answer,
            gold_evidence_texts=evidence,
        )
    return None


def _evidence_texts(spans: list[EvidenceSpan]) -> tuple[str, ...]:
    return tuple(span.text for span in spans)
