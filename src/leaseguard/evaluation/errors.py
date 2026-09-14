"""Attribute product-task failures to model, prompt, parse, retrieval, or validation."""

from typing import Any

from pydantic import ValidationError

from leaseguard.evaluation.baseline_models import ErrorClass, PromptMode
from leaseguard.ontology.models import DocumentAnswer, LeaseExtraction


def classify_error(
    *,
    prompt_mode: PromptMode,
    parsed: dict[str, Any] | None,
    schema_valid: bool,
    gold_evidence_in_context: bool,
    content_correct: bool,
) -> ErrorClass:
    """Assign one error class so fine-tuning hypotheses stay specific."""
    if parsed is None:
        if prompt_mode == "direct":
            return "prompt_error"
        return "parse_error"
    if not schema_valid:
        return "validation_error"
    if content_correct:
        return "correct"
    if not gold_evidence_in_context:
        return "retrieval_error"
    return "model_error"


def validate_extraction_payload(
    payload: dict[str, Any],
    *,
    document_id: str,
    family_id: str,
) -> tuple[LeaseExtraction | None, bool, dict[str, Any]]:
    """Validate extraction JSON, injecting known source identity when omitted."""
    working = dict(payload)
    injected = False
    source = working.get("source")
    if not isinstance(source, dict) or "document_id" not in source or "family_id" not in source:
        injected = True
        working["source"] = {
            **(source if isinstance(source, dict) else {}),
            "document_id": document_id,
            "family_id": family_id,
        }
    try:
        extraction = LeaseExtraction.model_validate(working)
    except (ValidationError, ValueError):
        return None, injected, working
    return extraction, injected, working


def validate_answer_payload(
    payload: dict[str, Any],
    *,
    question: str,
    perspective: str,
) -> tuple[DocumentAnswer | None, dict[str, Any]]:
    """Validate an answer JSON object, filling known question identity when omitted."""
    working = dict(payload)
    if "question" not in working:
        working["question"] = question
    if "perspective" not in working:
        working["perspective"] = perspective
    try:
        answer = DocumentAnswer.model_validate(working)
    except (ValidationError, ValueError):
        return None, working
    return answer, working
