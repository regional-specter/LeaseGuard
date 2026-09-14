"""Load frozen Phase 6 prompt templates and render them."""

from pathlib import Path

from leaseguard.dataset.hashing import sha256_json
from leaseguard.evaluation.baseline_models import PromptCatalog, PromptMode
from leaseguard.evaluation.paths import default_prompts_path

EXTRACTION_SCHEMA = """{
  "schema_version": "1.0.0",
  "source": {"document_id": "string", "family_id": "string", "filename": "string",
    "sha256": "64-char hex",
    "document_type": "office_lease|retail_lease|amendment|exhibit|other_related",
    "source_name": "string", "rights_status": "open_license", "source_license": "string",
    "jurisdiction": "US", "language": "en"},
  "parties": [{"name": "string", "role": "landlord|tenant|guarantor|property_manager|other",
    "evidence": [{"document_id": "string", "text": "verbatim quote", "section": "string"}]}],
  "premises": [{"description": "string", "suite": "string", "address": "string",
    "evidence": [{"document_id": "string", "text": "verbatim quote", "section": "string"}]}],
  "clauses": [{"clause_id": "string", "clause_types": ["base_rent"],
    "heading": "string", "text": "string"}],
  "monetary_terms": [{"label": "string", "original_text": "string", "amount": "string",
    "currency": "USD", "frequency": "string",
    "evidence": [{"document_id": "string", "text": "verbatim quote", "section": "string"}]}],
  "date_terms": [{"label": "string", "original_text": "string", "normalized_date": "YYYY-MM-DD",
    "evidence": [{"document_id": "string", "text": "verbatim quote", "section": "string"}]}],
  "obligations": [{"obligation_id": "string", "responsible_party": "string", "action": "string",
    "timing_type": "recurring", "recurrence": "string",
    "evidence": [{"document_id": "string", "text": "verbatim quote", "section": "string"}]}],
  "amendment_effects": [{"effect_id": "string", "action": "replace", "description": "string",
    "evidence": [{"document_id": "string", "text": "verbatim quote", "section": "string"}]}],
  "warnings": ["string"]
}"""

ANSWER_SCHEMA = """{
  "schema_version": "1.0.0",
  "question": "string",
  "perspective": "landlord|tenant|neutral",
  "status": "answered|insufficient_evidence|conflicting_evidence|unreadable_source",
  "explanation": "string",
  "confidence": 0.0,
  "evidence": [{"document_id": "string", "text": "verbatim quote", "section": "string"}],
  "warnings": ["string"]
}"""

REQUIRED_TEMPLATE_NAMES = (
    "direct_extraction",
    "schema_extraction",
    "direct_answer",
    "schema_answer",
)


def load_prompt_catalog(path: Path | None = None) -> PromptCatalog:
    """Read the frozen prompt file."""
    catalog_path = path or default_prompts_path()
    catalog = PromptCatalog.model_validate_json(catalog_path.read_text(encoding="utf-8"))
    missing = [name for name in REQUIRED_TEMPLATE_NAMES if name not in catalog.templates]
    if missing:
        raise ValueError("prompt catalog missing templates: " + ", ".join(missing))
    return catalog


def template_sha256(catalog: PromptCatalog, names: tuple[str, ...]) -> str:
    """Hash the named templates so a run can pin the exact prompt text."""
    payload = {name: catalog.templates[name] for name in names}
    return sha256_json(payload)


def render_extraction_prompt(
    catalog: PromptCatalog,
    *,
    prompt_mode: PromptMode,
    record_id: str,
    document_id: str,
    family_id: str,
    context: str,
) -> str:
    """Render a structured-extraction prompt."""
    name = "schema_extraction" if prompt_mode == "schema_constrained" else "direct_extraction"
    return catalog.templates[name].format(
        schema=EXTRACTION_SCHEMA,
        record_id=record_id,
        document_id=document_id,
        family_id=family_id,
        context=context,
    )


def render_answer_prompt(
    catalog: PromptCatalog,
    *,
    prompt_mode: PromptMode,
    record_id: str,
    perspective: str,
    question: str,
    context: str,
) -> str:
    """Render an evidence-grounded answer or refusal prompt."""
    name = "schema_answer" if prompt_mode == "schema_constrained" else "direct_answer"
    return catalog.templates[name].format(
        schema=ANSWER_SCHEMA,
        record_id=record_id,
        perspective=perspective,
        question=question,
        context=context,
    )
