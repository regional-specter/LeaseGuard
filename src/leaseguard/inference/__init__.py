"""Local model inference and output validation."""

from leaseguard.inference.backend import (
    ALLOW_MODEL_DOWNLOAD_ENV,
    ModelBackendNotAvailableError,
    ModelDownloadError,
    ScriptedBackend,
    create_backend,
    current_model_download_env,
    load_scripted_outputs,
)
from leaseguard.inference.context import ContextBundle, build_context, gold_evidence_in_context
from leaseguard.inference.parse import extract_json_object
from leaseguard.inference.prompts import (
    load_prompt_catalog,
    render_answer_prompt,
    render_extraction_prompt,
)

__all__ = [
    "ALLOW_MODEL_DOWNLOAD_ENV",
    "ContextBundle",
    "ModelBackendNotAvailableError",
    "ModelDownloadError",
    "ScriptedBackend",
    "build_context",
    "create_backend",
    "current_model_download_env",
    "extract_json_object",
    "gold_evidence_in_context",
    "load_prompt_catalog",
    "load_scripted_outputs",
    "render_answer_prompt",
    "render_extraction_prompt",
]
