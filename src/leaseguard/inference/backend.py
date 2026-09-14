"""Optional model backends. Transformers and CUDA stay off the local laptop."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from leaseguard.evaluation.baseline_models import (
    CandidateConfig,
    GenerationSettings,
    QuantizationName,
)
from leaseguard.inference.models import GenerationResult
from leaseguard.inference.resources import timed_call

ALLOW_MODEL_DOWNLOAD_ENV = "LEASEGUARD_ALLOW_MODEL_DOWNLOAD"


class ModelDownloadError(PermissionError):
    """Raised when a local machine tries to download publisher weights."""


class ModelBackendNotAvailableError(RuntimeError):
    """Raised when transformers or CUDA libraries are not installed."""


class GenerationBackend(Protocol):
    """Minimal text-generation surface used by the baseline runner."""

    kind: str
    resolved_revision: str | None

    def generate(self, prompt: str, settings: GenerationSettings) -> GenerationResult:
        """Return one completion for a fully rendered prompt."""

    def close(self) -> None:
        """Release any held GPU memory."""


class ScriptedBackend:
    """Deterministic local stand-in. Results cannot be published as baselines."""

    kind = "scripted"
    resolved_revision: str | None

    def __init__(self, outputs: Mapping[str, str]) -> None:
        self._outputs = dict(outputs)
        self.resolved_revision = "scripted"

    def generate(self, prompt: str, settings: GenerationSettings) -> GenerationResult:
        del settings
        text = self._lookup(prompt)
        return GenerationResult(
            text=text,
            peak_vram_mb=0.0,
            latency_seconds=0.0,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            resolved_revision=self.resolved_revision,
        )

    def close(self) -> None:
        return None

    def _lookup(self, prompt: str) -> str:
        for record_id, output in self._outputs.items():
            marker = f"Record ID: {record_id}"
            if marker in prompt:
                return output
        return "NO_JSON"


class TransformersBackend:
    """bitsandbytes-quantized Hugging Face generate() for Colab T4."""

    kind = "transformers"

    def __init__(
        self,
        candidate: CandidateConfig,
        quantization: QuantizationName,
        context_length_limit: int,
    ) -> None:
        self.candidate = candidate
        self.quantization = quantization
        self.context_length_limit = context_length_limit
        self.resolved_revision: str | None = None
        self._tokenizer: Any
        self._model: Any
        self._tokenizer, self._model = _load_transformers_model(
            candidate,
            quantization,
            context_length_limit,
        )
        revision = getattr(self._model, "config", None)
        self.resolved_revision = getattr(revision, "_name_or_path", candidate.revision)

    def generate(self, prompt: str, settings: GenerationSettings) -> GenerationResult:
        tokenizer = self._tokenizer
        model = self._model
        torch = _require_torch()
        encoded = _apply_chat_template(tokenizer, prompt)
        inputs = tokenizer(
            encoded,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_length_limit,
        )
        device = getattr(model, "device", None)
        if device is None:
            device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        def _generate() -> object:
            generate_kwargs: dict[str, object] = {
                "max_new_tokens": settings.max_new_tokens,
                "do_sample": settings.do_sample,
            }
            if settings.do_sample:
                generate_kwargs["temperature"] = settings.temperature
                generate_kwargs["top_p"] = settings.top_p
            with torch.inference_mode():
                return model.generate(**inputs, **generate_kwargs)

        output_ids, latency = timed_call(_generate)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_vram_mb = float(torch.cuda.max_memory_allocated()) / (1024 * 1024)
        else:
            peak_vram_mb = 0.0
        sequences: Any = getattr(output_ids, "sequences", output_ids)
        generated = sequences[0][prompt_tokens:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            peak_vram_mb=peak_vram_mb,
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(getattr(generated, "shape", [0])[-1]),
            resolved_revision=self.resolved_revision,
        )

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        torch = _optional_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_scripted_outputs(path: Path) -> dict[str, str]:
    """Read a record_id-to-output map used by the local scripted backend."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scripted output file must contain a JSON object")
    outputs: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("scripted outputs must map record_id strings to output strings")
        outputs[key] = value
    if not outputs:
        raise ValueError("scripted output file is empty")
    return outputs


def create_backend(
    candidate: CandidateConfig,
    quantization: QuantizationName,
    *,
    backend: str,
    allow_download: bool,
    download_env: str | None,
    scripted_outputs: Mapping[str, str] | None = None,
    context_length_limit: int | None = None,
) -> GenerationBackend:
    """Construct the requested backend without importing CUDA unless needed."""
    if backend == "scripted":
        if scripted_outputs is None:
            raise ValueError("scripted backend requires an output map")
        return ScriptedBackend(scripted_outputs)
    if backend != "transformers":
        raise ValueError(f"unknown backend: {backend}")
    _assert_model_download_allowed(allow_download, download_env)
    return TransformersBackend(
        candidate,
        quantization,
        context_length_limit or candidate.practical_context_length,
    )


def current_model_download_env() -> str | None:
    """Return the Colab gate that permits publisher-weight downloads."""
    return os.environ.get(ALLOW_MODEL_DOWNLOAD_ENV)


def _assert_model_download_allowed(allow_download: bool, download_env: str | None) -> None:
    if allow_download and download_env == "1":
        return
    raise ModelDownloadError(
        "publisher model weights can be downloaded only in Colab after setting "
        f"{ALLOW_MODEL_DOWNLOAD_ENV}=1. The local machine must not fetch checkpoints."
    )


def _require_torch() -> Any:
    torch = _optional_torch()
    if torch is None:
        raise ModelBackendNotAvailableError(
            "torch is not installed. Install CUDA libraries only on Colab."
        )
    return torch


def _optional_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return None
    return torch


def _apply_chat_template(tokenizer: Any, prompt: str) -> str:
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        rendered = apply(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = apply(messages, tokenize=False, add_generation_prompt=True)
    if isinstance(rendered, str) and rendered.strip():
        return rendered
    return prompt


def _load_transformers_model(
    candidate: CandidateConfig,
    quantization: QuantizationName,
    context_length_limit: int,
) -> tuple[Any, Any]:
    del context_length_limit
    try:
        import torch
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as error:
        raise ModelBackendNotAvailableError(
            "transformers, accelerate, and bitsandbytes must be installed only on Colab."
        ) from error

    dtype = torch.float16
    if quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        candidate.model_id,
        revision=candidate.revision,
        trust_remote_code=True,
    )
    load_kwargs: dict[str, object] = {
        "revision": candidate.revision,
        "quantization_config": quant_config,
        "device_map": "auto",
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(candidate.model_id, **load_kwargs)
    except (OSError, ValueError, TypeError):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(candidate.model_id, **load_kwargs)
    return tokenizer, model
