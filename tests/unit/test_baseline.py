"""Tests for frozen eval splits, baseline configs, and Colab-ready runners."""

from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from leaseguard.dataset.paths import default_seed_examples_path
from leaseguard.evaluation.baseline import (
    BaselineConfigError,
    append_prediction_checkpoint,
    get_candidate,
    get_comparison,
    load_baseline_registry,
    load_prediction_checkpoint,
    run_baseline,
    run_id_for,
)
from leaseguard.evaluation.baseline_models import (
    BaselinePrediction,
    BaselineRegistry,
    BaselineRun,
    ResourceMetrics,
)
from leaseguard.evaluation.cli import main
from leaseguard.evaluation.compare import (
    compare_baseline_runs,
    dominant_error_class,
    fine_tuning_hypothesis,
)
from leaseguard.evaluation.errors import (
    classify_error,
    validate_answer_payload,
    validate_extraction_payload,
)
from leaseguard.evaluation.eval_split import (
    EvalSplitError,
    assert_eval_split_matches_dataset,
    assert_eval_split_not_used_for_development,
    load_eval_split,
)
from leaseguard.evaluation.models import MetricResult
from leaseguard.evaluation.protocol import HoldoutIntegrityError
from leaseguard.evaluation.scoring import content_is_correct, summarize_metrics
from leaseguard.evaluation.tasks import load_baseline_tasks
from leaseguard.inference.backend import (
    ModelDownloadError,
    ScriptedBackend,
    TransformersBackend,
    create_backend,
    load_scripted_outputs,
)
from leaseguard.inference.context import build_context, gold_evidence_in_context
from leaseguard.inference.models import GenerationResult
from leaseguard.inference.parse import extract_json_object
from leaseguard.inference.prompts import (
    load_prompt_catalog,
    render_answer_prompt,
    render_extraction_prompt,
)
from leaseguard.inference.resources import aggregate_resources, empty_resources, timed_call
from leaseguard.retrieval.keyword import overlap_score, rank_blocks, tokenize


def gold_scripted_outputs() -> dict[str, str]:
    """Replay Dataset v1 gold JSON. Used only for local pipeline tests."""
    split = load_eval_split()
    outputs: dict[str, str] = {}
    for task in load_baseline_tasks(default_seed_examples_path(), split):
        if task.gold_extraction is not None:
            outputs[task.record_id] = task.gold_extraction.model_dump_json()
        elif task.gold_answer is not None:
            outputs[task.record_id] = task.gold_answer.model_dump_json()
    return outputs


def _resources(**changes: Any) -> ResourceMetrics:
    values: dict[str, Any] = {
        "hardware": "Google Colab T4",
        "peak_vram_mb": 8000.0,
        "latency_seconds": 10.0,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "tokens_per_second": 5.0,
        "context_length_limit": 4096,
    }
    values.update(changes)
    return ResourceMetrics(**values)


def _prediction(record_id: str, **changes: Any) -> BaselinePrediction:
    values: dict[str, Any] = {
        "record_id": record_id,
        "family_id": "ds1-office-family",
        "subset": "structured_extraction",
        "prompt_mode": "schema_constrained",
        "context_mode": "full_document",
        "quantization": "4bit",
        "error_class": "correct",
        "schema_valid": True,
        "content_score": 1.0,
        "gold_evidence_in_context": True,
        "raw_output": "{}",
        "detail": "ok",
        "resources": _resources(),
    }
    values.update(changes)
    return BaselinePrediction(**values)


def _run(candidate_id: str, **changes: Any) -> BaselineRun:
    started = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    values: dict[str, Any] = {
        "run_id": f"{candidate_id}-schema-full-4bit",
        "candidate_id": candidate_id,
        "comparison_id": "schema-full-4bit",
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": "main",
        "dataset_version": "1.0.0",
        "eval_split_version": "1.0.0",
        "prompt_version": "1.0.0",
        "prompt_template_sha256": "a" * 64,
        "backend_kind": "transformers",
        "hardware": "Google Colab T4",
        "quantization": "4bit",
        "prompt_mode": "schema_constrained",
        "context_mode": "full_document",
        "random_seed": 3407,
        "started_at": started,
        "completed_at": started + timedelta(minutes=5),
        "contamination_notes": "unknown pretraining overlap",
        "metrics": [
            MetricResult(name="schema_validity", value=0.5, official=False),
            MetricResult(name="extraction_f1", value=0.4, official=False),
            MetricResult(name="answer_status_accuracy", value=0.5, official=False),
            MetricResult(name="evidence_recall", value=1.0, official=False),
        ],
        "error_counts": {"correct": 1, "model_error": 1},
        "resources": _resources(),
        "predictions": [
            _prediction(f"{candidate_id}-one"),
            _prediction(f"{candidate_id}-two", error_class="model_error", content_score=0.0),
        ],
    }
    values.update(changes)
    return BaselineRun(**values)


def test_eval_split_matches_dataset_v1_family_hash() -> None:
    """The frozen seed family assignments must match Dataset v1 hashing."""
    split = load_eval_split()
    assert_eval_split_matches_dataset(split)
    assert split.headline_claim_allowed is False
    families = {item.family_id: item.split for item in split.seed_families}
    assert families["ds1-office-family"] == "test"
    assert families["ds1-retail-family"] == "train"
    tasks = load_baseline_tasks(default_seed_examples_path(), split)
    assert {task.record_id for task in tasks} == set(split.seed_eval_record_ids)


def test_eval_split_rejects_development_use_and_salt_drift() -> None:
    """The holdout cannot be used for prompt search and must keep the Dataset v1 salt."""
    with pytest.raises(HoldoutIntegrityError, match="prompt-selection"):
        assert_eval_split_not_used_for_development("prompt-selection")
    split = load_eval_split()
    payload = split.model_dump(mode="json")
    payload["salt"] = "changed-salt"
    drifted = type(split).model_validate(payload)
    with pytest.raises(EvalSplitError, match="salt"):
        assert_eval_split_matches_dataset(drifted)


def test_baseline_registry_pins_three_t4_candidates() -> None:
    """Every candidate has one frozen config and the required 4-bit comparison."""
    registry = load_baseline_registry()
    assert {item.candidate_id for item in registry.candidates} == {
        "qwen35-4b",
        "qwen35-9b",
        "gemma3-12b",
    }
    required = get_comparison(registry, registry.required_comparison_id)
    assert required.prompt_mode == "schema_constrained"
    assert required.quantization == "4bit"
    assert get_candidate(registry, "gemma3-12b").gated is True
    with pytest.raises(BaselineConfigError, match="unknown candidate"):
        get_candidate(registry, "missing")
    with pytest.raises(BaselineConfigError, match="unknown comparison"):
        get_comparison(registry, "missing")


def test_registry_rejects_unknown_comparison_candidate() -> None:
    """A comparison cannot name a candidate that is not in the registry."""
    registry = load_baseline_registry()
    payload = registry.model_dump(mode="json")
    payload["comparisons"][0]["candidate_ids"].append("unknown-model")
    with pytest.raises(ValidationError, match="unknown comparison candidate"):
        BaselineRegistry.model_validate(payload)


def test_json_parser_reads_fences_and_rejects_non_objects() -> None:
    """Model output may wrap JSON in markdown fences."""
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('prefix {"b": 2} suffix') == {"b": 2}
    assert extract_json_object("") is None
    assert extract_json_object("[1, 2]") is None
    assert extract_json_object("```json\n[1]\n```") is None


def test_error_classes_separate_prompt_parse_retrieval_and_model() -> None:
    """Fine-tuning hypotheses depend on a specific failure class."""
    assert (
        classify_error(
            prompt_mode="direct",
            parsed=None,
            schema_valid=False,
            gold_evidence_in_context=True,
            content_correct=False,
        )
        == "prompt_error"
    )
    assert (
        classify_error(
            prompt_mode="schema_constrained",
            parsed=None,
            schema_valid=False,
            gold_evidence_in_context=True,
            content_correct=False,
        )
        == "parse_error"
    )
    assert (
        classify_error(
            prompt_mode="schema_constrained",
            parsed={"x": 1},
            schema_valid=False,
            gold_evidence_in_context=True,
            content_correct=False,
        )
        == "validation_error"
    )
    assert (
        classify_error(
            prompt_mode="schema_constrained",
            parsed={"x": 1},
            schema_valid=True,
            gold_evidence_in_context=False,
            content_correct=False,
        )
        == "retrieval_error"
    )
    assert (
        classify_error(
            prompt_mode="schema_constrained",
            parsed={"x": 1},
            schema_valid=True,
            gold_evidence_in_context=True,
            content_correct=True,
        )
        == "correct"
    )


def test_keyword_retrieval_and_context_modes() -> None:
    """Clause and retrieval context are built without downloading models."""
    texts = {
        "doc-a": "Section 1. Base Rent. Tenant shall pay USD 5000 monthly.",
        "doc-b": "Exhibit A. Parking is not included.",
    }
    retrieval = load_baseline_registry().retrieval
    full = build_context(texts, mode="full_document", query="base rent", retrieval=retrieval)
    clause = build_context(texts, mode="clause_level", query="base rent", retrieval=retrieval)
    retrieved = build_context(
        texts, mode="retrieval_assisted", query="base rent USD", retrieval=retrieval
    )
    assert "[doc-a]" in full.text
    assert clause.block_ids
    assert retrieved.block_ids
    assert gold_evidence_in_context(["USD 5000"], full.text) is True
    assert gold_evidence_in_context(["missing quote"], full.text) is False
    ranked = rank_blocks(
        "parking", [("a", "Base rent"), ("b", "Parking is not included")], top_k=1, min_score=0.0
    )
    assert ranked[0].block_id == "b"
    assert tokenize("USD 5,000") == ["usd", "5", "000"]
    assert overlap_score("", "text") == 0.0


def test_scripted_baseline_run_scores_seed_holdout(tmp_path: Path) -> None:
    """The local scripted backend exercises the full product-task loop."""
    backend = ScriptedBackend(gold_scripted_outputs())
    run = run_baseline(
        candidate_id="qwen35-4b",
        comparison_id="schema-full-4bit",
        backend=backend,
        output_dir=tmp_path,
        checkpoint_dir=tmp_path / "ckpt",
        hardware="cpu-test",
    )
    assert run.backend_kind == "scripted"
    assert run.headline_claim_allowed is False
    assert {item.record_id for item in run.predictions} == set(
        load_eval_split().seed_eval_record_ids
    )
    assert all(item.error_class == "correct" for item in run.predictions)
    saved = tmp_path / f"{run.run_id}.json"
    assert saved.is_file()
    resumed = load_prediction_checkpoint(tmp_path / "ckpt" / f"{run.run_id}.predictions.jsonl")
    assert len(resumed) == len(run.predictions)


def test_checkpoint_resume_skips_completed_records(tmp_path: Path) -> None:
    """A restarted Colab session must not redo finished examples."""
    run_id = run_id_for("qwen35-4b", "direct-full-4bit")
    path = tmp_path / f"{run_id}.predictions.jsonl"
    first = _prediction("ds1-extract-amendment", raw_output="checkpoint")
    append_prediction_checkpoint(path, first)
    backend = ScriptedBackend(gold_scripted_outputs())
    run = run_baseline(
        candidate_id="qwen35-4b",
        comparison_id="direct-full-4bit",
        backend=backend,
        checkpoint_dir=tmp_path,
        hardware="cpu-test",
        limit=1,
    )
    assert run.predictions[0].raw_output == "checkpoint"


def test_direct_prompting_without_json_is_prompt_error() -> None:
    """Direct prompting that returns prose is not scored as a model label error."""
    backend = ScriptedBackend({"ds1-extract-office": "The rent is five thousand dollars."})
    run = run_baseline(
        candidate_id="qwen35-4b",
        comparison_id="direct-full-4bit",
        backend=backend,
        hardware="cpu-test",
        limit=1,
    )
    assert run.predictions[0].error_class == "prompt_error"


def test_schema_constrained_invalid_json_is_parse_error() -> None:
    """Schema prompting that still returns prose is a parse failure."""
    backend = ScriptedBackend({"ds1-extract-office": "NO_JSON"})
    run = run_baseline(
        candidate_id="qwen35-4b",
        comparison_id="schema-full-4bit",
        backend=backend,
        hardware="cpu-test",
        limit=1,
    )
    assert run.predictions[0].error_class == "parse_error"


def test_create_backend_blocks_local_weight_downloads() -> None:
    """Publisher checkpoints stay off the laptop unless Colab sets the gate."""
    candidate = get_candidate(load_baseline_registry(), "qwen35-4b")
    with pytest.raises(ModelDownloadError, match="LEASEGUARD_ALLOW_MODEL_DOWNLOAD"):
        create_backend(
            candidate,
            "4bit",
            backend="transformers",
            allow_download=False,
            download_env=None,
        )
    with pytest.raises(ValueError, match="scripted backend requires"):
        create_backend(
            candidate,
            "4bit",
            backend="scripted",
            allow_download=False,
            download_env=None,
        )
    with pytest.raises(ValueError, match="unknown backend"):
        create_backend(
            candidate,
            "4bit",
            backend="vllm",
            allow_download=True,
            download_env="1",
        )


def test_scripted_output_file_validation(tmp_path: Path) -> None:
    """The local backend map must be a non-empty string-to-string object."""
    path = tmp_path / "script.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_scripted_outputs(path)
    path.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="record_id strings"):
        load_scripted_outputs(path)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_scripted_outputs(path)


def test_compare_selects_strongest_practical_candidate() -> None:
    """Selection uses lexicographic quality, then speed, then lower VRAM."""
    registry = load_baseline_registry()
    weak = _run(
        "qwen35-4b",
        metrics=[
            MetricResult(name="schema_validity", value=0.2, official=False),
            MetricResult(name="extraction_f1", value=0.2, official=False),
            MetricResult(name="answer_status_accuracy", value=0.2, official=False),
            MetricResult(name="evidence_recall", value=0.2, official=False),
        ],
    )
    strong = _run(
        "qwen35-9b",
        model_id="Qwen/Qwen3.5-9B",
        metrics=[
            MetricResult(name="schema_validity", value=0.9, official=False),
            MetricResult(name="extraction_f1", value=0.8, official=False),
            MetricResult(name="answer_status_accuracy", value=0.7, official=False),
            MetricResult(name="evidence_recall", value=1.0, official=False),
        ],
        resources=_resources(peak_vram_mb=9000.0, tokens_per_second=8.0),
    )
    oom = _run(
        "gemma3-12b",
        model_id="google/gemma-3-12b-it",
        oom=True,
        resources=_resources(peak_vram_mb=16001.0),
        error_counts={"model_error": 2},
    )
    report = compare_baseline_runs([weak, strong, oom], registry)
    assert report.incomplete is False
    assert report.selected_candidate_id == "qwen35-9b"
    assert report.headline_claim_allowed is False
    assert "lexicographic" in report.notes


def test_compare_marks_incomplete_without_all_candidates() -> None:
    """A winner is not chosen until every candidate has the required run."""
    registry = load_baseline_registry()
    report = compare_baseline_runs([_run("qwen35-4b")], registry)
    assert report.incomplete is True
    assert report.selected_candidate_id is None
    assert "qwen35-9b" in report.notes


def test_scripted_runs_are_not_practical() -> None:
    """Local dry-runs cannot select a base model."""
    registry = load_baseline_registry()
    scripted = _run("qwen35-4b", backend_kind="scripted")
    assert dominant_error_class(scripted) == "model_error"
    report = compare_baseline_runs(
        [
            scripted,
            _run("qwen35-9b", backend_kind="scripted"),
            _run("gemma3-12b", backend_kind="scripted"),
        ],
        registry,
    )
    assert report.selected_candidate_id is None
    assert (
        "practical" in report.fine_tuning_hypothesis.lower()
        or "T4" in report.fine_tuning_hypothesis
    )


def test_fine_tuning_hypothesis_maps_error_classes() -> None:
    """Each dominant failure class produces a specific Phase 7 hypothesis."""
    ranking = compare_baseline_runs(
        [_run("qwen35-4b"), _run("qwen35-9b"), _run("gemma3-12b")],
        load_baseline_registry(),
    ).rankings
    ranking[0].dominant_error_class = "parse_error"
    text = fine_tuning_hypothesis(ranking)
    assert "JSON" in text


def test_cli_validate_and_scripted_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI commands stay available without CUDA libraries."""
    assert main(["list-candidates"]) == 0
    assert main(["validate-baselines"]) == 0
    assert main(["show-eval-split"]) == 0
    script = tmp_path / "script.json"
    script.write_text(json.dumps(gold_scripted_outputs()), encoding="utf-8")
    status = main(
        [
            "baseline",
            "--candidate",
            "qwen35-4b",
            "--comparison",
            "schema-clause-4bit",
            "--backend",
            "scripted",
            "--script",
            str(script),
            "--output-dir",
            str(tmp_path / "out"),
            "--hardware",
            "cpu-test",
            "--limit",
            "1",
        ]
    )
    assert status == 0
    captured = capsys.readouterr()
    assert "scripted backend results cannot be published" in captured.out
    compare_status = main(
        [
            "compare-baselines",
            str(tmp_path / "out" / "qwen35-4b-schema-clause-4bit.json"),
            "--output",
            str(tmp_path / "compare.json"),
        ]
    )
    assert compare_status == 0


def test_cli_rejects_local_model_download(capsys: pytest.CaptureFixture[str]) -> None:
    """The transformers backend stays gated on the local machine."""
    status = main(
        [
            "baseline",
            "--candidate",
            "qwen35-4b",
            "--comparison",
            "schema-full-4bit",
            "--backend",
            "transformers",
        ]
    )
    assert status == 2
    assert "LEASEGUARD_ALLOW_MODEL_DOWNLOAD" in capsys.readouterr().err


def test_prompts_render_record_identity() -> None:
    """Every prompt includes the record ID used by the scripted backend."""
    catalog = load_prompt_catalog()
    extraction = render_extraction_prompt(
        catalog,
        prompt_mode="schema_constrained",
        record_id="ds1-extract-office",
        document_id="ds1-office-lease",
        family_id="ds1-office-family",
        context="lease text",
    )
    answer = render_answer_prompt(
        catalog,
        prompt_mode="direct",
        record_id="ds1-qa-office-rent",
        perspective="neutral",
        question="When is rent due?",
        context="lease text",
    )
    assert "Record ID: ds1-extract-office" in extraction
    assert "JSON:" in extraction
    assert "Record ID: ds1-qa-office-rent" in answer


def test_source_identity_injection_and_answer_validation() -> None:
    """Known document identity may be filled in; invented facts may not."""
    extraction, injected, _working = validate_extraction_payload(
        {"schema_version": "1.0.0"},
        document_id="doc-1",
        family_id="fam-1",
    )
    assert extraction is None
    assert injected is True
    answer, working = validate_answer_payload(
        {
            "schema_version": "1.0.0",
            "status": "insufficient_evidence",
            "explanation": "The lease does not say.",
            "confidence": 0.1,
        },
        question="What statute applies?",
        perspective="neutral",
    )
    assert answer is not None
    assert working["question"] == "What statute applies?"
    assert content_is_correct(
        predicted_extraction=None,
        gold_extraction=None,
        predicted_answer=answer,
        gold_answer=answer,
    )


def test_resource_helpers_and_timed_call() -> None:
    """VRAM and latency aggregations stay defined with empty runs."""
    empty = empty_resources("cpu", 1024)
    assert empty.tokens_per_second == 0.0
    combined = aggregate_resources(
        [
            _resources(latency_seconds=2, completion_tokens=10),
            _resources(latency_seconds=2, completion_tokens=10),
        ],
        hardware="cpu",
        context_length_limit=1024,
    )
    assert combined.completion_tokens == 20
    assert combined.tokens_per_second == pytest.approx(5.0)
    value, elapsed = timed_call(lambda: 7)
    assert value == 7
    assert elapsed >= 0
    assert (
        summarize_metrics(
            schema_valid_count=0,
            extraction_scores=[],
            answer_correct_count=0,
            answer_count=0,
            evidence_scores=[],
            total=0,
        )[0].value
        == 0.0
    )


def test_baseline_run_rejects_official_metrics_and_bad_times() -> None:
    """Product-task baselines cannot be labeled as official lab results."""
    started = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    with pytest.raises(ValidationError, match="cannot be marked official"):
        _run(
            "qwen35-4b",
            metrics=[MetricResult(name="schema_validity", value=0.5, official=True)],
        )
    with pytest.raises(ValidationError, match="cannot be earlier"):
        _run("qwen35-4b", completed_at=started - timedelta(minutes=1))


def test_oom_backend_marks_run(tmp_path: Path) -> None:
    """A MemoryError is recorded so Gemma-style T4 failures stay visible."""

    class OomBackend:
        kind = "transformers"
        resolved_revision: str | None = "oom"

        def generate(self, prompt: str, settings: object) -> GenerationResult:
            del prompt, settings
            raise MemoryError

        def close(self) -> None:
            return None

    run = run_baseline(
        candidate_id="gemma3-12b",
        comparison_id="schema-full-4bit",
        backend=OomBackend(),
        output_dir=tmp_path,
        hardware="Google Colab T4",
        limit=1,
    )
    assert run.oom is True
    assert run.predictions[0].detail.startswith("generation raised MemoryError")


def test_transformers_backend_generate_with_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Colab generate() is unit-tested without CUDA wheels on the laptop."""

    class FakeTensor:
        def __init__(self, data: list[int]) -> None:
            self._data = data
            self.shape = (1, len(data))

        def to(self, device: object) -> FakeTensor:
            del device
            return self

        def __getitem__(self, item: object) -> FakeTensor:
            if item == 0:
                return FakeTensor(self._data)
            if isinstance(item, slice):
                return FakeTensor(self._data[item])
            return self

        def __len__(self) -> int:
            return len(self._data)

    class FakeTokenizer:
        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
            del kwargs
            return messages[0]["content"]

        def __call__(self, encoded: str, **kwargs: object) -> dict[str, FakeTensor]:
            del encoded, kwargs
            return {"input_ids": FakeTensor([1, 2, 3])}

        def decode(self, generated: FakeTensor, skip_special_tokens: bool = True) -> str:
            del generated, skip_special_tokens
            return '{"schema_version": "1.0.0"}'

    class FakeModel:
        device = "cpu"
        config = type("Cfg", (), {"_name_or_path": "mocked-revision"})()

        def parameters(self) -> object:
            return iter([FakeTensor([1])])

        def generate(self, **kwargs: object) -> FakeTensor:
            del kwargs
            return FakeTensor([1, 2, 3, 4, 5])

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()
        inference_mode = staticmethod(lambda: nullcontext())

        @staticmethod
        def float16() -> str:
            return "float16"

    monkeypatch.setattr(
        "leaseguard.inference.backend._load_transformers_model",
        lambda *args, **kwargs: (FakeTokenizer(), FakeModel()),
    )
    monkeypatch.setattr("leaseguard.inference.backend._require_torch", lambda: FakeTorch())
    monkeypatch.setattr("leaseguard.inference.backend._optional_torch", lambda: FakeTorch())
    candidate = get_candidate(load_baseline_registry(), "qwen35-4b")
    backend = TransformersBackend(candidate, "4bit", 1024)
    result = backend.generate("hello", load_baseline_registry().generation)
    assert "schema_version" in result.text
    sampled = load_baseline_registry().generation.model_copy(
        update={"do_sample": True, "temperature": 0.2}
    )
    sampled_result = backend.generate("hello", sampled)
    assert sampled_result.completion_tokens >= 0
    backend.close()


def test_transformers_generate_records_cuda_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """T4 peak VRAM is recorded when CUDA reports allocations."""

    class FakeTensor:
        def __init__(self, data: list[int]) -> None:
            self._data = data
            self.shape = (1, len(data))

        def to(self, device: object) -> FakeTensor:
            del device
            return self

        def __getitem__(self, item: object) -> FakeTensor:
            if item == 0:
                return FakeTensor(self._data)
            if isinstance(item, slice):
                return FakeTensor(self._data[item])
            return self

    class FakeTokenizer:
        def __call__(self, encoded: str, **kwargs: object) -> dict[str, FakeTensor]:
            del encoded, kwargs
            return {"input_ids": FakeTensor([1, 2, 3])}

        def decode(self, generated: FakeTensor, skip_special_tokens: bool = True) -> str:
            del generated, skip_special_tokens
            return "{}"

    class FakeModel:
        device = "cuda"
        config = type("Cfg", (), {"_name_or_path": "cuda-rev"})()

        def generate(self, **kwargs: object) -> FakeTensor:
            del kwargs
            return FakeTensor([1, 2, 3, 4])

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def reset_peak_memory_stats() -> None:
            return None

        @staticmethod
        def synchronize() -> None:
            return None

        @staticmethod
        def max_memory_allocated() -> int:
            return 2 * 1024 * 1024

        @staticmethod
        def empty_cache() -> None:
            return None

    class FakeTorch:
        cuda = FakeCuda()
        inference_mode = staticmethod(lambda: nullcontext())

    monkeypatch.setattr(
        "leaseguard.inference.backend._load_transformers_model",
        lambda *args, **kwargs: (FakeTokenizer(), FakeModel()),
    )
    monkeypatch.setattr("leaseguard.inference.backend._require_torch", lambda: FakeTorch())
    monkeypatch.setattr("leaseguard.inference.backend._optional_torch", lambda: FakeTorch())
    candidate = get_candidate(load_baseline_registry(), "qwen35-4b")
    backend = TransformersBackend(candidate, "4bit", 1024)
    result = backend.generate("hello", load_baseline_registry().generation)
    assert result.peak_vram_mb == pytest.approx(2.0)
    backend.close()


def test_load_transformers_model_uses_stubbed_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Colab model loading is exercised without installing CUDA wheels."""
    import sys
    import types

    from leaseguard.inference import backend as backend_mod

    torch_mod = types.ModuleType("torch")
    torch_mod.float16 = "fp16"  # type: ignore[attr-defined]
    transformers_mod = types.ModuleType("transformers")

    class BitsAndBytesConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> str:
            del args, kwargs
            return "tok"

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> str:
            del args, kwargs
            return "causal"

    class AutoModelForImageTextToText:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> str:
            del args, kwargs
            return "vlm"

    transformers_mod.BitsAndBytesConfig = BitsAndBytesConfig  # type: ignore[attr-defined]
    transformers_mod.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
    transformers_mod.AutoModelForCausalLM = AutoModelForCausalLM  # type: ignore[attr-defined]
    transformers_mod.AutoModelForImageTextToText = AutoModelForImageTextToText  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    candidate = get_candidate(load_baseline_registry(), "qwen35-4b")
    tokenizer, model = backend_mod._load_transformers_model(candidate, "4bit", 1024)
    assert tokenizer == "tok"
    assert model == "causal"
    _, eight_bit = backend_mod._load_transformers_model(candidate, "8bit", 1024)
    assert eight_bit == "causal"

    class Boom:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> str:
            del args, kwargs
            raise OSError("missing causal weights")

    transformers_mod.AutoModelForCausalLM = Boom  # type: ignore[attr-defined]
    _, vision = backend_mod._load_transformers_model(candidate, "4bit", 1024)
    assert vision == "vlm"


def test_backend_helpers_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing CUDA libraries stay a Colab-only error."""
    from leaseguard.inference.backend import (
        ModelBackendNotAvailableError,
        _apply_chat_template,
        _require_torch,
    )

    monkeypatch.setattr("leaseguard.inference.backend._optional_torch", lambda: None)
    with pytest.raises(ModelBackendNotAvailableError, match="torch is not installed"):
        _require_torch()
    assert _apply_chat_template(object(), "plain") == "plain"

    class EmptyTemplate:
        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
            del messages, kwargs
            return "   "

    assert _apply_chat_template(EmptyTemplate(), "plain") == "plain"

    class ThinkingOnly:
        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
            if "enable_thinking" in kwargs:
                raise TypeError("no thinking flag")
            return messages[0]["content"] + " /chat"

    assert _apply_chat_template(ThinkingOnly(), "plain") == "plain /chat"


def test_wrong_json_is_model_error_and_validation_json_is_validation_error() -> None:
    """Schema-valid wrong labels are model errors; invalid JSON objects are validation errors."""
    gold = gold_scripted_outputs()
    first_id = "ds1-extract-amendment"
    wrong = json.loads(gold[first_id])
    if wrong.get("monetary_terms"):
        wrong["monetary_terms"][0]["amount"] = "1"
    elif wrong.get("parties"):
        wrong["parties"][0]["name"] = "Someone Else LLC"
    backend = ScriptedBackend({**gold, first_id: json.dumps(wrong)})
    model_run = run_baseline(
        candidate_id="qwen35-4b",
        comparison_id="schema-full-4bit",
        backend=backend,
        hardware="cpu-test",
        limit=1,
    )
    assert model_run.predictions[0].error_class == "model_error"
    invalid = ScriptedBackend({**gold, first_id: '{"schema_version": "1.0.0"}'})
    validation_run = run_baseline(
        candidate_id="qwen35-4b",
        comparison_id="schema-full-4bit",
        backend=invalid,
        hardware="cpu-test",
        limit=1,
    )
    assert validation_run.predictions[0].error_class == "validation_error"


def test_disallowed_comparison_and_empty_eval_tasks(tmp_path: Path) -> None:
    """A candidate cannot run a comparison it was not configured for."""
    with pytest.raises(BaselineConfigError, match="not part of comparison"):
        run_baseline(
            candidate_id="gemma3-12b",
            comparison_id="schema-full-8bit",
            backend=ScriptedBackend({"x": "{}"}),
            hardware="cpu-test",
        )
    empty = tmp_path / "examples"
    empty.mkdir()
    (empty / "raw.json").write_text(
        (default_seed_examples_path() / "raw_domain_text" / "ds1-raw-retail.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    with pytest.raises(BaselineConfigError, match="no Dataset v1 evaluation tasks"):
        run_baseline(
            candidate_id="qwen35-4b",
            comparison_id="schema-full-4bit",
            backend=ScriptedBackend({"x": "{}"}),
            examples_root=empty,
            hardware="cpu-test",
        )


def test_eval_split_alignment_failures() -> None:
    """The frozen split file cannot drift from Dataset v1 hashing or holdout IDs."""
    from leaseguard.dataset.config import load_dataset_config
    from leaseguard.evaluation.baseline_models import SeedFamilySplit

    split = load_eval_split()
    split_method = split.model_copy()
    object.__setattr__(split_method, "method", "other")
    with pytest.raises(EvalSplitError, match="method"):
        assert_eval_split_matches_dataset(split_method, load_dataset_config())
    versioned = split.model_copy()
    object.__setattr__(versioned, "dataset_version", "9.9.9")
    with pytest.raises(EvalSplitError, match="dataset_version"):
        assert_eval_split_matches_dataset(versioned, load_dataset_config())
    blocked = split.model_copy(
        update={
            "blocked_from_training_document_ids": [
                *split.blocked_from_training_document_ids,
                "extra-id",
            ]
        }
    )
    with pytest.raises(EvalSplitError, match="blocked IDs"):
        assert_eval_split_matches_dataset(blocked, load_dataset_config())
    family = split.model_copy(
        update={
            "seed_families": [
                SeedFamilySplit(family_id="ds1-office-family", split="train"),
                SeedFamilySplit(family_id="ds1-retail-family", split="train"),
            ]
        }
    )
    with pytest.raises(EvalSplitError, match="frozen as train"):
        assert_eval_split_matches_dataset(family, load_dataset_config())
    assert_eval_split_not_used_for_development("local-regression")


def test_scoring_helpers_cover_quotes_and_missing_predictions() -> None:
    """Extraction and answer helpers handle empty predictions and gold quotes."""
    from leaseguard.evaluation.scoring import (
        content_is_correct,
        evidence_recall,
        predicted_quotes,
    )
    from leaseguard.evaluation.tasks import load_baseline_tasks

    task = next(
        item
        for item in load_baseline_tasks(default_seed_examples_path(), load_eval_split())
        if item.gold_extraction is not None
    )
    assert task.gold_extraction is not None
    quotes = predicted_quotes(task.gold_extraction, None)
    assert quotes
    answer_task = next(
        item
        for item in load_baseline_tasks(default_seed_examples_path(), load_eval_split())
        if item.gold_answer is not None
    )
    assert predicted_quotes(None, answer_task.gold_answer) == list(answer_task.gold_evidence_texts)
    assert evidence_recall(quotes, quotes) == 1.0
    assert (
        content_is_correct(
            predicted_extraction=None,
            gold_extraction=task.gold_extraction,
            predicted_answer=None,
            gold_answer=None,
        )
        is False
    )
    assert (
        content_is_correct(
            predicted_extraction=None,
            gold_extraction=None,
            predicted_answer=None,
            gold_answer=answer_task.gold_answer,
        )
        is False
    )
    assert (
        content_is_correct(
            predicted_extraction=None,
            gold_extraction=None,
            predicted_answer=None,
            gold_answer=None,
        )
        is False
    )


def test_resources_zero_speed_and_empty_aggregate() -> None:
    """Speed is zero when latency or completions are missing."""
    from leaseguard.inference.resources import aggregate_resources, resources_from_generation

    result = GenerationResult(
        text="ok", peak_vram_mb=1.0, latency_seconds=0.0, prompt_tokens=3, completion_tokens=2
    )
    resources = resources_from_generation(result, hardware="cpu", context_length_limit=128)
    assert resources.tokens_per_second == 0.0
    empty = aggregate_resources([], hardware="cpu", context_length_limit=128)
    assert empty.peak_vram_mb == 0.0


def test_retrieval_fallback_and_blank_checkpoint(tmp_path: Path) -> None:
    """Retrieval with no overlapping blocks falls back to the full document."""
    from leaseguard.evaluation.baseline import load_prediction_checkpoint, write_run_manifest
    from leaseguard.evaluation.baseline_models import RetrievalSettings

    bundle = build_context(
        {"doc": "Tenant shall pay rent."},
        mode="retrieval_assisted",
        query="zzzz not in document",
        retrieval=RetrievalSettings(top_k=1, min_score=1.0),
    )
    assert bundle.block_ids == ("retrieval_fallback",)
    blank = tmp_path / "empty.predictions.jsonl"
    blank.write_text("\n\n", encoding="utf-8")
    assert load_prediction_checkpoint(blank) == {}
    assert load_prediction_checkpoint(tmp_path / "missing.jsonl") == {}
    run = _run("qwen35-4b")
    write_run_manifest(tmp_path / "manifest.json", run)
    assert (tmp_path / "manifest.json").is_file()


def test_prompt_catalog_requires_templates(tmp_path: Path) -> None:
    """A prompt file cannot drop a required template name."""
    from leaseguard.inference.prompts import load_prompt_catalog

    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            {
                "prompt_version": "1.0.0",
                "frozen_on": "2026-09-14",
                "templates": {"direct_extraction": "x"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing templates"):
        load_prompt_catalog(path)


def test_invalid_answer_payload_and_parse_unclosed_fence() -> None:
    """Answer validation failures and unclosed fences stay parse-safe."""
    answer, _working = validate_answer_payload(
        {"schema_version": "1.0.0", "status": "answered", "explanation": "x", "confidence": 0.1},
        question="Q",
        perspective="neutral",
    )
    assert answer is None
    assert extract_json_object('```json\n{"a": 1}') == {"a": 1}


def test_hypothesis_variants_and_registry_validators() -> None:
    """Each measured error class and config invariant is documented."""
    from leaseguard.evaluation.baseline_models import CandidateRanking
    from leaseguard.evaluation.compare import fine_tuning_hypothesis

    base = CandidateRanking(
        candidate_id="qwen35-4b",
        practical=True,
        schema_validity=1.0,
        extraction_f1=1.0,
        answer_status_accuracy=1.0,
        evidence_recall=1.0,
        tokens_per_second=1.0,
        peak_vram_mb=4000.0,
        dominant_error_class="correct",
        notes="ok",
    )
    assert "seed holdout" in fine_tuning_hypothesis([base])
    for error_class, needle in {
        "parse_error": "JSON",
        "prompt_error": "Direct prompting",
        "validation_error": "ontology",
        "retrieval_error": "Phase 8",
        "model_error": "Supervised fine-tuning",
    }.items():
        ranked = base.model_copy(update={"dominant_error_class": error_class})
        assert needle in fine_tuning_hypothesis([ranked])
    payload = load_baseline_registry().model_dump(mode="json")
    payload["candidates"].append(payload["candidates"][0])
    with pytest.raises(ValidationError, match="candidate_id"):
        BaselineRegistry.model_validate(payload)
    payload = load_baseline_registry().model_dump(mode="json")
    payload["comparisons"].append(payload["comparisons"][0])
    with pytest.raises(ValidationError, match="comparison_id"):
        BaselineRegistry.model_validate(payload)
    payload = load_baseline_registry().model_dump(mode="json")
    payload["required_comparison_id"] = "missing-comparison"
    with pytest.raises(ValidationError, match="required_comparison_id"):
        BaselineRegistry.model_validate(payload)
    payload = load_baseline_registry().model_dump(mode="json")
    payload["comparisons"][0]["candidate_ids"].append(payload["comparisons"][0]["candidate_ids"][0])
    with pytest.raises(ValidationError, match="candidate_ids must be unique"):
        BaselineRegistry.model_validate(payload)
    payload = load_baseline_registry().model_dump(mode="json")
    payload["candidates"][1]["quantizations"] = ["4bit"]
    payload["comparisons"][-1]["candidate_ids"] = ["qwen35-9b"]
    payload["comparisons"][-1]["quantization"] = "8bit"
    with pytest.raises(ValidationError, match="not allowed"):
        BaselineRegistry.model_validate(payload)


def test_cli_error_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI reports missing run files and unknown candidates without crashing."""
    assert (
        main(
            [
                "baseline",
                "--candidate",
                "missing",
                "--comparison",
                "schema-full-4bit",
                "--backend",
                "scripted",
                "--script",
                str(tmp_path / "nope.json"),
            ]
        )
        == 1
    )
    assert main(["compare-baselines", str(tmp_path / "missing.json")]) == 1
    captured = capsys.readouterr()
    assert captured.err
