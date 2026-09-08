"""Tests for Dataset v1 rules, leakage checks, and the seed bundle."""

import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from leaseguard.dataset.build import (
    build_from_examples,
    build_seed_dataset,
    compute_checksums,
    export_bundle,
    verify_bundle_checksums,
)
from leaseguard.dataset.cli import main, write_report
from leaseguard.dataset.config import (
    DatasetConfigError,
    assert_holdout_alignment,
    load_dataset_config,
)
from leaseguard.dataset.eligibility import (
    DatasetEligibilityError,
    assert_record_eligible,
    canonical_text,
    exact_duplicate_ids,
    near_duplicate_groups,
    record_is_training_eligible,
)
from leaseguard.dataset.evidence import evidence_failures
from leaseguard.dataset.examples import DatasetExampleError, load_examples, subset_records
from leaseguard.dataset.models import (
    ConversationTurn,
    DatasetConfig,
    EvidenceConversationRecord,
    MultiTurnRecord,
    PreferencePairRecord,
    RawDomainRecord,
    RefusalRecord,
    SplitRules,
    StructuredExtractionRecord,
)
from leaseguard.dataset.paths import default_seed_examples_path
from leaseguard.dataset.quality import reviews_are_complete, select_quality_samples
from leaseguard.dataset.splits import assign_split, family_bucket
from leaseguard.dataset.validate import collection_issues, record_issues
from leaseguard.ontology.enums import AnswerStatus, RightsStatus
from leaseguard.ontology.models import EvidenceSpan


def _seed_payload(name: str) -> dict[str, Any]:
    path = default_seed_examples_path() / "raw_domain_text" / "ds1-raw-office.json"
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    payload["record_id"] = name
    return payload


def test_seed_dataset_builds_and_passes_quality_review() -> None:
    """The committed seed covers every subset, checksums, and accepted samples."""
    bundle, report = build_seed_dataset()

    assert report.passed
    assert bundle is not None
    assert bundle.statistics.record_count == 10
    assert bundle.statistics.family_count == 2
    assert reviews_are_complete(bundle.quality_samples)
    assert not verify_bundle_checksums(bundle, load_dataset_config())
    office_splits = {
        record.split
        for record in (
            *bundle.raw_domain_text,
            *bundle.structured_extraction,
            *bundle.evidence_conversation,
        )
        if record.family_id == "ds1-office-family"
    }
    assert office_splits == {assign_split("ds1-office-family", load_dataset_config().splits)}


def test_holdout_alignment_requires_regression_ids() -> None:
    """Dataset v1 cannot drop a blocked regression document identifier."""
    config = load_dataset_config()
    payload = config.model_dump(mode="json")
    payload["exclusion"]["blocked_from_training_document_ids"] = ["ds1-office-lease"]
    narrowed = DatasetConfig.model_validate(payload)

    with pytest.raises(DatasetConfigError, match="gsa-l100-2026"):
        assert_holdout_alignment(narrowed)


def test_blocked_document_cannot_enter_dataset(tmp_path: Path) -> None:
    """Regression holdout identifiers stay out of Dataset v1."""
    payload = _seed_payload("blocked-raw")
    office_text = cast(dict[str, str], payload["source_texts"])["ds1-office-lease"]
    payload["document_ids"] = ["gsa-l100-2026"]
    payload["source_texts"] = {"gsa-l100-2026": office_text}
    example = tmp_path / "raw.json"
    example.write_text(json.dumps(payload), encoding="utf-8")

    bundle, report = build_from_examples(tmp_path, load_dataset_config())

    assert bundle is None
    assert any(issue.code == "eligibility" for issue in report.issues)


def test_quoted_evidence_must_occur_in_source() -> None:
    """Character offsets and quoted text are checked against registered source text."""
    source = {"doc-a": "Tenant shall pay rent monthly."}
    missing = EvidenceSpan(document_id="doc-a", text="late fee of 5%", section="Fees")
    offset = EvidenceSpan(
        document_id="doc-a",
        text="pay rent",
        start_char=0,
        end_char=8,
    )

    assert evidence_failures(missing, source)
    assert evidence_failures(offset, source)
    assert not evidence_failures(
        EvidenceSpan(document_id="doc-a", text="pay rent monthly", section="Rent"),
        source,
    )
    assert evidence_failures(
        EvidenceSpan(document_id="doc-a", text="pay rent monthly", section="Rent"),
        {},
    )


def test_family_hash_is_stable() -> None:
    """The same family always lands in the same split bucket."""
    rules = SplitRules(
        method="family_hash",
        salt="leaseguard-dataset-v1",
        train_percent=70,
        validation_percent=15,
        test_percent=15,
    )

    assert family_bucket("ds1-office-family", rules.salt) == family_bucket(
        "ds1-office-family", rules.salt
    )


def test_family_hash_covers_every_split() -> None:
    """Train, validation, and test buckets are all reachable."""
    rules = SplitRules(
        method="family_hash",
        salt="leaseguard-dataset-v1",
        train_percent=70,
        validation_percent=15,
        test_percent=15,
    )
    seen: set[str] = set()
    index = 0
    while len(seen) < 3 and index < 500:
        seen.add(assign_split(f"family-{index}", rules))
        index += 1
    assert seen == {"train", "validation", "test"}


def test_exact_duplicates_are_reported(tmp_path: Path) -> None:
    """Identical canonical examples cannot both remain."""
    first = json.loads(
        (default_seed_examples_path() / "raw_domain_text" / "ds1-raw-office.json").read_text(
            encoding="utf-8"
        )
    )
    second = dict(first)
    second["record_id"] = "ds1-raw-office-copy"
    (tmp_path / "a.json").write_text(json.dumps(first), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(second), encoding="utf-8")
    records = load_examples(tmp_path)

    duplicates = exact_duplicate_ids(records)
    assert duplicates["ds1-raw-office-copy"] == "ds1-raw-office"
    assert "ds1-office-lease" in canonical_text(records[0])


def test_family_split_leakage_is_rejected() -> None:
    """A lease and its amendment cannot be placed in two splits."""
    records = load_examples(default_seed_examples_path())
    config = load_dataset_config()
    for record in records:
        record.split = "train"
    next(record for record in records if record.record_id == "ds1-raw-office").split = "train"
    next(record for record in records if record.record_id == "ds1-extract-amendment").split = "test"
    issues = collection_issues(records, config)

    assert any(issue.code == "family_split_leakage" for issue in issues)


def test_preference_pair_rejects_invented_preferred_answer() -> None:
    """The preferred side of a pair cannot invent legal facts."""
    payload = json.loads(
        (default_seed_examples_path() / "preference_pairs" / "ds1-pref-late-fee.json").read_text(
            encoding="utf-8"
        )
    )
    payload["preferred"]["invented_facts"] = True

    with pytest.raises(ValidationError, match="cannot invent"):
        PreferencePairRecord.model_validate(payload)


def test_record_validators_and_quality_issues() -> None:
    """Subset-specific rules catch invented evidence, refusals, and identity drift."""
    extraction_payload = json.loads(
        (
            default_seed_examples_path() / "structured_extraction" / "ds1-extract-office.json"
        ).read_text(encoding="utf-8")
    )
    extraction_payload["family_id"] = "other-family"
    with pytest.raises(ValidationError, match="family_id"):
        StructuredExtractionRecord.model_validate(extraction_payload)

    multi_payload = json.loads(
        (default_seed_examples_path() / "multi_turn" / "ds1-multiturn-office.json").read_text(
            encoding="utf-8"
        )
    )
    multi_payload["turns"][0]["role"] = "assistant"
    multi_payload["turns"][0]["answer"] = multi_payload["turns"][1]["answer"]
    with pytest.raises(ValidationError):
        MultiTurnRecord.model_validate(multi_payload)

    with pytest.raises(ValidationError, match="assistant turns"):
        ConversationTurn.model_validate({"role": "assistant", "text": "Hello"})
    with pytest.raises(ValidationError, match="user turns"):
        ConversationTurn.model_validate(
            {
                "role": "user",
                "text": "Hello",
                "answer": json.loads(
                    (
                        default_seed_examples_path()
                        / "evidence_conversation"
                        / "ds1-qa-office-rent.json"
                    ).read_text(encoding="utf-8")
                )["answer"],
            }
        )

    records = load_examples(default_seed_examples_path())
    config = load_dataset_config()
    conversation = next(record for record in records if record.record_id == "ds1-qa-office-rent")
    assert isinstance(conversation, EvidenceConversationRecord)
    conversation.answer.evidence[0].text = "this quote is not in the lease"
    assert any(issue.code == "evidence" for issue in record_issues(conversation, config))

    records = load_examples(default_seed_examples_path())
    extraction = next(record for record in records if record.record_id == "ds1-extract-office")
    assert isinstance(extraction, StructuredExtractionRecord)
    extraction.extraction.clauses[0].text = "missing clause text"
    assert any(issue.code == "clause_text" for issue in record_issues(extraction, config))

    records = load_examples(default_seed_examples_path())
    refusal = next(record for record in records if record.record_id == "ds1-refuse-state-law")
    office = next(record for record in records if record.record_id == "ds1-qa-office-rent")
    assert isinstance(refusal, RefusalRecord)
    assert isinstance(office, EvidenceConversationRecord)
    refusal.answer = refusal.answer.model_copy(
        update={"status": AnswerStatus.ANSWERED, "evidence": office.answer.evidence}
    )
    assert any(issue.code == "refusal_status" for issue in record_issues(refusal, config))

    duplicate_ids = load_examples(default_seed_examples_path())
    duplicate_ids[1].record_id = duplicate_ids[0].record_id
    assert any(issue.code == "record_id" for issue in collection_issues(duplicate_ids, config))


def test_load_examples_requires_json_objects(tmp_path: Path) -> None:
    """Non-object example files fail with a dataset-specific error."""
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    with pytest.raises(DatasetExampleError):
        load_examples(tmp_path)
    array_dir = tmp_path / "array"
    array_dir.mkdir()
    (array_dir / "bad.json").write_text("[1]", encoding="utf-8")
    with pytest.raises(DatasetExampleError, match="JSON object"):
        load_examples(array_dir)
    with pytest.raises(DatasetExampleError, match="does not exist"):
        load_examples(tmp_path / "missing")


def test_cli_validate_config_and_seed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The Dataset v1 CLI validates rules, the seed, and exported statistics."""
    assert main(["validate-config"]) == 0
    assert main(["validate-seed"]) == 0
    output = tmp_path / "dataset.v1.json"
    assert main(["build", "--output", str(output)]) == 0
    assert output.is_file()
    assert main(["stats", "--bundle", str(output)]) == 0
    captured = capsys.readouterr()
    assert "record_count" in captured.out
    bundle, report = build_seed_dataset()
    assert bundle is not None
    write_report(tmp_path / "report.json", report)
    export_bundle(tmp_path / "copy.json", bundle)
    assert (tmp_path / "report.json").is_file()


def test_cli_rejects_broken_config(tmp_path: Path) -> None:
    """A config that drops holdout IDs fails before a build starts."""
    config = load_dataset_config()
    payload = config.model_dump(mode="json")
    payload["exclusion"]["blocked_from_training_document_ids"] = ["only-this"]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["--config", str(path), "validate-config"]) == 1


def test_quality_sample_overlay_and_subset_filter() -> None:
    """Manual review overlays replace the pending status for sampled IDs."""
    records = load_examples(default_seed_examples_path())
    config = load_dataset_config()
    for record in records:
        record.split = "train"
    samples = select_quality_samples(
        records,
        config,
        {"ds1-raw-office": {"review_status": "rejected", "notes": "flagged"}},
    )
    office = next(sample for sample in samples if sample.record_id == "ds1-raw-office")
    assert office.review_status == "rejected"
    assert not reviews_are_complete(samples)
    raw = subset_records(records, "raw_domain_text")
    assert {record.record_id for record in raw} == {"ds1-raw-office", "ds1-raw-retail"}


def test_checksum_mismatch_is_detected() -> None:
    """Tampering with subset hashes fails verification."""
    bundle, report = build_seed_dataset()
    assert bundle is not None
    assert report.passed
    tampered = bundle.model_copy(
        update={"checksums": bundle.checksums.model_copy(update={"records_sha256": "0" * 64})}
    )
    issues = verify_bundle_checksums(tampered, load_dataset_config())
    assert issues
    expected = compute_checksums(
        load_dataset_config(),
        sorted(
            [
                *bundle.raw_domain_text,
                *bundle.structured_extraction,
                *bundle.evidence_conversation,
                *bundle.refusal_uncertainty,
                *bundle.multi_turn,
                *bundle.preference_pairs,
            ],
            key=lambda item: (item.subset, item.record_id),
        ),
    )
    assert expected.records_sha256 == bundle.checksums.records_sha256


def test_empty_subset_fails_build(tmp_path: Path) -> None:
    """Dataset v1 must contain every required subset."""
    dest = tmp_path / "examples"
    dest.mkdir()
    for path in (default_seed_examples_path() / "raw_domain_text").glob("*.json"):
        dest.joinpath(path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    bundle, report = build_from_examples(dest, load_dataset_config())

    assert bundle is None
    assert any(issue.code == "empty_subset" for issue in report.issues)


def test_unknown_subset_and_empty_directory(tmp_path: Path) -> None:
    """Example loading rejects unknown tags and empty folders."""
    (tmp_path / "x.json").write_text('{"subset": "unknown"}', encoding="utf-8")
    with pytest.raises(DatasetExampleError, match="unknown"):
        load_examples(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DatasetExampleError, match="no JSON"):
        load_examples(empty)


def test_document_family_and_near_duplicate_leakage() -> None:
    """The same document cannot sit in two families; near-duplicates cannot cross families."""
    records = load_examples(default_seed_examples_path())
    retail = next(record for record in records if record.record_id == "ds1-raw-retail")
    office = next(record for record in records if record.record_id == "ds1-raw-office")
    retail.document_ids = ["ds1-office-lease"]
    retail.source_texts = {"ds1-office-lease": office.source_texts["ds1-office-lease"]}
    issues = collection_issues(records, load_dataset_config())
    assert any(issue.code == "document_family_leakage" for issue in issues)

    records = load_examples(default_seed_examples_path())
    retail = next(record for record in records if record.record_id == "ds1-raw-retail")
    office = next(record for record in records if record.record_id == "ds1-raw-office")
    retail.source_texts = {
        "ds1-retail-lease": office.source_texts["ds1-office-lease"]
        + " "
        + office.source_texts["ds1-office-lease"]
    }
    leaks = near_duplicate_groups(
        [office, retail],
        threshold=0.5,
        shingle_size=5,
    )
    assert leaks


def test_config_rejects_incomplete_split_and_subset_lists() -> None:
    """Frozen Dataset v1 rules must stay complete."""
    payload = load_dataset_config().model_dump(mode="json")
    payload["subsets"] = ["raw_domain_text"]
    with pytest.raises(ValidationError, match="every required subset"):
        DatasetConfig.model_validate(payload)
    payload = load_dataset_config().model_dump(mode="json")
    payload["splits"]["test_percent"] = 1
    with pytest.raises(ValidationError, match="sum to 100"):
        DatasetConfig.model_validate(payload)


def test_record_envelope_validators() -> None:
    """Identity, uniqueness, and ranking validators reject inconsistent records."""
    payload = load_dataset_config().model_dump(mode="json")
    payload["exclusion"]["blocked_from_training_document_ids"].append(
        payload["exclusion"]["blocked_from_training_document_ids"][0]
    )
    with pytest.raises(ValidationError, match="unique"):
        DatasetConfig.model_validate(payload)

    payload = load_dataset_config().model_dump(mode="json")
    payload["subsets"] = [*payload["subsets"], payload["subsets"][0]]
    with pytest.raises(ValidationError, match="unique"):
        DatasetConfig.model_validate(payload)

    raw = json.loads(
        (default_seed_examples_path() / "raw_domain_text" / "ds1-raw-office.json").read_text(
            encoding="utf-8"
        )
    )
    missing = dict(raw)
    missing["source_texts"] = {"other-doc": raw["source_texts"]["ds1-office-lease"]}
    with pytest.raises(ValidationError, match="missing"):
        RawDomainRecord.model_validate(missing)
    extra = dict(raw)
    extra["source_texts"] = {**raw["source_texts"], "extra-doc": "x" * 80}
    with pytest.raises(ValidationError, match="unused"):
        RawDomainRecord.model_validate(extra)

    extraction = json.loads(
        (
            default_seed_examples_path() / "structured_extraction" / "ds1-extract-office.json"
        ).read_text(encoding="utf-8")
    )
    extraction["document_ids"] = ["other-doc"]
    extraction["source_texts"] = {"other-doc": extraction["source_texts"]["ds1-office-lease"]}
    with pytest.raises(ValidationError, match="document_id"):
        StructuredExtractionRecord.model_validate(extraction)

    pref = json.loads(
        (default_seed_examples_path() / "preference_pairs" / "ds1-pref-late-fee.json").read_text(
            encoding="utf-8"
        )
    )
    pref["rejected"]["text"] = pref["preferred"]["text"]
    with pytest.raises(ValidationError, match="must differ"):
        PreferencePairRecord.model_validate(pref)

    multi = json.loads(
        (default_seed_examples_path() / "multi_turn" / "ds1-multiturn-office.json").read_text(
            encoding="utf-8"
        )
    )
    multi["turns"][3] = {"role": "user", "text": "and again?"}
    with pytest.raises(ValidationError, match="alternate"):
        MultiTurnRecord.model_validate(multi)

    office = next(
        record
        for record in load_examples(default_seed_examples_path())
        if record.record_id == "ds1-raw-office"
    )
    narrowed = load_dataset_config().model_dump(mode="json")
    narrowed["inclusion"]["allowed_rights"] = ["public_domain"]
    assert not record_is_training_eligible(office, DatasetConfig.model_validate(narrowed))


def test_cli_stats_and_tampered_bundle(tmp_path: Path) -> None:
    """Stats reads the seed and rejects a bundle whose checksums no longer match."""
    assert main(["stats"]) == 0
    bundle, report = build_seed_dataset()
    assert bundle is not None and report.passed
    tampered = bundle.model_copy(
        update={"checksums": bundle.checksums.model_copy(update={"config_sha256": "0" * 64})}
    )
    path = tmp_path / "tampered.json"
    export_bundle(path, tampered)
    assert main(["stats", "--bundle", str(path)]) == 1
    assert main(["stats", "--bundle", str(tmp_path / "missing.json")]) == 1
    subset_tampered = bundle.model_copy(
        update={
            "checksums": bundle.checksums.model_copy(
                update={"subsets": {"raw_domain_text": "0" * 64}}
            )
        }
    )
    issues = verify_bundle_checksums(subset_tampered, load_dataset_config())
    assert any("subset" in issue.message for issue in issues)

    records = load_examples(default_seed_examples_path())
    office = next(record for record in records if record.record_id == "ds1-raw-office")
    blocked = office.model_copy(update={"rights_status": RightsStatus.REVIEW_REQUIRED})
    assert not record_is_training_eligible(blocked, load_dataset_config())
    short = office.model_copy(update={"source_texts": {"ds1-office-lease": "too short"}})
    with pytest.raises(DatasetEligibilityError, match="shorter"):
        assert_record_eligible(short, load_dataset_config())
