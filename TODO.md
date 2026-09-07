# LeaseGuard Project Checklist

This checklist tracks the implementation plan in [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md). Update it in the same commit as completed work so progress remains visible outside chat history.

Last reviewed: 2026-09-07

## Current Focus

Phase 2 is complete. Ontology version `1.0.0` is frozen. The next goal is Phase 3: define the evaluation policy, metrics, acceptance thresholds, and frozen evaluation-set structure before collecting training data.

## Phase 0 — Product Decisions

- [x] Select nationwide US document analysis as the initial jurisdiction scope.
- [x] Limit the initial document scope to commercial leases.
- [x] Prioritize office and retail leases.
- [x] Define property managers and real-estate operations teams as the primary users.
- [x] Select metadata and clause extraction as an initial task.
- [x] Select evidence-based questions and answers as an initial task.
- [x] Select obligation tracking as an initial task.
- [x] Support landlord, tenant, and neutral perspectives selected by the user.
- [x] Require simple explanations, validated JSON, and source citations.
- [x] Limit initial input to English digital PDF and DOCX files.
- [x] Require local processing without reusing user documents for training.
- [x] Define LeaseGuard as decision support rather than direct legal advice.
- [x] Defer state-law conclusions, OCR, redlining, and risk scoring.
- [x] Record the approved scope in `docs/scope.md`.

**Completion gate:** The users, documents, outputs, jurisdiction boundary, safety limits, and non-goals are documented.

## Phase 1 — Repository Foundation

- [x] Create the `src/leaseguard` Python package.
- [x] Create directories for configuration, data, documentation, scripts, notebooks, tests, and artifacts.
- [x] Add Python 3.11 and 3.12 project configuration.
- [x] Add reproducible dependency locking with `uv`.
- [x] Configure Ruff formatting and linting.
- [x] Configure strict mypy type checking.
- [x] Configure pytest and coverage reporting.
- [x] Configure pre-commit checks.
- [x] Add GitHub Actions quality checks.
- [x] Add `CONTRIBUTING.md`.
- [x] Add `SECURITY.md`.
- [x] Add data privacy and large-artifact exclusions to `.gitignore`.
- [x] Document the repository structure and development commands.
- [x] Verify formatting, linting, typing, tests, and pre-commit hooks.

**Completion gate:** A contributor can install the project and run all documented checks.

## Phase 2 — Schemas and Lease Ontology

- [x] Define ontology version `1.0.0`.
- [x] Define supported office, retail, amendment, exhibit, and related document types.
- [x] Define party roles.
- [x] Define landlord, tenant, and neutral perspectives.
- [x] Define the initial clause taxonomy.
- [x] Define evidence-span rules and page references.
- [x] Define monetary and date term records.
- [x] Define obligation records and timing types.
- [x] Define answer statuses and abstention behavior.
- [x] Implement strict Pydantic models.
- [x] Export versioned JSON Schemas.
- [x] Add synthetic extraction and answer examples.
- [x] Test validation rules and generated-schema synchronization.
- [x] Collect a small validation set of reusable office leases.
- [x] Collect a small validation set of reusable retail-capable lease templates.
- [x] Add a municipal retail-like lease for local ontology validation while its reuse rights are reviewed.
- [x] Include at least one amendment and related template family.
- [x] Label representative documents using ontology version `1.0.0`.
- [x] Record concepts that do not fit the initial ontology.
- [x] Review whether clause labels are too broad, narrow, or overlapping.
- [x] Review obligation timing against real lease wording.
- [x] Define evidence-offset behavior and defer exact offset checks to the Phase 4 parser.
- [x] Revise models, examples, and documentation from validation findings.
- [x] Record that no migration is required because version `1.0.0` has not been released.
- [x] Freeze the validated ontology and schemas for Dataset v1.

**Completion gate:** Several office and retail leases can be labelled consistently without unsupported fields or ambiguous labels.

## Phase 3 — Evaluation Before Training

- [ ] Write the evaluation-set inclusion and exclusion rules.
- [ ] Define agreement-family split rules.
- [ ] Define extraction precision, recall, and F1.
- [ ] Define evidence-span accuracy.
- [ ] Define structured-output validity.
- [ ] Define date and monetary accuracy.
- [ ] Define obligation and responsible-party accuracy.
- [ ] Define unsupported-claim and correct-abstention measures.
- [ ] Define conflicting-provision and amendment-handling measures.
- [ ] Select acceptance thresholds for each important metric.
- [ ] Collect 50–100 representative evaluation leases.
- [ ] Include missing, conflicting, unusual, and table-based provisions.
- [ ] Include both landlord-friendly and tenant-friendly language.
- [ ] Create expert-reviewed reference annotations.
- [ ] Build a repeatable evaluation command.
- [ ] Save machine-readable evaluation reports.
- [ ] Prevent evaluation documents from entering training or synthetic generation.

**Completion gate:** Any candidate model can be evaluated with one command and compared against saved results.

## Phase 4 — Document Pipeline

- [ ] Create one Google Colab T4 notebook that selects and runs every heavy task.
- [ ] Keep the Colab notebook as a thin caller of package code and versioned configurations.
- [ ] Mount approved remote storage for corpora, checkpoints, and generated artifacts.
- [ ] Add checkpoint and resume behavior for Colab session limits.
- [ ] Define the source-manifest format.
- [ ] Record source URL, license, date, jurisdiction, and checksum.
- [ ] Implement approved document downloading and local importing.
- [ ] Validate file type and file integrity.
- [ ] Parse digital PDF files while preserving page boundaries.
- [ ] Parse DOCX files while preserving headings and tables.
- [ ] Keep raw documents unchanged.
- [ ] Create separate normalized and labelled records.
- [ ] Segment documents into headings, sections, clauses, tables, and exhibits.
- [ ] Detect exact duplicates.
- [ ] Detect near-duplicate filings.
- [ ] Link leases, amendments, exhibits, and related documents by family.
- [ ] Detect and redact private information where required.
- [ ] Validate processed records against the Phase 2 schemas.
- [ ] Add pipeline unit and integration tests.
- [ ] Produce processing reports with errors and warnings.

**Completion gate:** An approved source can be reproduced from its manifest entry as a validated intermediate record.

## Phase 5 — Dataset Version 1

- [ ] Write Dataset v1 inclusion, exclusion, and quality rules.
- [ ] Build the raw lease-domain text subset.
- [ ] Build the structured extraction subset.
- [ ] Build the evidence-grounded conversation subset.
- [ ] Build the refusal and uncertainty subset.
- [ ] Build the multi-turn conversation subset.
- [ ] Build preference pairs only from clearly ranked responses.
- [ ] Validate all records against versioned schemas.
- [ ] Verify that quoted evidence exists in the source text.
- [ ] Remove duplicate and near-duplicate examples.
- [ ] Split by complete agreement family.
- [ ] Check for leakage across train, validation, and test sets.
- [ ] Track dataset sources and licenses.
- [ ] Generate checksums and dataset statistics.
- [ ] Perform manual quality sampling.
- [ ] Write the Dataset v1 card and limitations.
- [ ] Publish or store the dataset without committing large files to Git.

**Completion gate:** Dataset v1 is documented, licensed, validated, reproducible, and leakage-safe.

## Phase 6 — Base-Model Evaluation

- [ ] Freeze the Dataset v1 evaluation split.
- [ ] Define one reproducible baseline configuration per candidate.
- [ ] Evaluate Qwen3.5-4B.
- [ ] Evaluate Qwen3.5-9B.
- [ ] Evaluate Gemma 3 12B.
- [ ] Compare direct prompting.
- [ ] Compare schema-constrained prompting.
- [ ] Compare clause-level context.
- [ ] Compare retrieval-assisted context.
- [ ] Compare practical quantized inference.
- [ ] Record memory use, speed, context length, and output quality.
- [ ] Separate model errors from parsing, prompt, retrieval, and validation errors.
- [ ] Select the strongest practical base model.
- [ ] Publish a baseline report and fine-tuning hypothesis.

**Completion gate:** The selected base model and the weaknesses that fine-tuning must address are supported by measured results.

## Phase 7 — First Supervised Fine-Tune

- [ ] Create the initial QLoRA training configuration.
- [ ] Pin the base model name and exact revision.
- [ ] Pin Dataset v1 and its checksums.
- [ ] Record seeds and environment details.
- [ ] Train first on structured extraction and evidence-grounded answers.
- [ ] Include safe refusal and uncertainty examples.
- [ ] Save resumable checkpoints for free GPU session limits.
- [ ] Track training and validation loss.
- [ ] Evaluate every candidate checkpoint with the frozen evaluation set.
- [ ] Compare the adapter against the unmodified base model.
- [ ] Check unsupported claims and missed clauses for regressions.
- [ ] Test landlord, tenant, and neutral perspective consistency.
- [ ] Select the best checkpoint using predefined metrics.
- [ ] Write the first model card and training report.

**Completion gate:** Fine-tuning improves target metrics without harming grounding, abstention, or important clause recall.

## Phase 8 — Retrieval and Deterministic Validation

- [ ] Define the lease chunking and retrieval strategy.
- [ ] Add exact keyword or BM25 retrieval.
- [ ] Select and evaluate a free legal-domain embedding model.
- [ ] Add semantic retrieval.
- [ ] Add reranking only if it produces a measured improvement.
- [ ] Preserve page and section references through retrieval.
- [ ] Validate structured output against JSON Schema.
- [ ] Validate dates and monetary values with deterministic code.
- [ ] Verify that quotations exist in the source document.
- [ ] Reject or warn about incorrect citations.
- [ ] Detect conflicts across clauses and amendments.
- [ ] Add a dated official-source knowledge base only after defining its scope.
- [ ] Keep state-law guidance disabled until separately evaluated.
- [ ] Produce an auditable result containing model, data, and source versions.

**Completion gate:** Important conclusions are traceable to lease text or an approved dated source, and invalid citations are rejected.

## Phase 9 — Advanced Training

- [ ] Review measured weaknesses remaining after supervised fine-tuning.
- [ ] Write a clear hypothesis before each advanced experiment.
- [ ] Consider preference optimization for response-quality problems.
- [ ] Consider continued pretraining for measured domain-language problems.
- [ ] Consider GRPO only for automatically verifiable rewards.
- [ ] Consider long-context adaptation only if retrieval is insufficient.
- [ ] Consider distillation only when a larger validated model exists.
- [ ] Compare every experiment against the same frozen evaluation.
- [ ] Reject methods that improve appearance while reducing correctness.
- [ ] Document the benefit and cost of every retained method.

**Completion gate:** Every advanced method included in the project has a reproducible, measured benefit.

## Phase 10 — First Public Release

- [ ] Freeze the release code version.
- [ ] Freeze the dataset version.
- [ ] Freeze the model and adapter version.
- [ ] Export a practical local quantization.
- [ ] Verify local inference without paid APIs.
- [ ] Publish the model card.
- [ ] Publish the dataset card.
- [ ] Publish the evaluation and safety reports.
- [ ] Document privacy behavior and known limitations.
- [ ] Add a reproducible local inference example.
- [ ] Test installation from a clean environment.
- [ ] Test representative office and retail leases end to end.
- [ ] Add release notes and upgrade guidance.
- [ ] Tag the reproducible `v1.0` release.

**Completion gate:** Another person can run LeaseGuard locally and reproduce its documented evaluation without a paid API.

## Continuous Project Tasks

- [ ] Update this checklist in the same commit as completed work.
- [ ] Keep `docs/DEVELOPMENT.md` aligned with scope or phase changes.
- [ ] Run formatting, linting, typing, tests, and pre-commit before merging.
- [ ] Add tests for every bug fix and schema rule.
- [ ] Record licenses for every data source and model.
- [ ] Keep private leases, secrets, datasets, and model weights out of Git.
- [ ] Review dependency updates and security reports.
- [ ] Record breaking changes and migration instructions.
