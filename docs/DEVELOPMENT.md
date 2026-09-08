# LeaseGuard Development Plan

This document explains how LeaseGuard should be built in a clear and repeatable order. The project will begin with a reliable software and data foundation, then create an evaluated dataset, and only then fine-tune a model.

The order matters. Training a model before defining the expected outputs and evaluation rules would make it difficult to tell whether the model is actually improving.

Current implementation progress is tracked in [`TODO.md`](../TODO.md). Update that checklist in the same commit as completed roadmap work.

## Project Principles

LeaseGuard should remain private, evidence-based, and affordable to run. Important conclusions must point back to the exact lease text or to a dated official source. The system must also say when the available information is unclear or incomplete.

Reusable logic belongs in the `src/leaseguard` package. Scripts and notebooks should call that package instead of containing separate copies of the same logic. Data, model weights, and generated artifacts must be versioned independently from the application code.

Heavy work runs through one Google Colab notebook on a T4 GPU. The notebook is an orchestrator: it clones the repository and calls versioned package functions and configurations. Full datasets, model weights, CUDA dependencies, Unsloth, and checkpoints stay out of the local development machine.

## Repository Structure

```text
LeaseGuard/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   └── inference/
├── data/
│   ├── README.md
│   ├── manifests/
│   ├── samples/
│   └── schemas/
├── docs/
│   ├── DEVELOPMENT.md
│   ├── architecture.md
│   ├── dataset.md
│   ├── evaluation.md
│   ├── safety.md
│   └── scope.md
├── notebooks/
├── scripts/
├── src/leaseguard/
│   ├── dataset/
│   ├── evaluation/
│   ├── inference/
│   ├── ingestion/
│   ├── ontology/
│   ├── preprocessing/
│   ├── privacy/
│   ├── retrieval/
│   └── training/
└── tests/
    ├── fixtures/
    ├── integration/
    └── unit/
```

Only small examples, schemas, and source manifests should be committed under `data`. Downloaded leases, private annotations, model checkpoints, and generated outputs should remain outside Git.

## Phase 0 — Define the Product

The first step is to decide exactly what LeaseGuard will and will not do. This includes selecting the first jurisdiction, deciding whether the initial release supports commercial or residential leases, and identifying the main user.

The first release should focus on a small group of measurable tasks: lease metadata extraction, clause extraction with evidence, and document-grounded questions and answers. Automated redlining and legal risk scoring should come later because they require stronger evidence, careful stakeholder context, and expert review.

Phase 0 is complete when the supported users, documents, outputs, jurisdictions, safety limits, and non-goals are recorded in `docs/scope.md`.

Phase 0 is complete for the first release. The approved scope is nationwide US document analysis for English office and retail leases. The primary users are property managers and real-estate operations teams. Initial tasks are extraction, evidence-based questions and answers, and obligation tracking. State-law conclusions, OCR, redlining, and direct legal advice are deferred.

## Phase 1 — Build the Repository Foundation

This phase makes the repository easy to install, test, understand, and contribute to. It creates the Python package, dependency configuration, source layout, test layout, configuration directories, contribution guidance, and continuous integration.

The project uses `uv` to create reproducible environments. Ruff checks formatting and code quality, mypy checks types, and pytest runs tests. These same checks run in GitHub Actions so that local development and pull requests follow the same rules.

Phase 1 is complete when a new contributor can clone the repository, install the development environment, and run all quality checks using the documented commands.

## Phase 2 — Define Schemas and the Lease Ontology

Before collecting a large dataset, LeaseGuard needs a stable definition of what it extracts. The ontology will define known lease concepts such as parties, premises, rent, dates, renewals, assignment, repairs, insurance, default, termination, and dispute terms.

Structured schemas will define document metadata, evidence spans, clauses, obligations, risks, and model responses. Every extracted finding should include the answer, exact supporting text, page or clause reference, affected party, and uncertainty information.

Phase 2 is complete when several different leases can be labelled consistently with the same schemas and ontology.

Phase 2 is complete. Ontology version `1.0.0` was validated against reusable office and retail-capable templates, a municipal retail-like lease, and an amendment template family. Strict Python models and exported JSON Schemas cover provenance and rights, parties, premises, multi-label clauses, monetary and date terms, obligations, amendment effects, evidence spans, and document-grounded answers. The findings and remaining dataset licensing boundary are recorded in `docs/phase2-validation.md`.

## Phase 3 — Create Evaluation Before Training

Phase 3 freezes how LeaseGuard is judged before any training data is collected. A high score on a private lease set created by this project is not a breakthrough. The headline standard is a published, professionally constructed benchmark that frontier and large labs already use, scored with that benchmark's official test split and official evaluator.

The frozen professional suite is:

- LegalBench (NeurIPS 2023 Datasets and Benchmarks) for legal reasoning
- CUAD (NeurIPS 2021) for contract clause extraction
- ContractNLI (Findings of EMNLP 2021) for evidence-grounded contract inference
- LegalBench-RAG for legal retrieval

Numeric acceptance thresholds are not invented here. They will be frozen in Phase 6 after unmodified base models are scored on these official protocols.

A small internal office and retail regression suite still exists. It checks schema validity, evidence, family splits, rights gates, and abstention. It must always be described as regression, not as a professional benchmark. Entire leases and their amendments stay in the same family, and those documents are blocked from training.

Phase 3 is complete when any supported model can be selected through one evaluation command, professional benchmarks are pinned to official evaluators, and local regression can be compared against saved machine-readable reports.

Phase 3 is complete. The frozen registry, integrity rules, official-evaluator checkout pins, internal regression suite, and repeatable CLI are in `configs/evaluation/` and `src/leaseguard/evaluation/`. Full benchmark downloads and official scoring run from `notebooks/colab_orchestrator.ipynb` on Colab, not on the local machine.

## Phase 4 — Build the Document Pipeline

The document pipeline turns an approved source document into a clean and traceable intermediate record. It imports or downloads a source, validates the file, parses digital PDF or DOCX text, preserves page or heading boundaries, segments clauses, detects duplicates, redacts personal identifiers in the processed copy, and exports a schema-valid record.

The raw document is never overwritten. Raw, normalized, and labelled forms remain separate. Every record retains its source URL, license, checksum, jurisdiction, page boundaries, and family links to amendments or related agreements.

OCR is deferred. Empty digital text is recorded as `unreadable_source` rather than guessed.

Phase 4 is complete when an approved source document can be processed reproducibly from its manifest entry into a validated intermediate record.

Phase 4 is complete. `SourceManifest` and `ProcessedDocument` schemas, the gated acquire/parse/segment pipeline, duplicate and family checks, and the Colab `process-documents` task are in place. Local tests use tiny synthetic PDF, DOCX, and markdown files. Full source downloads stay on Colab.

## Phase 5 — Build Dataset Version 1

The first dataset should contain separate groups for raw domain text, structured extraction, evidence-grounded conversations, refusals, multi-turn conversations, and preference pairs. Keeping these groups separate makes their quality and purpose easier to measure.

Quality is more useful than raw volume. The project should begin with several hundred carefully checked examples instead of thousands of weak synthetic examples. Synthetic generation may expand wording and conversation styles, but it must not invent legal facts. Answers must be checked against their source evidence and schema.

Dataset splits must happen at the agreement-family level. A lease, its amendment, and a duplicate filing must never be spread across training and evaluation sets.

Phase 5 is complete when Dataset v1 has a dataset card, source and license records, summary statistics, validation results, checksums, and leakage-safe splits.

## Phase 6 — Establish Model Baselines

Candidate models should be tested before fine-tuning. Initial candidates include Qwen3.5-4B, Qwen3.5-9B, and Gemma 3 12B. Each model should receive the same prompts, context, retrieval results, output schema, and evaluation set.

Tests should compare direct prompting, structured prompting, clause-level context, retrieval-assisted context, and quantized inference. This reveals whether a weakness belongs to the model, the prompt, retrieval, parsing, or validation.

Phase 6 is complete when a baseline report selects the strongest practical model and lists the exact weaknesses that fine-tuning needs to improve.

## Phase 7 — Run the First Supervised Fine-Tune

The first model adaptation should use QLoRA supervised fine-tuning through Unsloth. Initial training should focus on structured extraction, grounded answers, evidence citations, and safe refusal behavior.

Every experiment must record the base model revision, dataset version, configuration, random seed, environment, checkpoint, and evaluation result. Only one major variable should change between comparable experiments.

Phase 7 is complete when the adapted model improves the chosen metrics without increasing unsupported claims, missed clauses, or unsafe certainty.

## Phase 8 — Add Retrieval and Deterministic Validation

The fine-tuned model will be combined with retrieval over the lease and a dated jurisdictional knowledge base. Retrieval should use both exact legal terms and semantic similarity, followed by reranking where useful.

Deterministic code should validate dates, monetary values, output structure, page references, and known rule patterns. The model should not be asked to perform work that normal code can perform more reliably.

Phase 8 is complete when important conclusions can be traced to the lease or an official dated source and incorrect citations are rejected.

## Phase 9 — Consider Advanced Training

Preference optimization, continued pretraining, GRPO, long-context adaptation, and distillation should only be introduced to solve a measured weakness. They should not be added because they are fashionable or available.

Each advanced experiment needs a clear hypothesis and an automatic or expert-reviewed evaluation. If it does not improve the target metric without harming safety, it should not become part of the main training process.

Phase 9 is complete when each retained training method has a documented and reproducible benefit.

## Phase 10 — Prepare the First Release

The first public release should include a quantized local model, model card, dataset card, evaluation report, known limitations, privacy guidance, and a reproducible inference example.

Code, dataset, and model versions should be tracked independently. For example, application code may be `v1.0.0` while the dataset and model each have their own version and release notes.

Phase 10 is complete when another person can reproduce the documented evaluation and run LeaseGuard locally without a paid API.

## Milestones

1. `v0.1` — Repository foundation
2. `v0.2` — Ontology and schemas
3. `v0.3` — Evaluation harness
4. `v0.4` — Document pipeline
5. `v0.5` — Curated Dataset v1
6. `v0.6` — Base-model benchmark
7. `v0.7` — First QLoRA model
8. `v0.8` — Retrieval and validation
9. `v0.9` — Safety and adversarial testing
10. `v1.0` — Reproducible local release

The required development order is:

```text
Scope → Schemas → Evaluation → Pipeline → Dataset
      → Baselines → Fine-tuning → Retrieval → Release
```
