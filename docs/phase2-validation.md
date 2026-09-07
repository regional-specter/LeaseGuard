# Phase 2 Ontology Validation

This review checks ontology version `1.0.0` against real lease language before the project begins evaluation and dataset construction.

## Compute and Storage Boundary

The local repository is limited to source code, manifests, schemas, tiny annotations, and tests. Large document collections, parsing jobs, dataset generation, model evaluation, and training will run in one Google Colab notebook on a T4 GPU.

Raw local validation files are small and ignored by Git. Full corpora, model weights, CUDA libraries, Unsloth, and checkpoints must not be installed or stored in the local project.

## Validation Sources

### GSA Global Lease Template L100

The May 2026 GSA template provides a detailed office lease structure. It contains premises, rentable and usable area, rent components, operating cost adjustments, renewal and termination rights, tenant improvements, building services, maintenance, accessibility, environmental requirements, and event-based duties.

The document is an uncompleted template. Placeholder values must remain placeholders and must not be converted into factual amounts, names, or dates.

### City of St. Joseph Concession Lease

The municipal lease provides retail-like operational language for a concession and event facility. It includes rent, automatic renewal, landlord and tenant termination rights, public and private use, assignment, maintenance, insurance, default, environmental restrictions, security, notices, and governing law.

The document is useful for ontology validation, but it is not a normal private shopping-center lease. Its municipal reuse rights also require review before it can enter a released training dataset.

### PandaDoc Commercial Lease Template

The MIT-licensed template is explicitly intended for office, retail, or industrial space. It provides a clearly reusable retail-capable source and tests section-based evidence for a document without stable page numbers.

The template is intentionally short. It does not replace the need for detailed shopping-center leases containing percentage rent, co-tenancy, exclusivity, continuous operation, radius restrictions, and sales reporting.

### GSA Lease Amendment Template

The amendment template changes rent, janitorial, cleaning, and brokerage provisions. It confirms that an amendment cannot be represented as an independent list of clauses alone. LeaseGuard must record the earlier document, target clause, action, effective date, and evidence for each change.

## Problems Found in the Initial Ontology

### Evidence needed a document identity

The first evidence model identified a page and section but not the document. That fails when an answer combines a lease and one or more amendments.

Each evidence span now requires `document_id`.

### Page numbers cannot always be required

PDFs usually have stable physical pages, but DOCX and SEC HTML documents may not. Evidence now accepts a physical page, printed page label, section, or normalized-text character offsets. At least one stable locator is required.

Character offsets are zero-based and end-exclusive against the normalized full-document text. Exact offset verification will be implemented with the Phase 4 parser.

### Premises needed structured fields

Real leases describe property names, addresses, suites, floors, rentable area, usable area, and permitted use. The `Premises` model now preserves these concepts separately while retaining exact evidence.

### Clauses need multiple labels

A single paragraph may cover base rent, operating expenses, tenant improvements, parking, taxes, and insurance. Forcing one label would discard useful meaning.

Clauses now accept one or more labels. The text remains exact source text, while an optional summary may explain it in simple English.

### Retail and operational labels were missing

The initial taxonomy lacked several concepts found in office and retail operations. Version `1.0.0` now includes additional rent, rent abatement, tenant share, common areas, tenant improvements, delivery condition, continuous operation, co-tenancy, exclusivity, radius restrictions, prohibited use, relocation, holdover, surrender, subordination, non-disturbance, estoppel, liens, brokerage, accessibility, environmental duties, services, janitorial work, security, and compliance with laws.

### Amendments needed explicit effects

The initial model could classify an amendment but could not say what it changed. `AmendmentEffect` now records add, delete, replace, modify, confirm, or other actions against a target document and clause.

### Source rights needed to be enforceable metadata

Public access does not always permit unrestricted training and redistribution. Every source now carries a rights status, license information where available, and review notes.

Open-license sources must name their license. Sources marked for review must explain why review is needed.

### Relative deadlines needed optional normalization

Operational tracking needs to distinguish “10 days” from “10 business days.” Obligations now preserve the original deadline language and may also store a normalized value and unit.

The system must not calculate a due date until the triggering event has a known date.

### Rent tables need context

Lease rent often changes by period and may be expressed monthly, annually, or per square foot. Monetary terms now include optional period and rate-basis fields. Complete table reconstruction remains a Phase 4 parsing responsibility.

## Labeling Rules Confirmed

- Preserve exact clause text separately from summaries.
- Assign every applicable clause label when a provision covers several concepts.
- Do not infer blank template values.
- Keep relative dates when a trigger date is unknown.
- Link every evidence span to its source document.
- Treat amendment language as a change to an earlier lease.
- Keep extraction confidence separate from legal enforceability.
- Keep sources awaiting rights review out of released training data.
- Do not turn federal, state, or local law references into legal conclusions.

## Remaining Phase 2 Boundary

Ontology version `1.0.0` is frozen for evaluation work. Generated schemas and all representative validation records pass automated checks.

The reusable PandaDoc template satisfies Phase 2's small retail-template requirement. A larger, detailed, clearly reusable private retail lease corpus is still required during Phase 5. The St. Joseph document remains local-validation-only and must not be silently promoted into release training data.
