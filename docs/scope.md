# Product Scope

This document defines the approved scope for the first usable LeaseGuard release. These decisions guide the ontology, dataset, evaluation set, and model training.

## Purpose

LeaseGuard will help property managers and real-estate operations teams read and manage US commercial leases. The first release will focus on office and retail leases.

LeaseGuard will explain what the supplied documents say. It will not make state-law conclusions in the first release. This boundary allows the project to support leases from across the United States without pretending that one general contract dataset represents every state's law.

## Initial Users

The primary users are property managers and real-estate operations teams. The system should use simple language while preserving the exact lease wording needed for review and audit.

Users may select a landlord, tenant, or neutral perspective for each request. Perspective changes which operational concerns are highlighted, but it must never change the underlying extracted facts.

## Initial Capabilities

Version 1 will support three main tasks:

1. Extract lease metadata and clauses into validated structured data.
2. Answer questions using exact evidence from the supplied lease.
3. Track obligations such as deadlines, payments, notices, and responsible parties.

Risk scoring, automated redlining, negotiation drafting, and state-law guidance are outside the first release. They may be considered after the first tasks are measured and reliable.

## Supported Documents

The first release will accept English-language digital PDF and DOCX files. Pasted text, scanned documents, photographs, handwriting, and multilingual documents are not required initially.

The document pipeline should still preserve an extension point for OCR. OCR support can be added later without changing the core lease schemas.

## Outputs

Each response should include:

- A short explanation in simple English.
- Valid structured JSON for software integrations.
- Exact supporting text from the supplied document.
- Page and section references when available.
- The selected landlord, tenant, or neutral perspective.
- Clear uncertainty and missing-information warnings.

Facts, interpretations, and operational suggestions must remain distinguishable in the output.

## Jurisdiction Boundary

The first release covers document-grounded analysis of US commercial leases. It may identify a governing-law clause, but it must not decide whether a provision is legal, enforceable, customary, or compliant with a particular state's law.

CUAD and ContractNLI may support general contract extraction and evidence-grounded reasoning. They are not complete lease-law sources and must not be presented as such.

## Privacy

Lease processing will be local by default. User documents, corrections, and outputs must not be reused for training.

The first release will not require cloud document storage. Logs must avoid storing full lease text or personal information unless the user explicitly enables a future secure storage feature.

## Safety Boundary

LeaseGuard is a decision-support system, not a lawyer or final decision-maker. It should:

- Show the evidence behind important answers.
- State when the document does not contain the answer.
- Report conflicting or unreadable provisions.
- Avoid unsupported legal conclusions.
- Recommend professional review when a request requires legal judgment.

## Success Criteria

The first release is successful when it can process representative office and retail leases, produce schema-valid outputs, answer with accurate evidence, identify operational obligations, and abstain when the source does not support an answer.
