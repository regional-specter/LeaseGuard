# Lease Ontology

The LeaseGuard ontology defines the facts and relationships that the first release can extract from US office and retail leases. It gives dataset creators, validators, evaluators, and models one shared vocabulary.

Ontology version `1.0.0` belongs to the first product scope. Future changes must follow semantic versioning. A breaking label or field change requires a new major version.

## Design Rules

The ontology separates document facts from interpretation. A party name, rent amount, or notice period can be extracted as a fact. A claim that a clause is enforceable or fair is a legal interpretation and is outside the first release.

Every important fact needs at least one evidence span. The evidence must contain exact text from the source document and identify its page. Section names and character offsets should be included when the document parser can provide them.

The normalized value must never replace the original wording. For example, a date may be normalized for software use, but the source expression must also remain available through evidence.

## Document Types

The first ontology supports:

- Office leases
- Retail leases
- Lease amendments
- Lease exhibits and schedules
- Other related lease documents when their relationship is known

Documents in the same lease family need a shared family identifier so they cannot leak across dataset splits.

## Parties

The first release recognizes:

- Landlord
- Tenant
- Guarantor
- Property manager
- Other named party

A party record contains its name, role, and supporting evidence. If a party's role is unclear, it should use the `other` role rather than guessing.

## Clause Types

The first ontology includes the following clause groups:

### Identity and Property

- Parties
- Premises
- Property use
- Definitions

### Term and Options

- Lease term
- Commencement
- Expiration
- Renewal option
- Extension option
- Early termination

### Money

- Base rent
- Rent increase
- Security deposit
- Operating expenses
- Common area maintenance
- Taxes
- Insurance
- Late fees
- Percentage rent

### Operations

- Maintenance
- Repairs
- Utilities
- Alterations
- Access
- Signage
- Parking

### Transfers and Changes

- Assignment
- Subletting
- Change of control
- Amendment
- Notice

### Enforcement and Protection

- Default
- Cure period
- Remedies
- Indemnity
- Liability
- Damage and destruction
- Condemnation
- Force majeure
- Dispute resolution
- Governing law

### Supporting Material

- Guarantee
- Exhibit
- Schedule
- Other

Labels describe what a clause discusses. They do not decide whether the clause is valid, balanced, or enforceable.

## Obligations

An obligation records:

- The party that must act
- The required action
- The party receiving the action, when stated
- The source wording
- A fixed date, relative deadline, or triggering event
- Recurrence, when stated
- A monetary amount, when stated
- Whether the obligation depends on a condition
- Supporting evidence

The system should keep relative expressions such as “within ten business days after notice” instead of inventing a calendar date when the triggering date is unknown.

## Evidence

An evidence span contains:

- Exact quoted text
- One-based page number
- Section or heading when available
- Start and end character offsets when available

Offsets use a zero-based, end-exclusive convention. The end offset must be greater than the start offset. Evidence from different pages should use separate spans.

## Answers

An answer contains a simple-English explanation, selected perspective, status, confidence, evidence, and optional warnings.

The allowed answer statuses are:

- `answered`: the document supports the answer
- `insufficient_evidence`: the document does not provide enough information
- `conflicting_evidence`: relevant provisions disagree
- `unreadable_source`: the source could not be read reliably

An `answered` response requires evidence. Other statuses may include evidence to explain the problem, but the system must not invent a normal answer.

## Confidence

Confidence is a value from `0.0` to `1.0` describing confidence in document extraction, not confidence about legal enforceability. It must not be presented as a probability that legal advice is correct.

## Stakeholder Perspective

The supported perspectives are:

- Landlord
- Tenant
- Neutral

Changing perspective may change emphasis and operational suggestions. It must not change names, amounts, dates, quoted evidence, or other extracted facts.

## Explicitly Deferred Concepts

The first ontology does not define:

- State-law compliance
- Enforceability
- Legal advice
- Automatic risk scores
- Market-standard comparisons
- Redline recommendations
- Negotiation positions

These concepts require additional legal sources, expert review, and separate evaluation before they can be added safely.
