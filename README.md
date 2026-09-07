<div align="center">

<img width="1799" height="504" alt="LeaseGuard" src="https://github.com/user-attachments/assets/a6ced70e-e695-484d-8809-7580141a536a" />

# LeaseGuard

LeaseGuard is an open-source, domain-adapted language model for reading, explaining, and reviewing real-estate lease agreements. It aims to provide private, evidence-based lease analysis without paid APIs.

</div>

## Overview

LeaseGuard will combine a small fine-tuned model with document retrieval, fixed validation rules, and clear source citations. Fine-tuning will teach the model how to handle lease tasks and conversations. Current laws and jurisdiction-specific guidance will come from a versioned knowledge base instead of model memory.

| Area | Current direction |
| --- | --- |
| Main use | Nationwide US office and retail lease analysis |
| Model size | About 9B–12B parameters |
| Candidate models | Qwen3.5-9B, Qwen3.5-4B, and Gemma 3 12B |
| Target users | Property managers and real-estate operations teams |
| Training | QLoRA supervised fine-tuning with Unsloth |
| Deployment | Local or on-premise quantized inference |
| Cost goal | Free and open resources with no paid APIs |

## Key Functions

| Function | Purpose | Scope |
| --- | --- | --- |
| Document intake | Read English digital PDF and DOCX leases | Version 1 |
| Lease extraction | Return parties, dates, rent, obligations, options, and clauses as structured data | Version 1 |
| Evidence-based Q&A | Answer questions with exact clause and page references | Version 1 |
| Obligation tracking | Find deadlines, notice periods, payments, and responsible parties | Version 1 |
| Uncertainty handling | State when information is missing, unclear, or needs professional review | Version 1 |
| Lease comparison | Compare drafts, amendments, and related agreements | Later |
| Risk review | Explain possible issues from the landlord or tenant viewpoint | Later |
| Clause redlining | Suggest clearer or safer wording for negotiation | Later |

## System Components

| Component | Responsibility |
| --- | --- |
| Ingestion | Validate files and keep page, section, table, and exhibit structure |
| OCR and layout | Preserve an extension point for future scanned-document support |
| Privacy | Detect and remove personal or confidential information |
| Clause parser | Split agreements into sections and known clause types |
| Retrieval | Find relevant lease text, laws, and official guidance |
| Fine-tuned model | Extract, explain, compare, and discuss lease content |
| Rule engine | Check dates, money, citations, and known risk patterns |
| Output validator | Enforce the required JSON structure and evidence fields |
| Audit log | Record sources, model version, evidence, and confidence |

## Lease Knowledge Areas

| Group | Topics |
| --- | --- |
| Core details | Parties, premises, term, commencement, expiry, and governing law |
| Money | Base rent, increases, deposits, late fees, taxes, insurance, and operating costs |
| Property use | Permitted use, exclusivity, alterations, maintenance, and repairs |
| Changes | Assignment, subletting, renewal, break options, amendments, and notices |
| Legal duties | Liability, indemnity, compliance, environmental duties, and accessibility |
| Enforcement | Default, cure periods, remedies, guarantees, termination, and disputes |

Every extracted fact should keep its exact supporting text and page or clause reference.

## Data Sources

Machine-learning datasets, signed leases, statutes, and court decisions serve different purposes. A dataset can teach a model how to find evidence, but it is not legal authority.

| Source | Contents | LeaseGuard use | License or restriction |
| --- | --- | --- | --- |
| [CUAD](https://www.atticusprojectai.org/cuad/) | 510 commercial contracts, 13,101 labels, and 41 clause types | Clause and evidence extraction | CC BY 4.0 |
| [ContractNLI](https://stanfordnlp.github.io/contract-nli/) | 607 NDAs tested against 17 legal statements | Entailed, contradicted, and not-mentioned reasoning | CC BY 4.0 |
| [ACORD](https://www.atticusprojectai.org/acord/) | 126,000+ expert-rated query and clause pairs | Clause retrieval and reranker training | CC BY 4.0 |
| [SEC EDGAR](https://www.sec.gov/search-filings) | Filed contracts, leases, amendments, guarantees, and exhibits | Build the lease-specific corpus | Public access; review each document's reuse rights |
| [LegalBench-RAG](https://github.com/zeroentropy-ai/legalbenchrag) | 6,858 legal retrieval questions | Retrieval evaluation only | MIT code; source dataset licenses still apply |
| [ContractEval](https://aclanthology.org/2025.nllp-1.19/) | CUAD-based model evaluation | Clause-review benchmark only | Verify repository and source terms |
| [MAUD](https://www.atticusprojectai.org/maud/) | 152 merger agreements, 47,000+ labels, and 92 questions | Optional contract reasoning experiments | CC BY 4.0 |
| LEDGAR | About 846,000 provisions from more than 60,000 contracts | Isolated research only | Official release is CC BY-NC 4.0 |

### What Each Dataset Can and Cannot Teach

**CUAD** is our main starting point for general contract extraction. It teaches a model to find clauses such as parties, dates, renewal terms, governing law, assignment, liability, insurance, and termination. It is not mainly a real-estate lease dataset and does not explain whether a clause is legally enforceable.

**ContractNLI** is useful because it teaches the difference between supported, contradicted, and missing information. Its evidence annotations can help LeaseGuard avoid unsupported answers. Every document is an NDA, so we will reuse its reasoning format rather than treat it as lease knowledge.

**ACORD** is designed for semantic legal search. It can help the retrieval system find the correct clause before the conversational model answers. It is general contract data rather than lease-specific law.

**MAUD** contains strong expert annotations, but it focuses on merger agreements. It may help with general legal reading experiments, but it should not be a major LeaseGuard training source.

**LEDGAR** has valuable scale, but its official non-commercial license is too restrictive for an unrestricted release model. It will remain outside the distributable training dataset unless a later license review permits its use.

**LegalBench-RAG** and **ContractEval** are benchmarks. They should measure retrieval and model quality without becoming sources of legal truth.

None of these datasets is enough by itself. LeaseGuard still needs a dedicated collection of office and retail leases, amendments, exhibits, subleases, guarantees, commencement certificates, and related documents.

SEC EDGAR is the best free starting point for that collection. The collector must follow the SEC fair-access rules, identify itself, and remain below the current limit of ten requests per second. Public access does not automatically make every company-filed exhibit copyright-free, so every document needs a provenance and reuse review before redistribution or training.

Each collected record must retain its source, license status, download date, jurisdiction, document type, checksum, and related-document family. Random internet scraping will not be a main source because it may include copyrighted, confidential, duplicated, or low-quality material.

## How US Commercial Lease Law Works

The United States does not have one nationwide commercial lease code. It has one federal Constitution, fifty state legal systems, federal statutes and regulations, and local city or county rules.

### The Lease Itself

The signed lease and its amendments are the first source for operational questions. They explain who must pay, when notice is due, who performs repairs, whether assignment is allowed, and what happens after default.

Commercial parties generally have broad freedom to negotiate. Applicable state, local, or federal law can still override some lease terms.

### State Law

Most commercial lease law comes from the state connected to the property. State statutes and court decisions may control:

- Contract formation and interpretation
- Statute of Frauds and writing requirements
- Lease recording
- Quiet enjoyment and constructive eviction
- Assignment and subletting
- Default, cure, eviction, and self-help procedures
- Mitigation after tenant default
- Indemnity, liability limits, late fees, and available remedies
- Casualty, condemnation, notices, governing law, and forum selection

These rules can differ substantially between states. A governing-law clause matters, but mandatory law where the property is located may still apply. Version 1 may detect the named state, but it will not make state-law conclusions.

### Local Law

Cities and counties may control zoning, permitted use, building and fire codes, occupancy certificates, signage, accessibility, operating permits, and local taxes. Local rules can be especially important for retail properties.

### Federal Law

Federal law does not replace state commercial lease law, but several federal rules can affect a lease.

| Federal area | Why it matters |
| --- | --- |
| [Bankruptcy, 11 U.S.C. §365](https://uscode.house.gov/view.xhtml?req=%28title%3A11+section%3A365+edition%3Aprelim%29) | Controls assumption, assignment, and rejection of unexpired leases |
| [Bankruptcy claims, 11 U.S.C. §502](https://uscode.house.gov/view.xhtml?edition=prelim&num=0&req=granuleid%3AUSC-prelim-title11-section502) | Limits certain landlord claims caused by lease termination |
| [ADA Title III](https://www.ada.gov/law-and-regs/regulations/title-iii-regulations/) | Applies accessibility rules to covered public accommodations and commercial facilities |
| CERCLA, 42 U.S.C. §9601 onward | Can affect owners, operators, tenants, contamination, and environmental indemnities |
| [E-SIGN, 15 U.S.C. §7001](https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title15-section7001&num=0&edition=prelim) | Prevents denial of a contract solely because an electronic record or signature was used |
| [42 U.S.C. §1981](https://uscode.house.gov/view.xhtml?edition=prelim&num=0&req=granuleid%3AUSC-prelim-title42-section1981) | Protects equal rights to make and enforce contracts |
| [42 U.S.C. §1982](https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title42-section1982&num=0&edition=prelim) | Protects equal rights to lease, purchase, hold, sell, and convey property |

An ADA clause may allocate accessibility work and cost between landlord and tenant. For covered properties, that private allocation does not necessarily remove either party's external statutory responsibility.

An environmental indemnity may allocate cleanup costs between the parties. Under CERCLA, it does not necessarily remove the underlying responsibility owed to the government.

### The US Constitution

The Constitution provides high-level protection against government action rather than ordinary lease-management rules.

- The **Contract Clause** limits certain state laws that impair existing contracts. It does not prevent all contract regulation.
- The **Fifth Amendment Takings Clause** can protect leasehold interests when government takes property for public use.
- The **Fifth and Fourteenth Amendment Due Process Clauses** can protect property interests against certain government action.

These provisions do not answer ordinary questions such as when rent is due or whether a tenant may sublease.

### Sources That Are Not Nationwide Commercial Lease Law

- The Uniform Residential Landlord and Tenant Act concerns residential tenancies, not our commercial scope.
- The Uniform Commercial Code mainly governs commercial transactions and leases of goods, not normal real-property leases.
- Restatements, legal textbooks, and practitioner articles can explain law but are not binding authority.
- CUAD, ContractNLI, ACORD, and other machine-learning datasets are not law.

### Legal Source Priority

LeaseGuard should use sources in this order:

1. The supplied lease, amendments, and exhibits
2. Official federal statutes and regulations
3. Official state statutes and regulations
4. Controlling federal and state court decisions
5. Official county and city codes
6. Official agency guidance
7. Model laws and reliable secondary explanations
8. Practitioner articles only for discovering stronger sources

Every legal record should include its jurisdiction, authority type, citation, effective date, official URL, retrieval date, amendment status, and checksum. Legal rules should live in a dated retrieval system rather than being memorized through fine-tuning.

## Dataset Types

| Dataset | What it teaches |
| --- | --- |
| Domain text | Lease vocabulary and drafting patterns |
| Structured extraction | Fields, clause types, dates, values, parties, and evidence spans |
| Grounded conversations | Clear answers supported by the supplied agreement |
| Risk reviews | Issue, severity, affected party, evidence, and negotiation point |
| Refusal examples | How to handle missing, unclear, or unreliable information |
| Multi-turn conversations | Follow-up questions, corrections, and document references |
| Preference pairs | Preference for accurate, complete, cited, and careful answers |

Training, validation, and test data will be split by complete agreement and document family. A lease and its amendments must not appear in different splits.

## Training Plan

| Stage | Work |
| --- | --- |
| 1. Scope | Choose the first jurisdiction, lease type, users, and safety limits |
| 2. Ontology | Define clause labels and structured output fields |
| 3. Gold evaluation | Pin professional benchmarks and keep a private regression set |
| 4. Baselines | Compare the unmodified 4B, 9B, and 12B candidate models |
| 5. Data pipeline | Collect, clean, deduplicate, redact, and label documents |
| 6. Supervised training | Run QLoRA fine-tuning with high-quality examples |
| 7. Retrieval | Add cited laws, guidance, and document search |
| 8. Validation | Add fixed checks for dates, money, and citations |
| 9. Advanced training | Add preference training or GRPO only when evaluation supports it |
| 10. Deployment | Quantize and run the selected model locally |

Long leases will use clause parsing and retrieval instead of placing every page into every training example. Continued pretraining and reinforcement learning will only be added if measured results justify their cost.

## Evaluation

Headline research claims must come from published professional benchmarks scored with their official evaluators:

| Benchmark | What it measures | Why it counts |
| --- | --- | --- |
| LegalBench | Legal reasoning across expert-authored tasks | NeurIPS 2023 Datasets and Benchmarks; widely used by labs |
| CUAD | Contract clause extraction | NeurIPS 2021; lawyer-supervised commercial-contract labels |
| ContractNLI | Evidence-grounded entailment, contradiction, and not-mentioned | Findings of EMNLP 2021 |
| LegalBench-RAG | Character-level legal retrieval | Published retrieval protocol used for legal RAG |

The internal office and retail set is a regression suite for LeaseGuard's JSON schema, evidence, family splits, and abstention. It is not a substitute for those benchmarks and cannot support a breakthrough claim.

There is no blended “LeaseGuard score.” Numeric thresholds are frozen only after unmodified base models are run on the official protocols.

## Free Tooling

| Need | Free options |
| --- | --- |
| Models and datasets | Hugging Face |
| Data processing | Python and open-source document tools |
| Fine-tuning | Unsloth, Transformers, and TRL |
| Heavy compute | One Google Colab notebook using a T4 GPU |
| Local inference | llama.cpp, Ollama, MLX, or vLLM |
| Apple Silicon | MLX or llama.cpp for local inference |

The local repository holds code, schemas, manifests, tiny samples, and tests. Bulk parsing, dataset generation, evaluation, fine-tuning, and model export run in Colab. Full datasets, CUDA libraries, Unsloth, model weights, and checkpoints must stay off the local machine.

Colab sessions have time and storage limits, so every heavy task must support checkpoints, safe restarts, pinned configurations, and remote output storage.

## Important Principle

LeaseGuard is a review and decision-support system, not a replacement for a qualified legal professional. It should show evidence, report uncertainty, distinguish landlord and tenant interests, and avoid conclusions that the source material cannot support.

## Development

Track current progress in [TODO.md](TODO.md) and read the detailed roadmap in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Project scope, architecture, dataset, evaluation, and safety decisions are documented separately under `docs/`.

LeaseGuard supports Python 3.11 and 3.12. Install the development environment and run the Phase 1 checks with:

```bash
uv sync --dev
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes.

## License

This project uses the MIT License.
