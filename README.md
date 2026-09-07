<div align="center">

<img width="1799" height="504" alt="LeaseGuard" src="https://github.com/user-attachments/assets/a6ced70e-e695-484d-8809-7580141a536a" />

# LeaseGuard

LeaseGuard is an open-source, domain-adapted language model for reading, explaining, and reviewing real-estate lease agreements. It aims to provide private, evidence-based lease analysis without paid APIs.

</div>

## Overview

LeaseGuard will combine a small fine-tuned model with document retrieval, fixed validation rules, and clear source citations. Fine-tuning will teach the model how to handle lease tasks and conversations. Current laws and jurisdiction-specific guidance will come from a versioned knowledge base instead of model memory.

| Area | Current direction |
| --- | --- |
| Main use | Commercial real-estate lease review |
| Model size | About 9B–12B parameters |
| Candidate models | Qwen3.5-9B, Qwen3.5-4B, and Gemma 3 12B |
| Target users | Property managers, real-estate teams, and law firms |
| Training | QLoRA supervised fine-tuning with Unsloth |
| Deployment | Local or on-premise quantized inference |
| Cost goal | Free and open resources with no paid APIs |

## Key Functions

| Function | Purpose |
| --- | --- |
| Document intake | Read PDF, DOCX, HTML, and scanned leases |
| Lease extraction | Return parties, dates, rent, obligations, options, and clauses as structured data |
| Evidence-based Q&A | Answer questions with exact clause and page references |
| Risk review | Explain possible issues from the landlord or tenant viewpoint |
| Lease comparison | Compare drafts, amendments, and related agreements |
| Obligation tracking | Find deadlines, notice periods, payments, and responsible parties |
| Clause redlining | Suggest clearer or safer wording for negotiation |
| Uncertainty handling | State when information is missing, unclear, or needs professional review |

## System Components

| Component | Responsibility |
| --- | --- |
| Ingestion | Validate files and keep page, section, table, and exhibit structure |
| OCR and layout | Read scans without losing the document layout |
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

| Source | Planned use |
| --- | --- |
| SEC EDGAR | Public lease exhibits, amendments, and related filings |
| Government sources | Lease templates, laws, regulations, and official guidance |
| CUAD | Expert-labelled contract clause extraction |
| ContractNLI | Evidence-based contract reasoning |
| LEDGAR | Contract clause classification |
| MAUD | Extra contract reasoning examples where relevant |

Each document must record its source, license, date, jurisdiction, and document type. Random internet scraping will not be a main source because it may include copyrighted, confidential, duplicated, or low-quality material.

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
| 3. Gold evaluation | Prepare a private expert-reviewed test set |
| 4. Baselines | Compare the unmodified 4B, 9B, and 12B candidate models |
| 5. Data pipeline | Collect, clean, deduplicate, redact, and label documents |
| 6. Supervised training | Run QLoRA fine-tuning with high-quality examples |
| 7. Retrieval | Add cited laws, guidance, and document search |
| 8. Validation | Add fixed checks for dates, money, and citations |
| 9. Advanced training | Add preference training or GRPO only when evaluation supports it |
| 10. Deployment | Quantize and run the selected model locally |

Long leases will use clause parsing and retrieval instead of placing every page into every training example. Continued pretraining and reinforcement learning will only be added if measured results justify their cost.

## Evaluation

| Area | Measures |
| --- | --- |
| Extraction | Precision, recall, F1, and evidence-span overlap |
| Structured output | Valid JSON and correct field types |
| Legal grounding | Citation accuracy and unsupported-claim rate |
| Lease reasoning | Obligation, contradiction, amendment, date, and money accuracy |
| Safety | Missed risks, uncertainty calibration, and correct abstention |
| Fairness | Consistent analysis from landlord and tenant viewpoints |
| Retrieval | Relevant-source recall and ranking quality |

CUAD, ContractNLI, LegalBench-RAG, and ContractEval can provide starting benchmarks. LeaseGuard will also need a private lease-specific evaluation set that is never used for training or synthetic data generation.

## Free Tooling

| Need | Free options |
| --- | --- |
| Models and datasets | Hugging Face |
| Data processing | Python and open-source document tools |
| Fine-tuning | Unsloth, Transformers, and TRL |
| Free GPU access | Kaggle or Google Colab, subject to their limits |
| Local inference | llama.cpp, Ollama, MLX, or vLLM |
| Apple Silicon | MLX or llama.cpp for local inference |

Unsloth training normally needs an NVIDIA CUDA GPU. Free GPU services have session, storage, and availability limits, so training must support checkpoints and safe restarts.

## Important Principle

LeaseGuard is a review and decision-support system, not a replacement for a qualified legal professional. It should show evidence, report uncertainty, distinguish landlord and tenant interests, and avoid conclusions that the source material cannot support.

## License

This project uses the MIT License.
