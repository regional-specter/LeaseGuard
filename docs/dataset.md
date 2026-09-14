# Dataset

LeaseGuard Dataset v1 is a leakage-safe, schema-validated set of lease-domain examples for supervised fine-tuning. It is not a professional benchmark. Headline evaluation remains LegalBench, CUAD, ContractNLI, and LegalBench-RAG.

The Git copy is a tiny synthetic seed plus frozen rules. Full corpora are written to ignored `data/datasets/` or Colab Drive and must not be committed.

The Dataset v1 evaluation split is frozen in `configs/evaluation/eval_split.v1.json`. The seed office family hashes into `test`; the seed retail family hashes into `train`. Unmodified base-model product-task scoring uses only the test split.

## Version

Dataset version `1.0.0` was frozen on 2026-09-08 against ontology version `1.0.0`. Construction rules live in `configs/data/dataset.v1.json`.

## Inclusion

A record may enter Dataset v1 only when all of the following hold:

- The document is an English US office lease, retail lease, amendment, exhibit, or related agreement.
- Reuse rights are `public_domain`, `open_license`, or `permission_granted`, with a named license for open-license sources.
- Every answered claim quotes evidence that occurs in registered source text.
- Synthetic wording does not invent legal facts that are absent from that source text.
- The record is not an exact duplicate of an earlier example.

## Exclusion

These items never enter training or validation:

- Every document ID listed in the internal regression holdout, including the Phase 2 GSA, PandaDoc, St. Joseph, and sample-office records
- Rights marked `review_required` or `restricted`
- Official professional-benchmark test splits
- User-uploaded documents
- Near-duplicates that would place similar text from two families into different splits

## Subsets

| Subset | Purpose |
| --- | --- |
| `raw_domain_text` | Lease language exposure without extraction labels |
| `structured_extraction` | Ontology-valid `LeaseExtraction` records |
| `evidence_conversation` | Single-turn grounded questions and answers |
| `refusal_uncertainty` | Abstention, missing evidence, and out-of-scope state-law questions |
| `multi_turn` | Follow-up conversation on one agreement family |
| `preference_pairs` | Clearly ranked responses; the preferred side cannot invent facts |

## Splits

Splits are assigned by hashing `family_id` with a frozen salt. A lease, its amendment, and related filings receive one split. The seed uses 70% train, 15% validation, and 15% test buckets. Dataset v1's test split is an internal holdout, not a lab benchmark.

## Quality checks

The builder verifies schemas, evidence substrings, optional character offsets, clause text, unique record IDs, family integrity, holdout alignment, checksums, and a deterministic manual-review sample. The committed seed sample is recorded in `data/samples/dataset_v1/quality_review.json`.

## Licenses

Seed examples are synthetic and released as CC0-1.0. Later full-corpus sources must keep their own license and rights status on every record. LEDGAR remains outside the distributable set because of its non-commercial license.

## Limitations

- The Git seed is small by design. It proves the rules; it does not replace a larger curated corpus assembled in Colab.
- Synthetic examples must not be cited as real leases or as law.
- Official benchmark data remains evaluation-only.

## Commands

```bash
uv run python scripts/prepare/build_dataset.py validate-config
uv run python scripts/prepare/build_dataset.py validate-seed
uv run python scripts/prepare/build_dataset.py build --output /tmp/leaseguard-dataset.v1.json
uv run python scripts/prepare/build_dataset.py stats
```

Full-size builds run from `notebooks/colab_orchestrator.ipynb` with `TASK=build-dataset`.
