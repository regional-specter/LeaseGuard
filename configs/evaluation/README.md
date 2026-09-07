# Evaluation Configuration

`benchmarks.v1.json` is the frozen registry of approved professional external benchmarks. It pins dataset repositories, official evaluator repositories, git revisions, test splits, metrics, licenses, venues, and benchmark roles.

The current headline suite is:

- LegalBench (NeurIPS 2023 Datasets and Benchmarks)
- CUAD (NeurIPS 2021)
- ContractNLI (Findings of EMNLP 2021)
- LegalBench-RAG (published legal retrieval benchmark)

Development code must select published runs from this registry. A local or newly invented test cannot be silently substituted for a published benchmark.

`regression.v1.json` is the internal office and retail product suite. It is frozen for engineering safety and must be described as regression, not as a professional benchmark.
