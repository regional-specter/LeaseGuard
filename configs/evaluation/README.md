# Evaluation Configuration

`benchmarks.v1.json` is the frozen registry of approved professional external benchmarks. It pins dataset repositories, official evaluator repositories, git revisions, test splits, metrics, licenses, venues, and benchmark roles.

The current headline suite is:

- LegalBench (NeurIPS 2023 Datasets and Benchmarks)
- CUAD (NeurIPS 2021)
- ContractNLI (Findings of EMNLP 2021)
- LegalBench-RAG (published legal retrieval benchmark)

Development code must select published runs from this registry. A local or newly invented test cannot be silently substituted for a published benchmark.

`regression.v1.json` is the internal office and retail product suite. It is frozen for engineering safety and must be described as regression, not as a professional benchmark.

`eval_split.v1.json` freezes the Dataset v1 evaluation split used for unmodified base-model product-task runs. It is a holdout, not a professional benchmark.

`baselines.v1.json` pins one configuration per candidate (Qwen3.5-4B, Qwen3.5-9B, Gemma 3 12B) and the T4 comparison matrix: direct vs schema-constrained prompting, full vs clause vs retrieval context, and 4-bit vs 8-bit inference.
