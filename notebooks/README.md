# Notebooks

Notebooks are for exploration, reports, and Colab orchestration. Production logic must live in `src/leaseguard` and be tested before the notebook calls it.

`colab_orchestrator.ipynb` is the thin Google Colab T4 entry point for heavy evaluation. It clones this repository, selects a named task, and calls `leaseguard.evaluation.cli`. Full LegalBench, CUAD, ContractNLI, and LegalBench-RAG downloads belong on Colab or Drive, not on the local machine.

Do not commit notebook outputs containing lease text, personal information, or large generated files.
