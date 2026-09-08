# Notebooks

Notebooks are for exploration, reports, and Colab orchestration. Production logic must live in `src/leaseguard` and be tested before the notebook calls it.

`colab_orchestrator.ipynb` is the thin Google Colab T4 entry point. It mounts Drive, clones this repository, selects a named task, and calls package CLIs. Use `process-documents` for parsing, `build-dataset` for Dataset v1, `regression` for the internal lease suite, and `evaluate-*` for official professional benchmarks.

Do not commit notebook outputs containing lease text, personal information, or large generated files.
