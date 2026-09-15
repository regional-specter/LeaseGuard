# Notebooks

Notebooks are for exploration, reports, and Colab orchestration. Production logic must live in `src/leaseguard` and be tested before the notebook calls it.

`colab_orchestrator.ipynb` is the thin Google Colab T4 entry point. It mounts Drive, clones this repository, selects a named task, and calls package CLIs. Use `process-documents` for parsing, `build-dataset` for Dataset v1, `regression` for the internal lease suite, `evaluate-*` for official professional benchmarks, `baseline-*` for unmodified base-model product-task runs, and `compare-baselines` to rank those runs.

Phase 6 also has dedicated T4 notebooks. Run them one candidate per session:

- `colab_baseline_qwen35_4b.ipynb`
- `colab_baseline_qwen35_9b.ipynb`
- `colab_baseline_gemma3_12b.ipynb`
- `colab_baseline_compare.ipynb` (CPU is enough after the three GPU runs)

Set Runtime to **T4 GPU** for the candidate notebooks. Gemma requires a Hugging Face token after you accept the Gemma license. Completed predictions resume from Drive checkpoints if Colab disconnects.

The first config cell clones `https://github.com/regional-specter/LeaseGuard.git`. Change `REPO_URL` if you are on a fork. If GitHub is private, copy the repo to Drive as `MyDrive/leaseguard/repo` instead. Install uses `src/` on `sys.path` when Colab's PEP 668 marker blocks `pip install -e .`.

Do not commit notebook outputs containing lease text, personal information, or large generated files.
