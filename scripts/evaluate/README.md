# Evaluation Scripts

Use one command for professional-benchmark bookkeeping, internal regression, and unmodified base-model baselines:

```bash
uv run python scripts/evaluate/run_benchmark.py list
uv run python scripts/evaluate/run_benchmark.py validate-registry
uv run python scripts/evaluate/run_benchmark.py regression
uv run python scripts/evaluate/run_benchmark.py score-ranked tests/fixtures/evaluation/ranked_predictions.json
uv run python scripts/evaluate/run_benchmark.py check-integrity --training-ids-file path/to/ids.txt
uv run python scripts/evaluate/run_benchmark.py validate-baselines
uv run python scripts/evaluate/run_benchmark.py show-eval-split
uv run python scripts/evaluate/run_benchmark.py list-candidates
```

`list` and `validate-registry` operate on the frozen professional suite: LegalBench, CUAD, ContractNLI, and LegalBench-RAG.

`regression` scores LeaseGuard's small internal lease records. It is a product check, not a lab benchmark, and cannot support a breakthrough claim.

`validate-baselines` and `show-eval-split` inspect the Phase 6 unmodified-model matrix. Full GPU runs use the Colab T4 notebooks. A local `baseline --backend scripted` dry-run is not a publishable result.

Official evaluator clones, publisher-weight downloads, and full test-set scoring run only in Colab after `LEASEGUARD_ALLOW_BENCHMARK_DOWNLOAD=1` or `LEASEGUARD_ALLOW_MODEL_DOWNLOAD=1` is set. The local machine must not download those corpora or checkpoints.
