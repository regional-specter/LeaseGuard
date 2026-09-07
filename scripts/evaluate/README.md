# Evaluation Scripts

Use one command for both professional-benchmark bookkeeping and the internal lease regression suite:

```bash
uv run python scripts/evaluate/run_benchmark.py list
uv run python scripts/evaluate/run_benchmark.py validate-registry
uv run python scripts/evaluate/run_benchmark.py regression
uv run python scripts/evaluate/run_benchmark.py score-ranked tests/fixtures/evaluation/ranked_predictions.json
uv run python scripts/evaluate/run_benchmark.py check-integrity --training-ids-file path/to/ids.txt
```

`list` and `validate-registry` operate on the frozen professional suite: LegalBench, CUAD, ContractNLI, and LegalBench-RAG.

`regression` scores LeaseGuard's small internal lease records. It is a product check, not a lab benchmark, and cannot support a breakthrough claim.

Official evaluator clones and full test-set scoring run only in the Colab notebook after `LEASEGUARD_ALLOW_BENCHMARK_DOWNLOAD=1` is set. The local machine must not download those corpora.
