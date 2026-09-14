# Evaluation Policy

LeaseGuard uses published, expert-built benchmarks as its primary research standard. A high score on a private test created by the project is not enough to support a major performance claim.

The project will still keep a small lease-specific regression set. That set protects product behavior, but it must not be advertised as an independent professional benchmark.

## External Benchmark Suite

### LegalBench

[LegalBench](https://hazyresearch.stanford.edu/legalbench/) is the headline legal reasoning benchmark. It was published in the NeurIPS 2023 Datasets and Benchmarks track and contains 162 tasks assembled from 36 sources by 40 contributors. Legal professionals helped design and hand-craft its tasks.

LeaseGuard should run the complete compatible LegalBench suite when practical. Contract interpretation and contract-understanding task groups should also be reported separately so a broad average does not hide weak contract performance.

LegalBench measures general legal reasoning rather than real-estate lease extraction. It is a professional external benchmark, but it is not sufficient by itself.

### CUAD

[CUAD](https://www.atticusprojectai.org/cuad/) is the primary contract clause extraction benchmark. It was published at NeurIPS 2021 and contains lawyer-supervised annotations across 41 clause categories.

The official test split and evaluator must remain unchanged. The original metrics are:

- Area under the precision-recall curve
- Precision at 80 percent recall
- Precision at 90 percent recall

Recall is especially important because missing a relevant clause is often more harmful than returning an extra candidate for review.

### ContractNLI

[ContractNLI](https://stanfordnlp.github.io/contract-nli/) is the primary evidence-grounded contract reasoning benchmark. It was published in Findings of EMNLP 2021.

The official test split measures whether a statement is entailed, contradicted, or not mentioned and whether the system identifies the correct evidence. Official metrics include:

- NLI accuracy
- Entailment F1
- Contradiction F1
- Evidence mean average precision
- Evidence precision at 80 percent recall

### LegalBench-RAG

[LegalBench-RAG](https://github.com/zeroentropy-ai/legalbenchrag) is the primary legal retrieval benchmark. It contains 6,858 queries over more than 79 million characters from expert-annotated legal datasets.

It measures character-level precision and recall for retrieved spans. This benchmark evaluates the retrieval component, not the conversational model by itself.

The lightweight mini version may be used during development. Final reports must use the complete benchmark.

## Benchmark Roles

The final report must keep different capabilities separate:

- LegalBench measures broad legal reasoning.
- CUAD measures commercial-contract clause extraction.
- ContractNLI measures evidence-grounded inference.
- LegalBench-RAG measures retrieval.
- LeaseGuard's regression set measures our specific JSON schema and office or retail workflow.

There will be no single blended “LeaseGuard score.” Combining unrelated metrics into one custom number would make comparison misleading.

## What Counts as a Breakthrough

LeaseGuard may claim strong or breakthrough benchmark performance only when:

1. The benchmark was published and professionally constructed.
2. The official test split and evaluator were used without modification.
3. The exact dataset and evaluator revisions were pinned.
4. The comparison model used the same protocol, prompts, context, and scoring rules.
5. LeaseGuard exceeds a strong reproducible published or open baseline.
6. Bootstrap confidence intervals or repeated runs show that the gain is not normal measurement noise.
7. All task-level results and regressions are published, not only the best aggregate.
8. Training-data overlap and possible benchmark contamination are disclosed.
9. The result can be reproduced from the Colab notebook without a paid API.

A private score, hand-picked example, changed test split, or model-graded result cannot support this claim.

## Test-Set Integrity

Official test examples must never be used for:

- Fine-tuning
- Continued pretraining
- Preference data
- Synthetic-data prompts
- Prompt selection
- Hyperparameter selection
- Error-driven training-example creation

Training and validation splits may be used according to the benchmark's official protocol. Test sets should be run only for frozen milestone candidates.

Public benchmarks may already appear in a base model's pretraining data. This cannot always be proven or removed. Every report must state this contamination risk and avoid claiming that the test set was certainly unseen by the base model.

## Reproducibility Record

Every benchmark run must save:

- Benchmark name and paper
- Dataset revision and checksum
- Official evaluator revision and checksum
- License
- Test split name
- Model name and exact revision
- Adapter and checkpoint revision
- Quantization
- Prompt template and chat template
- Context length
- Generation settings
- Random seed
- Hardware
- Software versions
- Raw predictions
- Official metric output
- Start and completion times

These records belong in remote experiment storage. Only small reports and manifests belong in Git.

## Acceptance Gates

Fine-tuning work should continue only when a candidate:

- Improves the target external benchmark over the unmodified base model
- Does not materially reduce other external benchmark scores
- Does not increase unsupported answers in the lease regression set
- Produces schema-valid output reliably
- Preserves evidence and source references

Numeric thresholds wait for unmodified Colab T4 baselines. `compare-baselines` records the measured winner and the dominant error class; it does not invent a blended score. Choosing gates before observing task difficulty would create arbitrary targets.

## Lease-Specific Regression Set

No established benchmark exactly matches LeaseGuard's US office and retail ontology. A small internal set is therefore still necessary for engineering safety.

It should check:

- JSON Schema validity
- Required evidence
- Premises, parties, dates, money, and obligations
- Amendment effects
- Landlord, tenant, and neutral perspective consistency
- Missing and conflicting information
- Refusal of state-law conclusions

This set is a regression suite, not the headline benchmark. It must always be described that way.

## Repeatable Command

Local machines can inspect the frozen professional suite and run product regression:

```bash
uv run python scripts/evaluate/run_benchmark.py list
uv run python scripts/evaluate/run_benchmark.py validate-registry
uv run python scripts/evaluate/run_benchmark.py regression
uv run python scripts/evaluate/run_benchmark.py validate-baselines
uv run python scripts/evaluate/run_benchmark.py list-candidates
```

`list` prints LegalBench, CUAD, ContractNLI, and LegalBench-RAG. `regression` scores the internal office and retail records and must be reported as regression, not as a lab result.

Official evaluator clones and full test-set scoring run only from `notebooks/colab_orchestrator.ipynb` after `LEASEGUARD_ALLOW_BENCHMARK_DOWNLOAD=1` is set. The local machine must not download those corpora.

## Colab Execution

All full benchmark runs will execute from the single Google Colab T4 notebook. The notebook will:

1. Clone the exact repository revision.
2. Install pinned benchmark and model dependencies.
3. Mount approved remote storage.
4. Download only the selected benchmark.
5. Run a named benchmark configuration.
6. Save raw predictions and resumable state.
7. Run the official evaluator.
8. Export a small report and run manifest.

The local machine will run metric unit tests, validate small result fixtures, and execute the internal regression command.

## Unmodified Base-Model Baselines

Phase 6 scores Qwen3.5-4B, Qwen3.5-9B, and Gemma 3 12B on the frozen Dataset v1 evaluation split before any LeaseGuard adapter is trained. Each candidate uses the same prompts, context builders, and generation settings. The comparison matrix is:

- Direct prompting versus schema-constrained JSON prompting
- Full-document context versus clause-level context versus keyword retrieval
- 4-bit NF4 inference on Tesla T4, plus 8-bit only for the 4B candidate

Product-task scores are unofficial. They cannot support a breakthrough claim. Official LegalBench, CUAD, ContractNLI, and LegalBench-RAG numbers still require those evaluators.

Run one candidate per Colab T4 session:

```text
notebooks/colab_baseline_qwen35_4b.ipynb
notebooks/colab_baseline_qwen35_9b.ipynb
notebooks/colab_baseline_gemma3_12b.ipynb
notebooks/colab_baseline_compare.ipynb
```

`compare-baselines` ranks practical T4 runs lexicographically (schema validity, extraction F1, answer status, evidence recall, speed, lower peak VRAM) and writes a fine-tuning hypothesis. Scripted local dry-runs are not eligible for selection.
