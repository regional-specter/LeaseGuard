# Inference Configuration

`prompts.v1.json` freezes the Phase 6 prompt templates shared by every unmodified base-model comparison. Direct and schema-constrained variants exist for extraction and for evidence-grounded answers.

Runtime, quantization, context length, and candidate pins live in `configs/evaluation/baselines.v1.json` so evaluation and inference stay on one frozen matrix.
