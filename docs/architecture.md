# Architecture

LeaseGuard combines document processing, retrieval, deterministic validation, and a fine-tuned language model. Component boundaries favor traceable evidence, local operation, and replaceable parts.

## Document pipeline

An approved source is processed from its manifest entry into a validated intermediate record:

1. **Acquire** a local copy or, in Colab only, an HTTPS download.
2. **Validate** media type, size, and SHA-256 without trusting the filename alone.
3. **Parse** digital PDF pages, DOCX headings and tables, or UTF-8 markdown/text.
4. **Redact** emails, phone numbers, and SSN-shaped values in the processed copy only.
5. **Segment** headings, paragraphs, tables, and exhibits with character offsets into the normalized text.
6. **Dedupe** exact checksum matches and near-duplicate shingles across the batch.
7. **Link** leases, amendments, and exhibits by `family_id` and `related_document_ids`.
8. **Export** a `ProcessedDocument` JSON record and a resumable checkpoint.

The raw file is never overwritten. OCR is an explicit future extension and is disabled in pipeline configuration. User documents are out of scope for this training-source pipeline.

Heavy runs use `notebooks/colab_orchestrator.ipynb` with Drive-backed `raw`, `processed`, `checkpoints`, and `reports` directories.
