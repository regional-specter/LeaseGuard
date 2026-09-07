# Phase 2 Validation Records

These small annotation records test whether LeaseGuard's ontology can represent real office, retail-like, and amendment language.

The raw source PDFs are downloaded to `data/raw/validation`, which is ignored by Git. The committed source manifest records URLs, checksums, rights status, and intended use.

Current records:

- `gsa_office_template.extraction.json` validates an official GSA office lease template.
- `gsa_amendment_template.extraction.json` validates related-document and amendment effects.
- `pandadoc_retail_template.extraction.json` validates an MIT-licensed retail-capable commercial lease template.
- `st_joseph_retail.extraction.json` validates a municipal concession lease with retail-like operations.

The GSA documents are federal government templates. The St. Joseph document remains local-validation-only until its reuse rights are confirmed. This distinction is enforced in source metadata and must carry forward into dataset selection.

These records are representative validation samples, not complete document annotations and not training data.
