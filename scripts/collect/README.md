# Collection Scripts

```bash
uv run python scripts/collect/process_documents.py validate-manifest
uv run python scripts/collect/process_documents.py process \
  --raw-root /tmp/leaseguard/raw \
  --processed-root /tmp/leaseguard/processed \
  --checkpoint-dir /tmp/leaseguard/checkpoints \
  --search-root .
```

Local runs import files that already exist. Full HTTPS downloads require Colab with `LEASEGUARD_ALLOW_DOCUMENT_DOWNLOAD=1` and `--allow-download`.
