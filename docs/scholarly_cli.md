# Scholarly CLI

The scholarly CLI consumes an existing retrieval index. It does not construct a
transformer and does not contact external services.

```bash
PYTHONPATH=src python3 -m localml_scholar.scholarly.cli analyze \
  --index outputs/scholarly_inspection/fixture_index.json \
  --document-id DOCUMENT_ID \
  --json \
  --output outputs/scholarly_cli/analysis.json
```

Commands:

- `analyze`
- `glossary`
- `equations`
- `methods`
- `experiments`
- `summary`
- `reproduction-checklist`
- `compare`
- `research-gaps`
- `inspect`

`compare` and `research-gaps` accept repeated `--document-id` arguments.
`--retrieval-method` selects the base method used by optional scholarly
retrieval. `--section-role` filters `analyze`, `glossary`, `equations`,
`methods`, and `experiments` to source sections carrying the requested role; it
rejects an absent role and never silently invents or substitutes one.
Aggregate summary, checklist, comparison, gap, and inspection commands reject
section filtering because their completeness semantics are paper-wide.
`--json` emits machine-readable payloads.
Without it, glossary, checklist, and comparison commands use Markdown tables.
`--output` writes a validated versioned artifact.

Missing and ambiguous fields remain visible. A citation establishes exact
source provenance, not semantic correctness.
