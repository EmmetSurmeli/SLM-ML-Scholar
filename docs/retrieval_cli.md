# Retrieval CLI

Install the package in editable mode as described in `README.md`, or prefix
commands from a source checkout with `PYTHONPATH=src`.

## Build

```bash
python3 -m localml_scholar.retrieval.search build \
  --sources notes.md appendix.txt \
  --output outputs/local_documents/index.json
```

Source paths are explicit; the command does not scan directories or download
data. Supported suffixes are `.txt`, `.md`, and `.markdown`. Input order does
not determine index order.

Boundary controls are:

- `--target-characters` (default `600`);
- `--maximum-characters` (default `900`);
- `--overlap-characters` (default `100`);
- `--minimum-characters` (default `80`).

The command prints document, section, chunk, vocabulary, hash, and average
chunk-length statistics.

## Inspect

```bash
python3 -m localml_scholar.retrieval.search inspect \
  --index outputs/local_documents/index.json
```

Inspection validates and reconstructs the index, then prints embedded
documents, sections, chunks, vocabulary, document frequencies, hashes, and
average chunk length.

## Search

```bash
python3 -m localml_scholar.retrieval.search search \
  --index outputs/local_documents/index.json \
  --query "How does causal masking prevent leakage?" \
  --method bm25 \
  --top-k 5 \
  --verbose
```

`--method` is `bm25` by default and may be `tfidf`. Positive-score results are
ranked deterministically. Human output prints the rank, score, source,
citation, and exact retrieved passage. `--json` emits structured output.
`--verbose` includes highlighted display text, term contributions, and scoring
details; highlighting does not modify stored source text.

Explicit filters are:

- `--document-id`;
- `--source-name`;
- `--media-type`;
- repeated `--heading-prefix` values;
- `--publication-year`;
- `--collection`.

The CLI never infers filters from the query. It never constructs an answer:
search JSON always contains `"answer_generated": false`.
