# Milestone 7 Retrieval-Readiness Audit

Date: 2026-07-23

Audited baseline: package 0.7.0, Git commit `9f6ff53`

Milestone 8 package target: 0.8.0

## Scope and files inspected

The audit was completed before and during retrieval implementation. It covered:

- every reusable package source under `src/localml_scholar/`;
- all existing tests under `tests/`, including checkpoint migration and
  tokenizer identity coverage;
- `README.md`, `pyproject.toml`, `.gitignore`, `docs/architecture.md`,
  `docs/roadmap.md`, all existing audit and derivation documents;
- all training, generation, tokenizer, and inspection scripts under
  `experiments/`;
- configuration files, committed test fixtures, output-directory policy, and
  Git tracked/untracked state;
- the new retrieval modules and tests after implementation.

Repository-wide file inspection used `rg --files`, targeted `rg` searches, and
direct reads of the relevant source, test, configuration, documentation, and
experiment files. No existing document-metadata or lexical-index abstraction
was present.

## Readiness findings

### Reusable infrastructure

- `serialization.atomic_write_text` already provided parent creation,
  same-directory temporary files, `fsync`, and atomic `os.replace`. The
  retrieval index and JSON experiment summaries reuse it.
- `tokenizer._canonical_json` and corpus/tokenizer SHA-256 patterns established
  deterministic state conventions. The function is private and accepts
  tokenizer-specific assumptions, so retrieval defines a strict public-local
  `canonical_json` that rejects lossy tuple conversion, non-string mapping
  keys, and non-finite numbers.
- Package versioning is centralized in `_version.py` and duplicated in
  `pyproject.toml`; both are updated together.
- Existing experiments write explicit JSON summaries below `outputs/`.
- Existing language-model tokenizers use `normalization="none"`. That policy
  cannot be reused implicitly for word retrieval, so the lexical tokenizer has
  its own explicit, serialized configuration.

### Assumptions verified

- The model, tokenizer, training, and generation code does not require a
  retrieval object or assume that every local document is one model-training
  corpus.
- `CorpusMetadata` describes training streams and is not safe to repurpose as a
  source-document model: it intentionally omits source text, sections, pages,
  and citation ranges.
- Local source paths appeared only as explicit strings in corpus metadata and
  CLIs. No canonical logical-source utility existed.
- Transformer import and construction are separate from the new
  `localml_scholar.retrieval` package. Search tests confirm that retrieval can
  run without constructing a model.
- `.gitignore` excludes `outputs/*`, raw/processed corpora, checkpoints,
  caches, virtual environments, and macOS metadata while retaining only
  intentional `.gitkeep` files.
- The committed retrieval corpus is short, purpose-built text created for this
  project. It contains no downloaded or large paper content.

### Verified issues and fixes

1. There was no retrieval-specific canonical JSON validator. Reusing the
   tokenizer-private helper would silently convert tuples to arrays. A strict
   retrieval canonicalizer now rejects state that would not round-trip exactly.
2. The first citation implementation validated display text on reload but did
   not validate malformed page/line ranges at construction. `Citation` now
   validates identifiers, heading paths, paired positive page bounds, and
   positive ordered line bounds. Tests mutate serialized citation locations and
   require explicit failures.
3. Serialized chunks initially relied on surrounding document/index validation
   for several fields. `Chunk` now directly validates its chunking SHA-256,
   heading path, optional page range, and metadata type.
4. Index construction initially assumed term-frequency rows and hash strings
   had already been validated by the builder. The constructor and loader now
   reject malformed sparse rows, invalid document frequencies/hashes, and
   chunk-to-document/section/source mismatches before search.
5. The first deterministic-order test repeated a no-result query and therefore
   did not exercise a scored tie. It was replaced with equal-length,
   equal-score documents and validates the documented tie keys for both TF-IDF
   and BM25.
6. A configured overlap could exceed `minimum_characters`. If the first
   preferred paragraph boundary was shorter than that overlap, the next chunk
   start would fail to advance. Split-boundary selection now requires a split
   chunk to be at least `overlap_characters + 1` characters long, and a
   regression test exercises this exact case.
7. Empty externally extracted PDF pages were first represented by
   separator-only sections. Ingestion could construct these, but index building
   correctly rejected their zero lexical terms. The adapter now joins only
   non-empty page strings, creates page-linked sections only for those strings,
   and records every omitted empty page number in inferred metadata. An
   integration test indexes pages 1 and 3 around an empty page 2 and requires
   the page-3 citation.
8. Direct execution of the two new experiment scripts initially depended on
   the repository root already being on `sys.path`. Both now add the repository
   root in the same explicit manner as existing standalone experiments, and
   their subprocess integration tests cover direct invocation.

No verified defect required a change to transformer mathematics, tokenizer
IDs, training behavior, generation, optimizer state, or checkpoint migration.

## Serialization decisions

The Milestone 8 index is one deterministic, UTF-8, non-pickle JSON snapshot.
It includes:

- format version, creating package version, and index type;
- document, section, chunk, source text, and citation-relevant metadata;
- chunking, lexical, BM25, and duplicate-content configuration;
- sorted vocabulary, sparse term-frequency rows, document frequencies, and
  average chunk length;
- corpus and complete-index SHA-256 identities.

Save is atomic. Load first parses into local objects, rejects missing and
unexpected keys, validates every nested model, verifies all hashes, rebuilds
statistics from documents/configuration, and requires the rebuilt state to
equal the serialized state before returning an index. A failed load cannot
partially mutate an existing index.

Indexes are immutable snapshots. Any added, removed, moved, or changed source,
or relevant configuration change, requires a deterministic full rebuild.
`change_reasons` reports these differences. This avoids an early mutable-index
protocol and stale partial state.

## Ingestion scope and PDF policy

Core ingestion accepts non-empty `.txt`, `.md`, and `.markdown` UTF-8 files.
Strict decoding is the default; replacement decoding is available only through
an explicit `errors="replace"` request. Original decoded text and all
whitespace are preserved.

Markdown support is deliberately narrow: ATX headings outside triple-backtick
or triple-tilde fences establish sections and heading paths. Content,
paragraphs, lists, quotes, inline equations, and fences remain exact source
text. Setext headings, HTML structure, front matter, nested container parsing,
and full CommonMark rendering are not implemented.

The PDF path accepts only ordered, externally extracted `PageText` values. It
preserves provided page numbers for non-empty page strings, records empty page
numbers without creating artificial text, labels the parser policy
`external_page_text_v1`, and does not parse PDF bytes, perform OCR, infer
reading order, or claim extraction accuracy. No PDF dependency is required.

## Citation policy

Citations are derived only from serialized source metadata:

- known PDF-derived pages take precedence over line display;
- otherwise exact source line ranges are used;
- a real heading path is appended when present;
- title is used only if supplied or directly inferred from an H1 heading;
- authors are returned only from a valid user-supplied list;
- page numbers, authors, publication dates, and section numbers are never
  invented.

The structured citation stores the exact document and chunk IDs and validates
that its formatted display survives index reload unchanged.

## Verification

The final verification commands and measured results are recorded here after
the complete implementation pass:

```text
python3 -m ruff format .
python3 -m ruff check .
python3 -m ruff format --check .
git diff --check
python3 -m pytest -q
python3 experiments/train_bigram.py --config configs/bigram_small.json
python3 experiments/train_mlp_xor.py
python3 experiments/inspect_single_head_attention.py
python3 experiments/inspect_pre_norm_decoder_block.py
python3 experiments/inspect_multi_head_attention.py
python3 experiments/compare_single_and_multi_head.py --steps 20
python3 experiments/inspect_bpe_tokenizer.py
python3 experiments/compare_tokenizers.py --steps 12
python3 experiments/inspect_document_ingestion.py
python3 experiments/compare_retrievers.py
PYTHONPATH=src python3 -m localml_scholar.retrieval.search build \
  --sources tests/fixtures/retrieval/attention.md \
    tests/fixtures/retrieval/optimization.md \
    tests/fixtures/retrieval/probability.txt \
  --output outputs/retrieval_cli_smoke/index.json
PYTHONPATH=src python3 -m localml_scholar.retrieval.search inspect \
  --index outputs/retrieval_cli_smoke/index.json
PYTHONPATH=src python3 -m localml_scholar.retrieval.search search \
  --index outputs/retrieval_cli_smoke/index.json \
  --query "How does causal masking prevent leakage?" \
  --method bm25 --top-k 3 --verbose
```

Python 3.13.5 and NumPy 2.4.3 were used. Ruff 0.15.18 reported no lint
violations and no formatting changes were required after the final format
pass. `git diff --check` was clean. Pytest 9.1.1 reported `481 passed in
6.59s`: the original 394 tests plus 87 retrieval tests.

Every listed experiment exited successfully. The bigram smoke retained best
validation loss `1.5488034950125846`; XOR retained predictions
`[0, 1, 1, 0]`; attention/decoder inspections retained exact future-mask
zeros; the one-/two-head, character/byte/BPE, interruption/resumption, and
checkpoint-reload demonstrations completed successfully.

The ingestion inspection produced 3 sections and 4 chunks, validated complete
source coverage, and preserved state/results exactly after reload. Its first
BM25 result scored `4.416544034773` with citation
`[Causal Attention, lines 3–9, § Causal Attention › Masking Future Positions]`.

The controlled retrieval corpus contained 3 documents, 9 chunks, a vocabulary
of 121 terms, and 5 judged queries. Both retrievers measured Precision@1
`1.0`, Precision@3 `0.3333333333333333`, Recall@3 `1.0`, MRR `1.0`, and Hit
Rate@3 `1.0`. All rankings were exactly preserved after reload. The serialized
fixture index was 24,260 bytes. This validates a deliberately simple fixture
only and is not a general quality claim.

Artifact inspection parsed all 48 generated JSON files, opened all 66 NPZ
files with `allow_pickle=False`, and reconstructed all 3 generated retrieval
indexes. Git reported generated outputs and caches as ignored; only
`data/raw/.gitkeep`, `data/processed/.gitkeep`, and `outputs/.gitkeep` were
tracked in those generated-data locations.

## Explicitly deferred

- answer generation, prompt construction, RAG, and transformer retrieval use;
- neural embeddings, vector databases, semantic search, reranking, and phrase
  search;
- direct PDF parsing, layout analysis, equations/figure extraction, OCR, and
  scanned-document handling;
- mutable or incremental index updates;
- stemming, lemmatization, stop-word removal, Unicode normalization, and
  language-specific tokenization;
- inverted-index or memory optimization;
- claims of general retrieval quality from the five-query fixture.
