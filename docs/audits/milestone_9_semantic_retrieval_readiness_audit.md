# Milestone 9 Semantic-Retrieval Readiness Audit

Audit date: 2026-07-24

Scope: the version 0.9.0 retrieval and grounded-answer implementation before
Milestone 10 changes.

## Files inspected

The audit covered every file under:

- `src/localml_scholar/retrieval/`
- `src/localml_scholar/answering/`
- retrieval and answering tests under `tests/`
- retrieval and grounded-QA fixtures
- retrieval and answering experiments
- `docs/retrieval_index_format.md`
- `docs/retrieval_cli.md`
- `docs/answering_cli.md`
- `docs/answer_artifact_format.md`
- relevant README, architecture, roadmap, package-version, and ignore rules

Document, section, chunk, and citation validation in
`retrieval/documents.py` and exact source reconstruction in
`retrieval/index.py` were inspected directly.

## Reusable foundations

- Documents, sections, and chunks have deterministic content-derived IDs.
- Chunks retain exact character, line, heading, and known page ranges.
- The lexical index already stores ordered chunks, a sorted vocabulary,
  per-chunk term frequencies, document frequencies, and BM25 configuration.
- TF-IDF uses independent, deterministic sublinear term-frequency and smooth
  inverse-document-frequency formulas.
- Search filters are explicit and are not inferred from natural-language
  queries.
- Lexical ties use score, document ID, chunk ordinal, and chunk ID.
- Index JSON is canonical, atomically replaced, and transactionally validated.
- Evidence and answer artifacts bind selected source text to the index hash,
  chunk ID, document ID, exact citation, and evidence hash.
- Answer validation rechecks source text and citation identity against the
  loaded index.

These properties allow semantic retrieval to select chunk identities without
reconstructing or weakening source metadata.

## Semantic baseline selected

The implementation selects **TF-IDF latent semantic analysis with NumPy SVD**.

This is appropriate because the project has deterministic lexical statistics
and NumPy but has neither:

- a trained and evaluated project-native dual encoder,
- enough authored data to train one credibly,
- nor a supplied, licensed local static embedding artifact.

LSA is transparent, local, deterministic after sign canonicalization, and
small enough for exact CPU fixture evaluation. It is documented as a linear
distributional baseline, not a modern neural embedding model and not deep
semantic understanding.

Alternatives considered:

1. A project-native contrastive encoder was deferred because the current
   fixtures cannot support honest training or generalization claims.
2. Static vectors remain possible behind a future adapter, but no local
   artifact was supplied and downloads are prohibited.
3. External embedding APIs, sentence-transformer packages, and vector
   databases violate the milestone constraints.

## Verified defects and integration gaps

1. `SearchResult` and `EvidenceItem` accepted only `tfidf` and `bm25`, so a
   new method could not flow through evidence artifacts.
2. `EvidenceSelectionConfig` and both CLIs exposed only lexical method names.
3. The answering metadata retained result rank and score but omitted full
   component scoring details needed to audit fusion and reranking.
4. Retrieval metrics lacked Average Precision, graded nDCG, mean relevant
   rank, no-result rate, and category aggregation.
5. Index format version 1 had no semantic-state slot and required the current
   package version exactly, preventing the requested explicit 0.8.0/0.9.0
   lexical compatibility policy.
6. Evidence selection assumed `matched_terms` represented exact lexical
   overlap. Semantic results required an explicit documented convention for
   query terms that contributed to projection.
7. No deterministic SVD sign policy, semantic configuration hash, vocabulary
   alignment hash, or semantic enrichment operation existed.

No defect was found in lexical TF-IDF/BM25 ranking, chunk identity, citation
construction, answer citation revalidation, or answer fallback semantics.

## Fixes made

- Added immutable LSA configuration and semantic state with float64 factors.
- Added exact matrix construction, rank checks, full-SVD reconstruction
  validation, truncation diagnostics, sign canonicalization, and state hashes.
- Added explicit lexical-only enrichment that preserves documents, chunks,
  corpus hash, lexical statistics, lexical rankings, and the original file.
- Added semantic, weighted hybrid, RRF, and hybrid-reranked method dispatch
  with no hidden method substitution.
- Added bounded deterministic candidate union and exact component metadata.
- Added a manually weighted reranker with phrase, heading, term, rare-term,
  number, identifier, length, and source-range redundancy features.
- Generalized evidence and answer artifacts to the new canonical method names.
- Recorded complete hybrid/reranking configurations and result explanations in
  answer metadata.
- Added MAP, nDCG, mean relevant rank, no-result rate, and per-category metrics.
- Added version 2 retrieval snapshots plus transactional loading of recognized
  0.8.0/0.9.0 lexical-only version 1 snapshots.

## Determinism policy

- Vocabulary and chunk order come only from the lexical snapshot.
- LSA uses no random state.
- Every retained SVD component has a deterministic sign based on the first
  largest-absolute right-singular-vector entry.
- All semantic fixture state is float64.
- Search and reranking ties use immutable source identity fields.
- Serialization is sorted canonical JSON and atomic text replacement.
- Reload recomputes lexical statistics and the complete semantic factorization
  before returning an index.

## Numerical policy

- Matrix construction, SVD, embeddings, query projection, cosine, and stored
  factors use float64.
- Non-finite inputs, factors, embeddings, norms, and scores raise.
- The requested dimension must not exceed effective numerical rank.
- The full SVD must reconstruct within the configured relative tolerance.
- Truncation error and squared singular-value fraction are reported, not
  hidden.
- Zero query vectors return no semantic results; all-zero fit matrices raise.

## Compatibility policy

- Current format version 2 may be lexical-only or semantically enriched.
- Recognized version 1 snapshots from package 0.8.0 or 0.9.0 load as
  lexical-only indexes.
- Semantic search on a lexical-only index raises and instructs the caller to
  enrich explicitly.
- Enrichment writes a separate output path through the CLI.
- An already enriched snapshot rejects a conflicting enrichment
  configuration; callers must enrich the original lexical snapshot.
- Lexical result construction and formulas are unchanged.
- Overall index hashes change when semantic state is attached; corpus and
  source identities do not.

## Tests added

Tests cover matrix values and alignment, zero rows, all-zero rejection, SVD
reconstruction and sign equivalence, rank validation, query projection and
OOV handling, exact cosine, semantic ties and filters, citation preservation,
enrichment/reload identity, legacy compatibility, malformed state, weighted
fusion, RRF, reranking features and totals, source redundancy, advanced
metrics, CLI behavior, answer integration, fixture identity, ablations,
sensitivity analysis, and grounded-answer regression.

## Exact verification commands

The final verification commands and results were:

```bash
python3 -m ruff check .
python3 -m ruff format --check .
git diff --check
python3 -m pytest -q
python3 experiments/evaluate_retrieval_ablation.py
python3 experiments/analyze_hybrid_sensitivity.py
python3 experiments/inspect_semantic_retrieval.py
python3 experiments/evaluate_grounded_retrievers.py
```

Results:

- Ruff lint: passed.
- Ruff formatting check: 137 files already formatted.
- `git diff --check`: passed.
- full pytest: 571 passed in 3.31 seconds.
- retrieval ablation: completed for six methods and 13 queries.
- semantic inspection: exact state and result equality after reload.
- hybrid sensitivity: 13 disclosed settings plus three redundancy thresholds.
- grounded-answer regression: four retrieval methods, citation validity and
  citation coverage 1.0 for every method.
- legacy bigram, XOR, attention, decoder, tokenizer, transformer
  train/resume, lexical retrieval, extractive answering, explicit-checkpoint
  rejection/fallback, and CLI smoke experiments: passed under package 1.0.0.

## Deferred features

- neural dual encoders and contrastive training
- supplied static-vector adapters
- neural cross-encoders
- approximate nearest-neighbor indexes
- vector databases
- model or corpus downloads
- learned reranking weights
- large-corpus sparse/truncated SVD
- paper-structure specialization, equation extraction, and cross-paper tools

Dense LSA and exact scanning are intentionally small-corpus reference
implementations.
