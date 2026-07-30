# Milestone 10 scholarly-analysis readiness audit

Date: 2026-07-28  
Starting package version: 1.0.0  
Scope: readiness of the verified ingestion, retrieval, and grounded-answer
systems for deterministic paper-specific analysis.

## Files and behavior inspected

The audit covered:

- `retrieval/documents.py`, `ingestion.py`, `chunking.py`, `text.py`,
  `index.py`, `search.py`, `semantic.py`, and `hybrid.py`
- `answering/models.py`, `evidence.py`, `context.py`, `citations.py`,
  `validation.py`, `serialization.py`, `pipeline.py`, and both existing CLIs
- retrieval and answer artifact documentation
- authored text, Markdown, page-derived PDF, table-like, equation-like,
  Unicode, code-block, citation, and hash tests
- project serialization and atomic-write helpers

## Reusable infrastructure

`Document` preserves exact UTF-8 source text, content hashes, immutable ordered
sections, headings, line ranges, and optional page ranges. Its constructor
proves that sections cover the source without gaps or rewriting. `Chunk`
preserves exact searchable slices and index identity. Retrieval results retain
the original chunk citation, while grounded-answer validation checks character
ranges and exact substrings against the indexed document.

The existing canonical JSON and atomic-write helpers are suitable for
transactional scholarly artifacts. The retrieval index supplies stable
document identity and makes analysis independent of transformer construction.

## Readiness gaps and verified fixes

No defect was found in source reconstruction, lexical/semantic scores,
tokenizer IDs, transformer mathematics, or grounded-answer citations.

The following integration gaps were verified and addressed:

1. Chunk citations do not represent arbitrary scholarly substrings. A separate
   `SourceCitation` now records exact character, line, section, page, and source
   hash information.
2. The prior package had no policy for equation-like extracted text. Explicit,
   non-overlapping text-only detection and conservative normalization were
   added.
3. Metadata, methods, experiments, and results lacked a common source-linked
   extraction model. `ScholarlyEvidence` now enforces exact text/hash linkage.
4. Paper-wide deduplication initially removed repeated dataset and metric
   occurrences needed for experiment scope. Occurrences are now retained by
   source range.
5. An early sentence splitter treated decimal points as sentence endings.
   Paragraph-aware punctuation boundaries now preserve decimal values.
6. Equation numbers immediately before `$$` or `\]` were initially missed.
   The explicit-number rule now recognizes those trailing delimiters.
7. Bumping the package version would have rejected valid 1.0.0 version-2
   retrieval snapshots. The loader now recognizes 1.0.0 and 1.1.0 version-2
   indexes while retaining full hash validation.
8. In-text reference markers were initially retained as unresolved even for
   exact local matches. Numbered markers and unique author/year markers now
   resolve deterministically; ambiguous matches remain unresolved.
9. The first CLI section-role option validated presence without filtering the
   payload. Extraction commands now return exact source-section views, while
   paper-wide aggregate commands reject filtering explicitly.

## Equation policy

Only textual representations already present in a `Document` are analyzed:
LaTeX-like delimiters, equation environments represented as text, Unicode
operators, and narrow operator-heavy lines. Raw text and exact offsets are
retained. No LaTeX is executed; no algebra is simplified; no equation
equivalence is inferred; malformed delimiters are safe. Visual PDF equations
are outside scope.

## Section-classification policy

Classification uses public heading fragments plus two narrow explicit-content
rules. Original headings and unknown sections are retained. Multiple roles are
allowed when a heading explicitly supports them. Reasons and categorical
confidence accompany every classification. No learned classifier is hidden in
the pipeline.

## Extraction-confidence policy

`high`, `medium`, and `low` are heuristic categories, not probabilities.
`validated`, `ambiguous`, `unresolved`, `conflicting`, and `not_found`
describe extraction state. A source citation validates provenance and range;
it does not prove semantic truth or entailment.

## Mathematical-text limitations

- Symbols inside extracted equations can be ambiguous.
- LaTeX commands are filtered conservatively; arbitrary macros are unsupported.
- Definitions are linked by explicit phrases and source distance, not semantic
  theorem proving.
- Cross-section symbol meaning changes are retained as conflicts.
- The system never generates a missing derivation.

## PDF-derived text limitations

The repository accepts externally extracted page text only. Page boundaries are
preserved, but glyph order, reading order, columns, superscripts, equations,
tables, captions, and references are only as reliable as supplied text. There
is no OCR, visual layout recovery, or figure interpretation.

## Table limitations

The adapter accepts explicit Markdown pipe tables or consistently delimited
rows. It rejects inconsistent widths and never guesses merged cells. It does
not recover visual PDF tables or fabricate row/column coordinates.

## Exact verification commands

```bash
python3 -m ruff check .
python3 -m ruff format --check .
git diff --check
python3 -m pytest -q
python3 experiments/evaluate_scholarly_extraction.py
python3 experiments/evaluate_reproduction_checklists.py
python3 experiments/evaluate_paper_comparison.py
python3 experiments/evaluate_research_gap_candidates.py
python3 experiments/inspect_scholarly_analysis.py
```

The focused scholarly suite reported 41 passing tests. The complete result is
`613 passed in 7.01s`. Ruff lint and formatting checks and `git diff --check`
also passed. Every prior documented experiment and the retrieval/
grounded-answer regressions were rerun successfully.

## Explicitly deferred

- OCR and PDF layout recovery
- visual equation, table, chart, and figure interpretation
- external metadata or reference resolution
- symbolic algebra and equivalence proving
- learned section classifiers, embeddings, or rerankers
- literature-wide novelty verification
- instruction tuning and generated scholarly explanation
