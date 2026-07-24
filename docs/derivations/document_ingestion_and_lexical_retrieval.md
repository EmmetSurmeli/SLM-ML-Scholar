# Document Ingestion and Lexical Retrieval

## 1. Why retrieval is separate from generation

Retrieval and answer generation solve different problems. Retrieval selects
source passages for a query and preserves where every passage came from.
Generation may later explain those passages, but it must not decide what the
source text was or invent its provenance.

Milestone 8 therefore ends at a ranked tuple of exact `SearchResult` passages.
`RetrievalIndex` imports no language-model module, constructs no transformer,
and generates no answer. A newly loaded document changes an immutable lexical
index; it does not require model training.

## 2. Document identity

Ingestion preserves the decoded source string \(x\) exactly. Its content
identity is

\[
h_x = \operatorname{SHA256}(\operatorname{UTF8}(x)).
\]

The canonical document identifier is a truncated display form of another
SHA-256 digest over a canonical JSON array containing the normalized logical
source identifier and \(h_x\):

\[
\operatorname{id}_{doc}
=
\operatorname{prefix}_{24}
\left(
\operatorname{SHA256}
([\operatorname{source}, h_x])
\right).
\]

Consequently, a content change changes both `content_sha256` and
`document_id`. Moving otherwise identical text to a different logical source
also changes `document_id`, while the unchanged content hash makes the nature
of the change distinguishable.

Source paths are normalized lexically (`\` to `/`, repeated separators
collapsed) but are not resolved against the current machine. Modification
times are not identities. User metadata and inferred ingestion metadata are
stored in separate `metadata["user"]` and `metadata["inferred"]` objects, and
both must be exactly JSON-serializable.

## 3. Sections and chunks

Plain text has one unnamed root section. The narrow Markdown parser recognizes
ATX headings outside fenced code blocks and maintains a heading stack, so a
level-three heading can have a path such as
`("Methods", "Training", "Optimizer")`. A section is an exact half-open source
slice:

\[
\operatorname{section.text}
=
x[\operatorname{start}:\operatorname{end}].
\]

Section slices are ordered, non-empty, and together cover the entire document
without gaps. Markdown markup, whitespace, equations, lists, quotes, and code
remain in the source text; the parser identifies boundaries rather than
rendering Markdown.

Chunking operates inside one section at a time. When the remaining section is
larger than the hard maximum, the boundary preference is:

1. paragraph break;
2. a deliberately simple punctuation-plus-whitespace heuristic;
3. whitespace;
4. the hard maximum.

The selected boundary favors the configured target while never exceeding the
maximum. A fenced code block is kept whole if one of its boundaries fits the
legal window; an oversized fence may be split at the hard maximum. This is a
source-preserving heuristic, not a complete Markdown or sentence parser.

For adjacent chunks in the same section with overlap \(o\),

\[
\operatorname{start}_{i+1} = \operatorname{end}_i-o.
\]

`validate_chunk_coverage` verifies exact source slices, bounds, ordering,
fixed same-section overlap, lack of gaps, and full-document coverage. Section
boundaries themselves have no artificial overlap. Chunk identities include
the document ID, section ID, source offsets, chunking-configuration hash, and
chunk-content hash. Boundary or configuration changes therefore change the
affected chunk identity.

## 4. Retrieval text normalization

Display and citation text is never replaced by indexed text. The default
retrieval policy case-folds each matched term but applies no Unicode
normalization (`normalization="none"`). Source character spans continue to
address the original Python string.

This policy is intentionally independent from character, byte, and BPE
language-model tokenization. Search terms and language-model tokens serve
different purposes and have different checkpoint identities.

## 5. Lexical tokenization

The reference regular expression emits Unicode word-like terms and their exact
source spans. Hyphens split terms. Apostrophes inside words and underscores
inside identifiers are preserved. Camel case is not split. Decimal forms such
as `3.14` remain one numeric term. Punctuation and standalone mathematical
symbols are delimiters, while adjacent Unicode letters are retained.

For chunk \(d\), the raw term frequency is

\[
f_{t,d} = \text{number of occurrences of term }t\text{ in }d.
\]

With \(N\) chunks, document frequency is counted at chunk granularity:

\[
df_t = \sum_{d=1}^{N}\mathbf{1}[f_{t,d}>0].
\]

The vocabulary and every sparse map are stored in sorted term order for
deterministic state and diagnostics.

## 6. TF-IDF

For a positive raw frequency, sublinear term frequency is

\[
\operatorname{tf}(t,d) = 1+\log f_{t,d},
\]

and it is zero when \(f_{t,d}=0\). Smoothed inverse document frequency is

\[
\operatorname{idf}(t)
=
\log\left(\frac{N+1}{df_t+1}\right)+1.
\]

The sparse coordinate weight is

\[
w_{t,d}=\operatorname{tf}(t,d)\operatorname{idf}(t).
\]

A query uses the same term-frequency and corpus-IDF rules. Its cosine score
against chunk \(d\) is

\[
\operatorname{score}_{\mathrm{tfidf}}(q,d)
=
\frac{\sum_t w_{t,q}w_{t,d}}
{\sqrt{\sum_t w_{t,q}^2}\sqrt{\sum_t w_{t,d}^2}}.
\]

Only shared terms are multiplied. A zero norm produces score zero, not a
division by zero. Query terms outside the vocabulary are omitted. The result
records the numerator, both norms, and each shared term's normalized score
contribution.

## 7. BM25

BM25 is the default retriever. For defaults \(k_1=1.2\) and \(b=0.75\), its
inverse document frequency is

\[
\operatorname{idf}_{BM25}(t)
=
\log\left(
1+\frac{N-df_t+0.5}{df_t+0.5}
\right).
\]

Let \(|d|\) be the number of lexical terms in a chunk and
\(\operatorname{avgdl}\) the mean chunk length. Define

\[
K_d
=
k_1\left(
1-b+b\frac{|d|}{\operatorname{avgdl}}
\right).
\]

The score is

\[
\operatorname{BM25}(q,d)
=
\sum_{t\in\operatorname{unique}(q)}
\operatorname{idf}_{BM25}(t)
\frac{f_{t,d}(k_1+1)}{f_{t,d}+K_d}.
\]

Repeated query terms intentionally contribute once. This policy is explicit
in each result's scoring details. Terms with zero chunk frequency contribute
zero. Configuration validation requires finite \(k_1>0\) and
\(0\leq b\leq1\). Each term contribution, IDF, raw frequency, document
frequency, and length-normalization factor is inspectable.

## 8. Deterministic tie-breaking

Positive-score candidates are ordered by:

1. descending full-precision score;
2. ascending document ID;
3. ascending global chunk ordinal;
4. ascending chunk ID.

Stable ties make fixture expectations, reload comparisons, and citations
repeatable. Rankings do not depend on input file enumeration or dictionary
insertion order.

## 9. Citations

A `Citation` stores identifiers, source name, optional real title, heading
path, exact line range, optional known page range, and exact chunk ID. It never
infers an author, page, or section number.

Page-aware examples are formatted as `[Title, p. 3]` or
`[Title, pp. 3–4]`. When pages are unknown, text/Markdown results use
`[source.md, lines 42–58, § Heading › Child]`. Structured fields and the
derived display string are both validated during round-trip loading.

Every chunk retains half-open source character offsets and an exact text
slice. The line or page location is therefore metadata about a passage whose
contents can be independently checked against the serialized document.

## 10. Retrieval evaluation

For returned IDs \(R_k\), relevant IDs \(G\), and a positive cutoff \(k\):

\[
P@k=\frac{|R_k\cap G|}{k},
\qquad
R@k=\frac{|R_k\cap G|}{|G|}.
\]

Precision deliberately uses \(k\), even when fewer than \(k\) results are
returned. Recall is rejected when \(G\) is empty because it would be
undefined. If the first relevant result appears at rank \(r\),

\[
RR=\frac{1}{r},
\]

and \(RR=0\) if none is retrieved. Hit Rate@\(k\) is one when
\(R_k\cap G\neq\varnothing\), otherwise zero. MRR is the arithmetic mean of
per-query reciprocal ranks. Aggregate metrics are means over validated
queries with exact, unique chunk IDs.

## 11. Limitations of lexical retrieval

The index cannot bridge synonyms or paraphrases that share no lexical term. It
does not understand that two formula spellings are equivalent, and punctuation
tokenization can discard useful mathematical symbols. Code identifiers are
preserved but not decomposed. Results remain sensitive to section and chunk
boundaries. There is no stemming, lemmatization, stop-word policy, phrase
score, semantic embedding, reranker, or answer generator.

The PDF adapter accepts already extracted page strings. It does not parse PDF
bytes, perform OCR, repair reading order, infer headings from layout, or claim
that supplied extraction is accurate. Non-empty page strings are joined with a
single canonical newline and retain their supplied page numbers. Exactly empty
page strings add no artificial source characters or termless section; their
page numbers remain explicit in `metadata["inferred"]["empty_pages"]`.

## 12. Complexity

For \(M\) source characters and \(L\) emitted lexical terms, ingestion and
chunk construction are linear in the inspected text apart from small boundary
search windows. Counting term frequencies is \(O(L)\); sorting a vocabulary of
size \(V\) costs \(O(V\log V)\). The transparent JSON snapshot stores source
text, chunks, and sparse frequency maps, so overlap intentionally duplicates
some text.

The current reference search scans every chunk. If the query has \(Q\) unique
terms and there are \(N\) chunks, the dominant work is proportional to the
stored sparse rows, approximately \(O(NQ)\) for BM25 membership/contribution
checks plus result sorting. This is appropriate for small local corpora but is
not an inverted-index performance claim.

## 13. Source mapping

| Concept | Implementation | Direct tests |
|---|---|---|
| canonical state, IDs, models, citations | `retrieval/documents.py` | `test_retrieval_ingestion.py`, `test_retrieval_index.py` |
| text/Markdown/PDF-derived ingestion | `retrieval/ingestion.py` | `test_retrieval_ingestion.py` |
| exact section-local chunking and coverage | `retrieval/chunking.py` | `test_retrieval_chunking.py` |
| lexical policy and source spans | `retrieval/text.py` | `test_retrieval_text.py` |
| TF-IDF equations | `retrieval/tfidf.py` | `test_retrieval_scoring.py` |
| BM25 equations | `retrieval/bm25.py` | `test_retrieval_scoring.py` |
| snapshot state, filters, ranking, explanations | `retrieval/index.py` | `test_retrieval_index.py` |
| evaluation formulas | `retrieval/metrics.py` | `test_retrieval_metrics.py` |
| build/inspect/search interface | `retrieval/search.py` | `test_retrieval_cli.py` |
| full fixture behavior | `experiments/*.py` | `test_retrieval_experiments.py` |

The independently calculated formula fixtures and exact save/load comparisons
are correctness checks. The five-query authored corpus is too small to support
a general retrieval-quality claim.
