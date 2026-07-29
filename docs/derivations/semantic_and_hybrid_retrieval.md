# Semantic and Hybrid Retrieval

This document describes the deterministic retrieval extension in package
version 1.0.0. It is a mathematical implementation map, not a claim that
latent similarity proves relevance.

## 1. Lexical versus semantic retrieval

TF-IDF and BM25 score terms that occur in both the query and a chunk. This is
precise and interpretable, but exact vocabulary mismatch can hide relevant
passages. Latent semantic analysis (LSA) instead represents terms and chunks
through shared directions in the corpus term matrix. Terms that occur in
similar chunk contexts can influence similar latent coordinates.

LSA is distributional and linear. It is not a neural embedding model, does not
understand language, and can reinforce misleading corpus correlations.

## 2. Baseline selection

The repository has independently implemented lexical statistics and NumPy
linear algebra, but it has no trained dual encoder or licensed local vector
artifact. Training a credible semantic encoder on the project CPU fixtures
would overstate what those fixtures support. Milestone 10 therefore selects
TF-IDF LSA:

- no external model or download,
- no vector database,
- exact deterministic full-corpus scoring,
- inspectable term and latent contributions,
- direct reuse of the immutable lexical vocabulary and chunk order.

Modern neural embeddings, approximate search, and cross-encoder reranking are
deliberately deferred.

## 3. Term-chunk matrix

For \(N\) chunks and an ordered vocabulary of \(M\) terms, the implementation
constructs

\[
A\in\mathbb{R}^{N\times M}.
\]

The entry for chunk \(d\) and term \(t\) is the existing sublinear TF-IDF
weight:

\[
A_{d,t}
=
\left(1+\log \operatorname{tf}_{d,t}\right)
\left[
\log\left(\frac{N+1}{\operatorname{df}_t+1}\right)+1
\right]
\]

when the count is positive, and zero otherwise. Matrix columns follow the
lexical index vocabulary exactly; rows follow immutable chunk order exactly.
`build_tfidf_matrix` performs this conversion without mutating lexical state.

## 4. Singular value decomposition

NumPy computes the full reduced decomposition

\[
A=U\Sigma V^\top.
\]

Before truncation, the implementation verifies

\[
\frac{\lVert A-U\Sigma V^\top\rVert_F}
{\max(1,\lVert A\rVert_F)}
\leq \tau.
\]

The effective rank counts singular values larger than both a configured
minimum and the standard shape-scaled float64 machine threshold. A requested
dimension \(k\) above that rank raises; the implementation never silently
reduces \(k\).

## 5. Truncated latent space

Retaining the first \(k\) components gives

\[
A_k=U_k\Sigma_kV_k^\top.
\]

Chunk coordinates use

\[
E_d=U_k\Sigma_k.
\]

If normalization is enabled, each nonzero row is divided by its Euclidean
norm. The snapshot records:

- original matrix shape,
- effective rank,
- selected dimension,
- retained singular values,
- Frobenius reconstruction error \(\lVert A-A_k\rVert_F\),
- the retained fraction
  \[
  \frac{\sum_{i=1}^{k}\sigma_i^2}
       {\sum_i\sigma_i^2},
  \]
- zero row indices,
- numerical tolerance.

The last quantity is named the *explained squared singular-value fraction*,
not variance explained.

## 6. Query projection

A query is tokenized with the index lexical configuration and weighted with
the same TF-IDF formula. Let

\[
q\in\mathbb{R}^{M}
\]

be that vocabulary-aligned row vector. Its latent representation is

\[
E_q=qV_k.
\]

Known and out-of-vocabulary query terms are reported separately. Every known
term's contribution before summation is

\[
q_t(V_k)_{:,t}.
\]

The implementation exposes the norm of this vector but does not assign a
human label to any latent coordinate. A query with no in-vocabulary term
produces the zero vector and no semantic results.

## 7. Exact semantic cosine

For a query vector \(q_\ell\) and chunk vector \(d_\ell\),

\[
s(q,d)
=
\frac{q_\ell^\top d_\ell}
{\lVert q_\ell\rVert_2\lVert d_\ell\rVert_2}.
\]

The search scans every filtered chunk. Zero norms produce score zero.
Non-finite values raise. Positive results sort by:

1. score descending,
2. document ID,
3. chunk ordinal,
4. chunk ID.

The explanation includes the cosine numerator, both norms, contributing query
terms, out-of-vocabulary terms, and largest absolute coordinate products.
Negative and zero cosine results are not returned. Similarity is not proof of
relevance.

The result field `matched_terms` remains an exact lexical intersection between
the normalized query and the returned chunk. `semantic_query_terms` separately
records the in-vocabulary query terms that contributed to the latent query
projection. Keeping these fields distinct prevents a latent match from being
misreported as literal term overlap.

## 8. SVD sign ambiguity

For any component, simultaneously replacing

\[
U_{:,i}\leftarrow-U_{:,i},
\qquad
V^\top_{i,:}\leftarrow-V^\top_{i,:}
\]

does not change the reconstruction. To make serialized state reproducible,
`canonicalize_svd_signs` finds the first largest-absolute entry in each
retained row of \(V^\top\). That pivot is required to be nonnegative. When it
is negative, both matching factors are multiplied by \(-1\).

The lowest term index resolves absolute-value ties. Tests verify unchanged
reconstruction and identical canonical state for sign-flipped factors.

## 9. Hybrid retrieval

Hybrid candidate generation retrieves bounded lexical and semantic lists,
forms their chunk-ID union, and keeps the component score and rank for every
candidate. Metadata filters are applied inside both component searches before
fusion.

### Weighted fusion

Each non-negative component list is normalized by its largest positive score:

\[
\tilde{s}(d)=
\begin{cases}
s(d)/\max_c s(c),&\max_c s(c)>0,\\
0,&\text{otherwise}.
\end{cases}
\]

The combined score is

\[
s_{\mathrm{weighted}}(d)
=
\alpha\tilde{s}_{\mathrm{lexical}}(d)
+(1-\alpha)\tilde{s}_{\mathrm{semantic}}(d),
\qquad 0\leq\alpha\leq1.
\]

Raw BM25 and cosine values are never added.

### Reciprocal rank fusion

For component methods \(m\),

\[
s_{\mathrm{RRF}}(d)
=
\sum_m\frac{w_m}{K+r_m(d)}.
\]

A missing candidate contributes zero for that method. The snapshot records
the rank constant, weights, candidate depths, ranks, component scores, and
each reciprocal contribution.

## 10. Deterministic reranking

The reranker is a manually weighted linear heuristic, not a learned
cross-encoder:

\[
R(d,q)=\sum_j w_jf_j(d,q).
\]

Current normalized features are:

- lexical component score,
- semantic component score,
- exact normalized phrase match,
- heading-term coverage,
- query-term coverage,
- IDF-weighted rare-term coverage,
- numerical-token overlap,
- code-identifier overlap,
- bounded chunk-length penalty.

All feature values, weights, signed contributions, any clamp adjustment, and
the final score are returned. Their sum equals the final score. No metadata
preference is inferred from query prose.

## 11. Redundancy

For two chunks in the same document,

\[
\operatorname{overlap}(a,b)=
\frac{
\max\left(0,\min(e_a,e_b)-\max(s_a,s_b)\right)
}{
\min(e_a-s_a,e_b-s_b)
}.
\]

Different documents have zero source-range overlap. During deterministic
selection, a candidate whose maximum overlap with an already selected chunk
meets the configured threshold receives a linear penalty. The explanation
records the measured overlap and whether the penalty was applied. Distinct
non-overlapping passages are unchanged.

## 12. Retrieval evaluation

The evaluation uses exact chunk IDs.

Average Precision is

\[
\operatorname{AP}
=
\frac{1}{|R|}
\sum_{r:\,d_r\in R}P@r.
\]

With authored nonnegative relevance grades, discounted cumulative gain is

\[
\operatorname{DCG}@k
=
\sum_{r=1}^{k}
\frac{2^{\operatorname{rel}_r}-1}{\log_2(r+1)}
\]

and

\[
\operatorname{nDCG}@k
=
\frac{\operatorname{DCG}@k}
{\operatorname{IDCG}@k}.
\]

The code retains Precision@k, Recall@k, reciprocal rank, and hit rate, and
adds AP, nDCG, mean relevant rank, and no-relevant-result rate. Aggregates are
macro means over queries. Category metrics are shown even when semantic
retrieval is worse.

The committed fixture is small and project-authored. Its results validate
formulas, serialization, and controlled failure modes; they do not establish
general retrieval quality.

## 13. Citation preservation

Semantic vectors never contain reconstructed source metadata. Every result is
created from an original `Chunk` in the lexical snapshot. The result citation
uses that chunk's unchanged document ID, chunk ID, heading path, character
range, line range, and known page range. Fusion and reranking operate only on
chunk identities and replace scores/ranks, not source identity.

Semantic enrichment changes the overall index hash because it adds validated
state. It does not change the corpus hash, document IDs, chunk IDs, lexical
statistics, or lexical rankings.

## 14. Limitations

- Dense SVD costs scale poorly with corpus and vocabulary size.
- LSA is linear, corpus-dependent, and sensitive to chunking.
- Polysemy and rare synonyms may collapse into misleading directions.
- Query projection is weak when query terms are absent from the vocabulary.
- Small corpora can produce unstable or unhelpful latent neighborhoods.
- Exact scanning has no approximate nearest-neighbor acceleration.
- There is no neural embedding model, vector database, or cross-encoder.
- Positive cosine similarity does not guarantee relevance.
- Relevance does not guarantee factual support.
- The deterministic reranker uses manually chosen, untrained weights.

## 15. Source mapping

| Mathematics | Source | Primary tests |
|---|---|---|
| TF-IDF matrix, SVD, sign policy | `retrieval/semantic.py` | `test_semantic_retrieval.py` |
| query projection and exact cosine | `retrieval/semantic.py`, `retrieval/index.py` | `test_semantic_retrieval.py` |
| score normalization and fusion | `retrieval/hybrid.py` | `test_hybrid_retrieval.py` |
| feature score and redundancy | `retrieval/hybrid.py`, `retrieval/index.py` | `test_hybrid_retrieval.py` |
| AP, nDCG, category metrics | `retrieval/metrics.py` | `test_retrieval_advanced_metrics.py` |
| semantic persistence/enrichment | `retrieval/index.py` | `test_semantic_retrieval.py` |
| answering integration | `answering/evidence.py`, `answering/pipeline.py` | `test_semantic_answering_integration.py` |
| controlled experiments | `experiments/evaluate_retrieval_ablation.py` | `test_semantic_retrieval_experiments.py` |
