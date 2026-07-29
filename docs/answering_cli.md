# Grounded answering CLI

Build an index first with the retrieval CLI, then run extractive answering:

```bash
python3 -m localml_scholar.answering.cli \
  --index outputs/local_documents/index.json \
  --question "How does causal masking prevent leakage?" \
  --method extractive \
  --retriever bm25 \
  --top-k 5 \
  --verbose
```

The human view prints the answer, abstention/fallback status, inline citations,
source bibliography, validation summary, raw generation when present, and
optional passages. Use `--json` for the complete structured state and `--save`
for an atomic answer artifact:

```bash
python3 -m localml_scholar.answering.cli \
  --index outputs/local_documents/index.json \
  --question "How does causal masking prevent leakage?" \
  --json \
  --save outputs/local_documents/causal_mask_answer.json
```

`--retriever` accepts `tfidf`, `bm25`, `semantic`, `hybrid`, and
`hybrid_reranked`. Semantic and hybrid methods require an explicitly enriched
index. Hybrid controls mirror the search CLI:

```bash
python3 -m localml_scholar.answering.cli \
  --index outputs/local_documents/index_lsa.json \
  --question "Why can later tokens not affect earlier predictions?" \
  --method extractive \
  --retriever hybrid \
  --lexical-method bm25 \
  --fusion rrf \
  --rerank
```

Normal human output identifies the selected retriever. `--verbose` includes
component scores and reranking explanations; latent details remain absent from
the normal view. JSON and saved artifacts record the full effective retrieval
configuration. A missing semantic index is an error—answer fallback never
means retrieval-method substitution.

Generative modes require an explicit model-only checkpoint that bundles its
matching tokenizer:

```bash
python3 -m localml_scholar.answering.cli \
  --index outputs/local_documents/index.json \
  --question "How does causal masking prevent leakage?" \
  --method generative-with-extractive-fallback \
  --checkpoint outputs/trained_model/final_model.npz \
  --greedy \
  --maximum-new-tokens 64
```

No checkpoint is loaded in extractive mode. A missing generative checkpoint is
an error. `--sample`, `--temperature`, `--sampling-top-k`, and `--seed`
provide deterministic seeded sampling when greedy mode is disabled.

Filters mirror retrieval search: `--document-id`, `--source-name`,
`--media-type`, repeated `--heading-prefix`, `--publication-year`, and
`--collection`. Evidence selection never expands beyond the explicit filtered
retrieval results.

An accepted citation proves that the label maps to an exact passage in the
recorded index. Semantic similarity does not prove relevance, and an accepted
citation does not prove the associated prose is semantically true.
