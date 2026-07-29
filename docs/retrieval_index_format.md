# Retrieval Index Format

Milestone 10 stores one immutable lexical or semantically enriched index as
deterministic UTF-8 JSON. The current `index_format_version` is `2`, created
by package version `1.0.0`. Pickle and NumPy object arrays are never used.

## Top-level schema

Every top-level key is required and unexpected keys are rejected:

| Key | Meaning |
|---|---|
| `index_format_version` | integer schema version |
| `package_version` | exact creating LocalML Scholar version |
| `index_type` | `immutable_retrieval_snapshot` |
| `index_config` | duplicate-content policy |
| `chunking_config` | complete deterministic boundary configuration |
| `lexical_config` | complete retrieval term-normalization policy |
| `bm25_config` | finite `k1` and `b` values |
| `documents` | canonical documents, sections, metadata, and exact source text |
| `chunks` | exact source slices and provenance |
| `vocabulary` | sorted unique lexical terms |
| `term_frequencies` | one sorted sparse term-count object per chunk |
| `document_frequencies` | chunk-frequency value for every vocabulary term |
| `average_chunk_length` | mean lexical term count |
| `corpus_sha256` | identity of ordered source/content records |
| `semantic_index` | `null` or the complete validated LSA snapshot |
| `index_sha256` | identity of every other top-level component |

The file embeds exact source and chunk text. Search therefore does not require
the original source file to remain available. Overlap duplicates some source
text by design; this format prioritizes inspection and validation over compact
storage.

## Deterministic encoding and hashes

State identities use canonical JSON with sorted object keys, compact
separators, UTF-8 characters preserved, and non-finite numbers rejected.
Identity input must already consist of JSON-native arrays, string-keyed
objects, and scalar values; lossy tuple or mapping-key conversion is rejected.

The saved file uses sorted keys and indentation for inspection. Formatting
whitespace is not part of `index_sha256`; the canonical state is. Two builds
with identical logical sources, text, configuration, and package version
produce the same state and file bytes.

`corpus_sha256` covers the deterministically ordered document ID, logical
source, and content hash records. `index_sha256` covers the complete state
except itself, including configuration, embedded documents/chunks, and all
lexical statistics and optional semantic state.

## Semantic state

An enriched version 2 snapshot stores:

- semantic format version and method (`lsa`);
- immutable semantic configuration and configuration hash;
- exact lexical vocabulary and vocabulary hash;
- chunk IDs in lexical index order;
- retained right singular vectors;
- chunk embeddings and pre-normalization norms;
- retained singular values;
- effective rank, matrix shape, reconstruction error, and squared
  singular-value fraction;
- zero TF-IDF row indices;
- canonical sign-pivot indices and tie policy;
- semantic component hash.

All arrays are JSON number arrays interpreted and validated as float64. Shapes,
finite values, normalization, chunk order, vocabulary alignment, sign pivots,
configuration hash, semantic hash, and a fresh deterministic LSA rebuild are
checked during load. JSON avoids pickle entirely; `allow_pickle` is therefore
not applicable.

## Atomic save

`RetrievalIndex.save` requires a `.json` destination. It writes a temporary
file in the destination directory, flushes and synchronizes it, then replaces
the destination atomically. A failed write does not intentionally expose a
partially written target.

Generated indexes belong under `outputs/` and are ignored by Git.

## Transactional load

`RetrievalIndex.load`:

1. decodes strict UTF-8 and parses JSON;
2. validates exact top-level keys, format, package, and index type;
3. constructs and validates every configuration, document, section, and chunk
   in local state;
4. validates vocabulary order, sparse-row counts, term counts, and document
   frequencies;
5. verifies corpus and index hashes;
6. rebuilds the index from embedded documents and configuration;
7. rebuilds any semantic factors and requires exact semantic state equality;
8. returns the new object only after all checks pass.

This process does not mutate an existing index. Unsupported versions, missing
components, unexpected components, invalid types/ranges, tampered source text,
and inconsistent derived statistics are errors.

## Snapshot and change policy

Indexes are immutable. There is no in-place document addition or deletion
protocol. `RetrievalIndex.change_reasons` compares a candidate source/config
set with an existing snapshot and reports:

- added or removed logical sources;
- content changes at the same logical source;
- source moves detectable as remove plus add;
- chunking configuration changes;
- lexical configuration changes;
- BM25 configuration changes;
- `unchanged` when none applies.

Any reported source change requires a complete deterministic rebuild.
`RetrievalIndex.enrich_semantic` is a distinct operation: it validates the
existing lexical state, computes LSA without re-ingesting sources, preserves
all source/chunk IDs and lexical rankings, and returns a new version 2 snapshot
with a different index hash. A conflicting request against an already enriched
snapshot raises instead of silently replacing its semantic configuration.

## Legacy compatibility

Recognized format version 1 lexical snapshots created by package version 0.8.0
or 0.9.0 load as lexical-only indexes after complete hash/statistic
validation. TF-IDF and BM25 remain available. Semantic or hybrid search raises
until the caller explicitly enriches and saves a new output file; the original
legacy file is never overwritten implicitly.

Incremental indexing, compact sparse persistence, and approximate search are
deferred until profiling demonstrates a need.
