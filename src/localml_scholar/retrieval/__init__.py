"""Deterministic local document ingestion and lexical retrieval."""

from localml_scholar.retrieval.bm25 import BM25Config
from localml_scholar.retrieval.chunking import (
    ChunkingConfig,
    chunk_document,
    validate_chunk_coverage,
)
from localml_scholar.retrieval.documents import (
    Chunk,
    Citation,
    Document,
    PageText,
    Section,
)
from localml_scholar.retrieval.index import (
    IndexConfig,
    RetrievalIndex,
    SearchFilters,
    SearchQuery,
    SearchResult,
    highlight_matches,
)
from localml_scholar.retrieval.ingestion import (
    ingest_file,
    ingest_files,
    ingest_markdown,
    ingest_pdf_text,
    ingest_plain_text,
)
from localml_scholar.retrieval.metrics import (
    RetrievalEvaluation,
    evaluate_rankings,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from localml_scholar.retrieval.text import (
    LexicalTerm,
    LexicalTokenizerConfig,
    lexical_terms,
    tokenize_lexically,
)

__all__ = [
    "BM25Config",
    "Chunk",
    "ChunkingConfig",
    "Citation",
    "Document",
    "evaluate_rankings",
    "highlight_matches",
    "hit_rate_at_k",
    "IndexConfig",
    "ingest_file",
    "ingest_files",
    "ingest_markdown",
    "ingest_pdf_text",
    "ingest_plain_text",
    "LexicalTerm",
    "LexicalTokenizerConfig",
    "lexical_terms",
    "PageText",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "RetrievalEvaluation",
    "RetrievalIndex",
    "SearchFilters",
    "SearchQuery",
    "SearchResult",
    "Section",
    "tokenize_lexically",
    "validate_chunk_coverage",
    "chunk_document",
]
