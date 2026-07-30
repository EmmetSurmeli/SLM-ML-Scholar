"""Immutable deterministic lexical index, search, filtering, and persistence."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from localml_scholar._version import __version__
from localml_scholar.retrieval.bm25 import (
    BM25Config,
    bm25_term_contribution,
)
from localml_scholar.retrieval.chunking import ChunkingConfig, chunk_document
from localml_scholar.retrieval.documents import (
    Chunk,
    Citation,
    Document,
    canonical_json,
)
from localml_scholar.retrieval.hybrid import (
    HybridRetrievalConfig,
    RerankingConfig,
    fuse_rankings,
    reranking_features,
    source_range_overlap,
    weighted_reranking_score,
)
from localml_scholar.retrieval.semantic import (
    SemanticIndex,
    SemanticRetrievalConfig,
    fit_lsa,
)
from localml_scholar.retrieval.text import (
    LexicalTokenizerConfig,
    lexical_terms,
    tokenize_lexically,
)
from localml_scholar.retrieval.tfidf import (
    cosine_score,
    smooth_inverse_document_frequency,
    sparse_tfidf_weights,
)
from localml_scholar.serialization import atomic_write_text

INDEX_FORMAT_VERSION = 2
LEGACY_INDEX_FORMAT_VERSION = 1
_RETRIEVAL_METHODS = {
    "tfidf",
    "bm25",
    "semantic",
    "hybrid",
    "hybrid_reranked",
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class IndexConfig:
    """Immutable snapshot and duplicate-content policy."""

    allow_duplicate_content: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.allow_duplicate_content, bool):
            raise TypeError("allow_duplicate_content must be boolean.")

    def to_dict(self) -> dict[str, bool]:
        return {"allow_duplicate_content": self.allow_duplicate_content}

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> IndexConfig:
        if not isinstance(state, dict) or set(state) != {"allow_duplicate_content"}:
            raise ValueError("Index configuration is malformed.")
        return cls(**state)


@dataclass(frozen=True)
class SearchFilters:
    """Explicit metadata filters; no natural-language filter inference."""

    document_id: str | None = None
    source_name: str | None = None
    media_type: str | None = None
    heading_path_prefix: tuple[str, ...] = ()
    publication_year: int | None = None
    logical_collection: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "document_id",
            "source_name",
            "media_type",
            "logical_collection",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be None or a non-empty string.")
        if not isinstance(self.heading_path_prefix, tuple) or not all(
            isinstance(value, str) and value for value in self.heading_path_prefix
        ):
            raise ValueError(
                "heading_path_prefix must be a tuple of non-empty strings."
            )
        if self.publication_year is not None and (
            isinstance(self.publication_year, bool)
            or not isinstance(self.publication_year, int)
        ):
            raise TypeError("publication_year must be None or an integer.")

    def to_dict(self) -> dict[str, Any]:
        state = dict(vars(self))
        state["heading_path_prefix"] = list(self.heading_path_prefix)
        return state


@dataclass(frozen=True)
class SearchQuery:
    """Validated raw query, normalized terms, explicit filters, and result limit."""

    raw_text: str
    normalized_terms: tuple[str, ...]
    top_k: int = 5
    filters: SearchFilters = field(default_factory=SearchFilters)

    def __post_init__(self) -> None:
        if not isinstance(self.raw_text, str) or not self.raw_text.strip():
            raise ValueError("Search query must contain non-whitespace text.")
        if not isinstance(self.normalized_terms, tuple) or not self.normalized_terms:
            raise ValueError("Search query must contain at least one lexical term.")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise TypeError("top_k must be an integer.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not isinstance(self.filters, SearchFilters):
            raise TypeError("filters must be SearchFilters.")

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        config: LexicalTokenizerConfig | None = None,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> SearchQuery:
        return cls(
            raw_text=text,
            normalized_terms=tokenize_lexically(text, config),
            top_k=top_k,
            filters=filters or SearchFilters(),
        )


@dataclass(frozen=True)
class SearchResult:
    """One exact ranked passage and transparent retrieval scoring evidence."""

    rank: int
    score: float
    retrieval_method: str
    chunk_id: str
    document_id: str
    source_name: str
    title: str | None
    authors: tuple[str, ...] | None
    heading_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    start_line: int
    end_line: int
    text: str
    matched_terms: tuple[str, ...]
    semantic_query_terms: tuple[str, ...]
    term_contributions: tuple[dict[str, Any], ...]
    scoring_details: dict[str, Any]
    citation: Citation

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("rank must be an integer.")
        if self.rank <= 0:
            raise ValueError("rank starts at one.")
        if not math.isfinite(self.score) or self.score < 0.0:
            raise ValueError("Search score must be finite and non-negative.")
        if self.retrieval_method not in _RETRIEVAL_METHODS:
            raise ValueError("Unknown retrieval method.")
        if self.authors is not None and (
            not isinstance(self.authors, tuple)
            or not all(isinstance(author, str) and author for author in self.authors)
        ):
            raise ValueError("authors must be None or a tuple of strings.")
        for name in ("matched_terms", "semantic_query_terms"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not all(
                isinstance(term, str) and term for term in value
            ):
                raise ValueError(f"{name} must be a tuple of non-empty strings.")
        if self.citation.chunk_id != self.chunk_id:
            raise ValueError("Citation must link to the exact result chunk.")
        canonical_json(list(self.term_contributions))
        canonical_json(self.scoring_details)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "retrieval_method": self.retrieval_method,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_name": self.source_name,
            "title": self.title,
            "authors": None if self.authors is None else list(self.authors),
            "heading_path": list(self.heading_path),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
            "matched_terms": list(self.matched_terms),
            "semantic_query_terms": list(self.semantic_query_terms),
            "term_contributions": list(self.term_contributions),
            "scoring_details": self.scoring_details,
            "citation": self.citation.to_dict(),
        }


class RetrievalIndex:
    """Validated immutable snapshot searchable without any language model."""

    def __init__(
        self,
        *,
        documents: tuple[Document, ...],
        chunks: tuple[Chunk, ...],
        index_config: IndexConfig,
        chunking_config: ChunkingConfig,
        lexical_config: LexicalTokenizerConfig,
        bm25_config: BM25Config,
        term_frequencies: tuple[dict[str, int], ...],
        document_frequencies: dict[str, int],
        vocabulary: tuple[str, ...],
        corpus_sha256: str,
        index_sha256: str,
        package_version: str = __version__,
        semantic_index: SemanticIndex | None = None,
        index_format_version: int = INDEX_FORMAT_VERSION,
    ) -> None:
        if not documents or not chunks:
            raise ValueError("Retrieval index requires documents and chunks.")
        if len(term_frequencies) != len(chunks):
            raise ValueError("Term-frequency rows must match chunks.")
        if not all(
            isinstance(row, dict)
            and row
            and all(
                isinstance(term, str)
                and term
                and not isinstance(count, bool)
                and isinstance(count, int)
                and count > 0
                for term, count in row.items()
            )
            for row in term_frequencies
        ):
            raise ValueError(
                "Each term-frequency row must map terms to positive integers."
            )
        if tuple(sorted(vocabulary)) != vocabulary or len(set(vocabulary)) != len(
            vocabulary
        ):
            raise ValueError("Vocabulary must be sorted and unique.")
        if not all(isinstance(term, str) and term for term in vocabulary):
            raise ValueError("Vocabulary entries must be non-empty strings.")
        if set(document_frequencies) != set(vocabulary):
            raise ValueError("Document frequencies must cover the exact vocabulary.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= len(chunks)
            for value in document_frequencies.values()
        ):
            raise ValueError("Document frequencies lie outside [1, chunk_count].")
        if any(
            sum(row.values()) != chunk.term_count
            for row, chunk in zip(term_frequencies, chunks, strict=True)
        ):
            raise ValueError("Chunk term counts and frequency rows differ.")
        document_ids = {document.document_id for document in documents}
        if any(chunk.document_id not in document_ids for chunk in chunks):
            raise ValueError("Every chunk must link to an indexed document.")
        document_by_id = {document.document_id: document for document in documents}
        section_links = {
            section.section_id: section.document_id
            for document in documents
            for section in document.sections
        }
        if any(
            section_links.get(chunk.section_id) != chunk.document_id
            or document_by_id[chunk.document_id].text[
                chunk.start_character : chunk.end_character
            ]
            != chunk.text
            for chunk in chunks
        ):
            raise ValueError("Every chunk must match its document and section source.")
        if not _is_sha256(corpus_sha256) or not _is_sha256(index_sha256):
            if index_sha256 != "pending":
                raise ValueError("Corpus and index identities must be SHA-256 digests.")
            if not _is_sha256(corpus_sha256):
                raise ValueError("Corpus identity must be a SHA-256 digest.")
        if not isinstance(package_version, str) or not package_version:
            raise ValueError("package_version must be a non-empty string.")
        if index_format_version not in {
            LEGACY_INDEX_FORMAT_VERSION,
            INDEX_FORMAT_VERSION,
        }:
            raise ValueError("Unsupported retrieval index format version.")
        if semantic_index is not None:
            if not isinstance(semantic_index, SemanticIndex):
                raise TypeError("semantic_index must be SemanticIndex or None.")
            if semantic_index.vocabulary != vocabulary:
                raise ValueError("Semantic and lexical vocabularies do not align.")
            if semantic_index.chunk_ids != tuple(chunk.chunk_id for chunk in chunks):
                raise ValueError("Semantic and lexical chunk order does not align.")
            if index_format_version != INDEX_FORMAT_VERSION:
                raise ValueError(
                    "Legacy lexical indexes cannot contain semantic state."
                )
        self.documents = documents
        self.chunks = chunks
        self.index_config = index_config
        self.chunking_config = chunking_config
        self.lexical_config = lexical_config
        self.bm25_config = bm25_config
        self.term_frequencies = term_frequencies
        self.document_frequencies = document_frequencies
        self.vocabulary = vocabulary
        self.corpus_sha256 = corpus_sha256
        self.index_sha256 = index_sha256
        self.package_version = package_version
        self.semantic_index = semantic_index
        self.index_format_version = index_format_version
        self._document_by_id = {
            document.document_id: document for document in documents
        }
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if len(self._document_by_id) != len(documents):
            raise ValueError("Duplicate document IDs are not allowed.")
        if len(self._chunk_by_id) != len(chunks):
            raise ValueError("Duplicate chunk IDs are not allowed.")
        if not self.average_chunk_length > 0.0:
            raise ValueError("Average chunk length must be positive.")

    @property
    def average_chunk_length(self) -> float:
        return math.fsum(chunk.term_count for chunk in self.chunks) / len(self.chunks)

    @classmethod
    def build(
        cls,
        documents: Sequence[Document],
        *,
        index_config: IndexConfig | None = None,
        chunking_config: ChunkingConfig | None = None,
        lexical_config: LexicalTokenizerConfig | None = None,
        bm25_config: BM25Config | None = None,
    ) -> RetrievalIndex:
        """Build a deterministic full immutable snapshot."""
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise TypeError("documents must be a sequence of Document objects.")
        ordered = tuple(sorted(documents, key=lambda document: document.document_id))
        if not ordered or not all(
            isinstance(document, Document) for document in ordered
        ):
            raise ValueError("At least one valid Document is required.")
        config = index_config or IndexConfig()
        chunk_config = chunking_config or ChunkingConfig()
        lexical = lexical_config or LexicalTokenizerConfig()
        bm25 = bm25_config or BM25Config()
        source_paths = [document.source_path for document in ordered]
        if len(source_paths) != len(set(source_paths)):
            raise ValueError("Duplicate logical source paths are not allowed.")
        content_hashes = [document.content_sha256 for document in ordered]
        if not config.allow_duplicate_content and len(content_hashes) != len(
            set(content_hashes)
        ):
            raise ValueError(
                "Duplicate document content requires allow_duplicate_content=True."
            )
        chunks: list[Chunk] = []
        for document in ordered:
            document_chunks = chunk_document(document, chunk_config, lexical)
            base_ordinal = len(chunks)
            for local_chunk in document_chunks:
                state = local_chunk.to_dict()
                state["ordinal"] = base_ordinal + local_chunk.ordinal
                chunks.append(Chunk.from_dict(state))
        frequencies: list[dict[str, int]] = []
        document_frequencies: Counter[str] = Counter()
        for chunk in chunks:
            row = dict(sorted(Counter(tokenize_lexically(chunk.text, lexical)).items()))
            if not row:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} has no lexical terms; "
                    "it cannot be indexed."
                )
            frequencies.append(row)
            document_frequencies.update(row.keys())
        vocabulary = tuple(sorted(document_frequencies))
        corpus_state = [
            {
                "document_id": document.document_id,
                "source_path": document.source_path,
                "content_sha256": document.content_sha256,
            }
            for document in ordered
        ]
        corpus_hash = hashlib.sha256(
            canonical_json(corpus_state).encode("utf-8")
        ).hexdigest()
        provisional = cls(
            documents=ordered,
            chunks=tuple(chunks),
            index_config=config,
            chunking_config=chunk_config,
            lexical_config=lexical,
            bm25_config=bm25,
            term_frequencies=tuple(frequencies),
            document_frequencies=dict(sorted(document_frequencies.items())),
            vocabulary=vocabulary,
            corpus_sha256=corpus_hash,
            index_sha256="pending",
            index_format_version=INDEX_FORMAT_VERSION,
        )
        index_hash = provisional._calculated_index_hash()
        return cls(
            documents=provisional.documents,
            chunks=provisional.chunks,
            index_config=config,
            chunking_config=chunk_config,
            lexical_config=lexical,
            bm25_config=bm25,
            term_frequencies=provisional.term_frequencies,
            document_frequencies=provisional.document_frequencies,
            vocabulary=vocabulary,
            corpus_sha256=corpus_hash,
            index_sha256=index_hash,
            index_format_version=INDEX_FORMAT_VERSION,
        )

    def _state_without_index_hash(self) -> dict[str, Any]:
        state = {
            "index_format_version": self.index_format_version,
            "package_version": self.package_version,
            "index_type": (
                "immutable_lexical_snapshot"
                if self.index_format_version == LEGACY_INDEX_FORMAT_VERSION
                else "immutable_retrieval_snapshot"
            ),
            "index_config": self.index_config.to_dict(),
            "chunking_config": self.chunking_config.to_dict(),
            "lexical_config": self.lexical_config.to_dict(),
            "bm25_config": self.bm25_config.to_dict(),
            "documents": [document.to_dict() for document in self.documents],
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "vocabulary": list(self.vocabulary),
            "term_frequencies": list(self.term_frequencies),
            "document_frequencies": self.document_frequencies,
            "average_chunk_length": self.average_chunk_length,
            "corpus_sha256": self.corpus_sha256,
        }
        if self.index_format_version == INDEX_FORMAT_VERSION:
            state["semantic_index"] = (
                None if self.semantic_index is None else self.semantic_index.to_dict()
            )
        return state

    def _calculated_index_hash(self) -> str:
        return hashlib.sha256(
            canonical_json(self._state_without_index_hash()).encode("utf-8")
        ).hexdigest()

    def state_dict(self) -> dict[str, Any]:
        state = self._state_without_index_hash()
        state["index_sha256"] = self.index_sha256
        return state

    def save(self, path: str | Path) -> Path:
        """Atomically save one deterministic human-inspectable index JSON."""
        destination = Path(path)
        if destination.suffix.casefold() != ".json":
            raise ValueError("Retrieval index path must end with .json.")
        payload = json.dumps(
            self.state_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        return atomic_write_text(destination, payload + "\n")

    @classmethod
    def load(cls, path: str | Path) -> RetrievalIndex:
        """Transactionally reconstruct and fully validate an index snapshot."""
        source = Path(path)
        try:
            state = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Retrieval index does not exist: {source}"
            ) from None
        except UnicodeDecodeError as error:
            raise ValueError("Retrieval index is not valid UTF-8.") from error
        except json.JSONDecodeError as error:
            raise ValueError("Retrieval index is not valid JSON.") from error
        return cls.from_state_dict(state)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> RetrievalIndex:
        common = {
            "index_format_version",
            "package_version",
            "index_type",
            "index_config",
            "chunking_config",
            "lexical_config",
            "bm25_config",
            "documents",
            "chunks",
            "vocabulary",
            "term_frequencies",
            "document_frequencies",
            "average_chunk_length",
            "corpus_sha256",
            "index_sha256",
        }
        if not isinstance(state, Mapping):
            raise ValueError("Retrieval index state must be a mapping.")
        version = state.get("index_format_version")
        expected = (
            common
            if version == LEGACY_INDEX_FORMAT_VERSION
            else common | {"semantic_index"}
        )
        if set(state) != expected:
            raise ValueError("Retrieval index state keys are malformed.")
        values = dict(state)
        if version not in {LEGACY_INDEX_FORMAT_VERSION, INDEX_FORMAT_VERSION}:
            raise ValueError("Unsupported retrieval index format version.")
        expected_type = (
            "immutable_lexical_snapshot"
            if version == LEGACY_INDEX_FORMAT_VERSION
            else "immutable_retrieval_snapshot"
        )
        if values["index_type"] != expected_type:
            raise ValueError("Retrieval index type is incompatible.")
        if version == LEGACY_INDEX_FORMAT_VERSION:
            if values["package_version"] not in {"0.8.0", "0.9.0"}:
                raise ValueError(
                    "Legacy retrieval index package version is incompatible."
                )
        elif values["package_version"] not in {"1.0.0", __version__}:
            raise ValueError("Retrieval index package version is incompatible.")
        for name in ("documents", "chunks", "vocabulary", "term_frequencies"):
            if not isinstance(values[name], list):
                raise ValueError(f"Retrieval index {name} must be a list.")
        if not all(isinstance(row, dict) for row in values["term_frequencies"]):
            raise ValueError("Retrieval term-frequency rows must be objects.")
        if not all(
            isinstance(term, str)
            and term
            and not isinstance(count, bool)
            and isinstance(count, int)
            and count > 0
            for row in values["term_frequencies"]
            for term, count in row.items()
        ):
            raise ValueError(
                "Retrieval term-frequency rows contain invalid terms or counts."
            )
        if not isinstance(values["document_frequencies"], dict):
            raise ValueError("Retrieval document frequencies must be an object.")
        if not all(
            isinstance(term, str)
            and term
            and not isinstance(count, bool)
            and isinstance(count, int)
            for term, count in values["document_frequencies"].items()
        ):
            raise ValueError(
                "Retrieval document frequencies contain invalid terms or counts."
            )
        if not all(isinstance(term, str) and term for term in values["vocabulary"]):
            raise ValueError("Retrieval vocabulary entries must be strings.")
        semantic_state = values.get("semantic_index")
        if semantic_state is not None and not isinstance(semantic_state, Mapping):
            raise ValueError("semantic_index must be null or an object.")
        semantic_index = (
            None if semantic_state is None else SemanticIndex.from_dict(semantic_state)
        )
        index = cls(
            documents=tuple(Document.from_dict(item) for item in values["documents"]),
            chunks=tuple(Chunk.from_dict(item) for item in values["chunks"]),
            index_config=IndexConfig.from_dict(values["index_config"]),
            chunking_config=ChunkingConfig.from_dict(values["chunking_config"]),
            lexical_config=LexicalTokenizerConfig.from_dict(values["lexical_config"]),
            bm25_config=BM25Config.from_dict(values["bm25_config"]),
            term_frequencies=tuple(
                dict(sorted(row.items())) for row in values["term_frequencies"]
            ),
            document_frequencies=dict(sorted(values["document_frequencies"].items())),
            vocabulary=tuple(values["vocabulary"]),
            corpus_sha256=values["corpus_sha256"],
            index_sha256=values["index_sha256"],
            package_version=values["package_version"],
            semantic_index=semantic_index,
            index_format_version=version,
        )
        if values["average_chunk_length"] != index.average_chunk_length:
            raise ValueError("Serialized average chunk length is inconsistent.")
        if index._calculated_index_hash() != index.index_sha256:
            raise ValueError("Retrieval index hash is inconsistent.")
        rebuilt = cls.build(
            index.documents,
            index_config=index.index_config,
            chunking_config=index.chunking_config,
            lexical_config=index.lexical_config,
            bm25_config=index.bm25_config,
        )
        lexical_attributes = (
            "documents",
            "chunks",
            "index_config",
            "chunking_config",
            "lexical_config",
            "bm25_config",
            "term_frequencies",
            "document_frequencies",
            "vocabulary",
            "corpus_sha256",
        )
        if any(
            getattr(rebuilt, name) != getattr(index, name)
            for name in lexical_attributes
        ):
            raise ValueError("Retrieval index statistics do not reconstruct exactly.")
        if semantic_index is not None:
            rebuilt_semantic = fit_lsa(
                term_frequencies=index.term_frequencies,
                document_frequencies=index.document_frequencies,
                vocabulary=index.vocabulary,
                chunk_ids=tuple(chunk.chunk_id for chunk in index.chunks),
                config=semantic_index.config,
            )
            if rebuilt_semantic.to_dict() != semantic_index.to_dict():
                raise ValueError(
                    "Semantic index factors do not reconstruct deterministically."
                )
        return index

    def enrich_semantic(
        self,
        config: SemanticRetrievalConfig | None = None,
    ) -> RetrievalIndex:
        """Return a new snapshot with deterministic LSA state and unchanged chunks."""
        if self.semantic_index is not None and config is None:
            return self
        resolved = config or SemanticRetrievalConfig()
        if not isinstance(resolved, SemanticRetrievalConfig):
            raise TypeError("config must be SemanticRetrievalConfig.")
        if self.semantic_index is not None:
            if self.semantic_index.config != resolved:
                raise ValueError(
                    "Semantic enrichment configuration does not match the "
                    "existing semantic state; enrich the original lexical "
                    "snapshot to change configuration."
                )
            return self
        semantic_index = fit_lsa(
            term_frequencies=self.term_frequencies,
            document_frequencies=self.document_frequencies,
            vocabulary=self.vocabulary,
            chunk_ids=tuple(chunk.chunk_id for chunk in self.chunks),
            config=resolved,
        )
        provisional = RetrievalIndex(
            documents=self.documents,
            chunks=self.chunks,
            index_config=self.index_config,
            chunking_config=self.chunking_config,
            lexical_config=self.lexical_config,
            bm25_config=self.bm25_config,
            term_frequencies=self.term_frequencies,
            document_frequencies=self.document_frequencies,
            vocabulary=self.vocabulary,
            corpus_sha256=self.corpus_sha256,
            index_sha256="pending",
            package_version=__version__,
            semantic_index=semantic_index,
            index_format_version=INDEX_FORMAT_VERSION,
        )
        return RetrievalIndex(
            documents=provisional.documents,
            chunks=provisional.chunks,
            index_config=provisional.index_config,
            chunking_config=provisional.chunking_config,
            lexical_config=provisional.lexical_config,
            bm25_config=provisional.bm25_config,
            term_frequencies=provisional.term_frequencies,
            document_frequencies=provisional.document_frequencies,
            vocabulary=provisional.vocabulary,
            corpus_sha256=provisional.corpus_sha256,
            index_sha256=provisional._calculated_index_hash(),
            package_version=__version__,
            semantic_index=semantic_index,
            index_format_version=INDEX_FORMAT_VERSION,
        )

    def _matches_filters(
        self,
        chunk: Chunk,
        document: Document,
        filters: SearchFilters,
    ) -> bool:
        user_metadata = document.metadata.get("user", {})
        if (
            filters.document_id is not None
            and document.document_id != filters.document_id
        ):
            return False
        if (
            filters.source_name is not None
            and document.source_name != filters.source_name
        ):
            return False
        if filters.media_type is not None and document.media_type != filters.media_type:
            return False
        prefix = filters.heading_path_prefix
        if prefix and chunk.heading_path[: len(prefix)] != prefix:
            return False
        if (
            filters.publication_year is not None
            and user_metadata.get("publication_year") != filters.publication_year
        ):
            return False
        return not (
            filters.logical_collection is not None
            and user_metadata.get("logical_collection") != filters.logical_collection
        )

    def search(
        self,
        query: str | SearchQuery,
        *,
        method: str = "bm25",
        top_k: int = 5,
        filters: SearchFilters | None = None,
        hybrid_config: HybridRetrievalConfig | None = None,
        reranking_config: RerankingConfig | None = None,
    ) -> tuple[SearchResult, ...]:
        """Return ranked source passages only; this method never generates prose."""
        if method not in _RETRIEVAL_METHODS:
            raise ValueError(
                "method must be tfidf, bm25, semantic, hybrid, or hybrid_reranked."
            )
        if isinstance(query, str):
            resolved = SearchQuery.from_text(
                query,
                config=self.lexical_config,
                top_k=top_k,
                filters=filters,
            )
        elif isinstance(query, SearchQuery):
            if filters is not None or top_k != 5:
                raise ValueError(
                    "filters/top_k cannot override a constructed SearchQuery."
                )
            resolved = query
        else:
            raise TypeError("query must be text or SearchQuery.")
        if method in {"tfidf", "bm25"}:
            if hybrid_config is not None or reranking_config is not None:
                raise ValueError(
                    "Hybrid and reranking configuration require a hybrid method."
                )
            return self._search_lexical(resolved, method)
        if method == "semantic":
            if hybrid_config is not None or reranking_config is not None:
                raise ValueError(
                    "Hybrid and reranking configuration do not apply to semantic "
                    "search."
                )
            return self._search_semantic(resolved)
        resolved_hybrid = hybrid_config or HybridRetrievalConfig()
        if not isinstance(resolved_hybrid, HybridRetrievalConfig):
            raise TypeError("hybrid_config must be HybridRetrievalConfig.")
        if method == "hybrid" and reranking_config is not None:
            raise ValueError("reranking_config requires method='hybrid_reranked'.")
        resolved_reranking = (
            None if method == "hybrid" else reranking_config or RerankingConfig()
        )
        return self._search_hybrid(
            resolved,
            method=method,
            hybrid_config=resolved_hybrid,
            reranking_config=resolved_reranking,
        )

    def _make_result(
        self,
        *,
        rank: int,
        score: float,
        method: str,
        chunk: Chunk,
        document: Document,
        matched_terms: tuple[str, ...],
        semantic_query_terms: tuple[str, ...] = (),
        contributions: tuple[dict[str, Any], ...],
        details: dict[str, Any],
    ) -> SearchResult:
        user_metadata = document.metadata.get("user", {})
        authors_value = user_metadata.get("authors")
        authors = (
            tuple(authors_value)
            if isinstance(authors_value, list)
            and all(isinstance(author, str) and author for author in authors_value)
            else None
        )
        citation = Citation(
            document_id=document.document_id,
            source_name=document.source_name,
            title=document.title,
            heading_path=chunk.heading_path,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_id=chunk.chunk_id,
        )
        return SearchResult(
            rank=rank,
            score=float(score),
            retrieval_method=method,
            chunk_id=chunk.chunk_id,
            document_id=document.document_id,
            source_name=document.source_name,
            title=document.title,
            authors=authors,
            heading_path=chunk.heading_path,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            text=chunk.text,
            matched_terms=matched_terms,
            semantic_query_terms=semantic_query_terms,
            term_contributions=contributions,
            scoring_details=details,
            citation=citation,
        )

    def _search_lexical(
        self,
        resolved: SearchQuery,
        method: str,
    ) -> tuple[SearchResult, ...]:
        query_counts = dict(sorted(Counter(resolved.normalized_terms).items()))
        query_weights = sparse_tfidf_weights(
            query_counts,
            self.document_frequencies,
            len(self.chunks),
        )
        scored: list[
            tuple[float, Chunk, Document, tuple[dict[str, Any], ...], dict[str, Any]]
        ] = []
        for chunk, frequencies in zip(
            self.chunks,
            self.term_frequencies,
            strict=True,
        ):
            document = self._document_by_id[chunk.document_id]
            if not self._matches_filters(chunk, document, resolved.filters):
                continue
            if method == "tfidf":
                document_weights = sparse_tfidf_weights(
                    frequencies,
                    self.document_frequencies,
                    len(self.chunks),
                )
                score, numerator, query_norm, document_norm, raw = cosine_score(
                    query_weights,
                    document_weights,
                )
                denominator = query_norm * document_norm
                contributions = tuple(
                    {
                        "term": term,
                        "term_frequency": frequencies[term],
                        "query_term_frequency": query_counts[term],
                        "document_frequency": self.document_frequencies[term],
                        "idf": smooth_inverse_document_frequency(
                            len(self.chunks),
                            self.document_frequencies[term],
                        ),
                        "query_weight": query_weights[term],
                        "chunk_weight": document_weights[term],
                        "dot_product": value,
                        "score_contribution": (
                            0.0 if denominator == 0.0 else value / denominator
                        ),
                    }
                    for term, value in raw.items()
                )
                details = {
                    "cosine_numerator": numerator,
                    "query_norm": query_norm,
                    "chunk_norm": document_norm,
                }
            else:
                records: list[dict[str, Any]] = []
                # Repeated BM25 query terms intentionally contribute once.
                for term in sorted(set(resolved.normalized_terms) & set(frequencies)):
                    contribution, inverse_document_frequency, length_normalization = (
                        bm25_term_contribution(
                            term_frequency=frequencies[term],
                            document_frequency=self.document_frequencies[term],
                            document_length=chunk.term_count,
                            average_document_length=self.average_chunk_length,
                            number_of_chunks=len(self.chunks),
                            config=self.bm25_config,
                        )
                    )
                    records.append(
                        {
                            "term": term,
                            "term_frequency": frequencies[term],
                            "document_frequency": self.document_frequencies[term],
                            "idf": inverse_document_frequency,
                            "length_normalization": length_normalization,
                            "score_contribution": contribution,
                        }
                    )
                contributions = tuple(records)
                score = math.fsum(
                    record["score_contribution"] for record in contributions
                )
                details = {
                    "chunk_length": chunk.term_count,
                    "average_chunk_length": self.average_chunk_length,
                    "k1": self.bm25_config.k1,
                    "b": self.bm25_config.b,
                    "repeated_query_term_policy": "unique_terms",
                }
            if score > 0.0:
                scored.append((score, chunk, document, contributions, details))
        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].document_id,
                item[1].ordinal,
                item[1].chunk_id,
            )
        )
        results: list[SearchResult] = []
        for rank, (score, chunk, document, contributions, details) in enumerate(
            scored[: resolved.top_k],
            start=1,
        ):
            results.append(
                self._make_result(
                    rank=rank,
                    score=score,
                    method=method,
                    chunk=chunk,
                    document=document,
                    matched_terms=tuple(record["term"] for record in contributions),
                    semantic_query_terms=(),
                    contributions=contributions,
                    details=details,
                )
            )
        return tuple(results)

    def _search_semantic(
        self,
        resolved: SearchQuery,
    ) -> tuple[SearchResult, ...]:
        semantic = self.semantic_index
        if semantic is None:
            raise ValueError(
                "Semantic retrieval is unavailable for this lexical-only index; "
                "enrich it explicitly first."
            )
        projection = semantic.project_query(
            resolved.normalized_terms,
            self.document_frequencies,
        )
        if projection.raw_norm == 0.0:
            return ()
        scores, numerators = semantic.cosine_scores(projection)
        scored: list[
            tuple[
                float,
                Chunk,
                Document,
                dict[str, Any],
                tuple[str, ...],
            ]
        ] = []
        for position, chunk in enumerate(self.chunks):
            document = self._document_by_id[chunk.document_id]
            if not self._matches_filters(chunk, document, resolved.filters):
                continue
            score = float(scores[position])
            if score <= 0.0:
                continue
            coordinate_products = (
                semantic.chunk_embeddings[position] * projection.embedding
            )
            strongest = sorted(
                (
                    {
                        "coordinate": coordinate,
                        "product": float(product),
                    }
                    for coordinate, product in enumerate(coordinate_products)
                ),
                key=lambda record: (
                    -abs(record["product"]),
                    record["coordinate"],
                ),
            )
            details = {
                "semantic_method": semantic.config.method,
                "semantic_sha256": semantic.semantic_sha256,
                "latent_dimension": semantic.config.dimensions,
                "query_raw_norm": projection.raw_norm,
                "query_stored_norm": float(np.linalg.norm(projection.embedding)),
                "chunk_raw_norm": float(semantic.chunk_raw_norms[position]),
                "chunk_stored_norm": float(
                    np.linalg.norm(semantic.chunk_embeddings[position])
                ),
                "cosine_numerator": float(numerators[position]),
                "known_query_terms": list(projection.known_terms),
                "out_of_vocabulary_query_terms": list(
                    projection.out_of_vocabulary_terms
                ),
                "query_tfidf_weights": projection.tfidf_weights,
                "query_term_latent_norms": projection.term_latent_norms,
                "strongest_latent_coordinate_products": strongest,
                "latent_dimension_labels": None,
                "similarity_is_not_proof_of_relevance": True,
            }
            exact_matches = tuple(
                sorted(
                    set(projection.known_terms) & set(self.term_frequencies[position])
                )
            )
            scored.append((score, chunk, document, details, exact_matches))
        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].document_id,
                item[1].ordinal,
                item[1].chunk_id,
            )
        )
        contributions = tuple(
            {
                "term": term,
                "query_weight": projection.tfidf_weights[term],
                "latent_contribution_norm": projection.term_latent_norms[term],
            }
            for term in projection.known_terms
        )
        return tuple(
            self._make_result(
                rank=rank,
                score=score,
                method="semantic",
                chunk=chunk,
                document=document,
                matched_terms=exact_matches,
                semantic_query_terms=projection.known_terms,
                contributions=contributions,
                details=details,
            )
            for rank, (score, chunk, document, details, exact_matches) in enumerate(
                scored[: resolved.top_k],
                start=1,
            )
        )

    def _search_hybrid(
        self,
        resolved: SearchQuery,
        *,
        method: str,
        hybrid_config: HybridRetrievalConfig,
        reranking_config: RerankingConfig | None,
    ) -> tuple[SearchResult, ...]:
        lexical_query = replace(
            resolved,
            top_k=hybrid_config.lexical_candidate_count,
        )
        semantic_query = replace(
            resolved,
            top_k=hybrid_config.semantic_candidate_count,
        )
        lexical_results = self._search_lexical(
            lexical_query,
            hybrid_config.lexical_method,
        )
        semantic_results = self._search_semantic(semantic_query)
        lexical_by_id = {result.chunk_id: result for result in lexical_results}
        semantic_by_id = {result.chunk_id: result for result in semantic_results}
        chunk_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        fused: list[SearchResult] = []
        for candidate in fuse_rankings(
            lexical_results,
            semantic_results,
            config=hybrid_config,
        ):
            if candidate["score"] <= 0.0:
                continue
            chunk = chunk_by_id[candidate["chunk_id"]]
            document = self._document_by_id[chunk.document_id]
            lexical_result = lexical_by_id.get(chunk.chunk_id)
            semantic_result = semantic_by_id.get(chunk.chunk_id)
            matched_terms = tuple(
                sorted(
                    {
                        term
                        for result in (lexical_result, semantic_result)
                        if result is not None
                        for term in result.matched_terms
                    }
                )
            )
            semantic_query_terms = tuple(
                sorted(
                    {
                        term
                        for result in (lexical_result, semantic_result)
                        if result is not None
                        for term in result.semantic_query_terms
                    }
                )
            )
            contributions = (
                () if lexical_result is None else lexical_result.term_contributions
            )
            details = {
                "fusion": candidate,
                "lexical_scoring": (
                    None if lexical_result is None else lexical_result.scoring_details
                ),
                "semantic_scoring": (
                    None if semantic_result is None else semantic_result.scoring_details
                ),
                "deterministic_tie_break": ("score_desc_document_id_ordinal_chunk_id"),
            }
            fused.append(
                self._make_result(
                    rank=1,
                    score=candidate["score"],
                    method="hybrid",
                    chunk=chunk,
                    document=document,
                    matched_terms=matched_terms,
                    semantic_query_terms=semantic_query_terms,
                    contributions=contributions,
                    details=details,
                )
            )
        fused.sort(
            key=lambda result: (
                -result.score,
                result.document_id,
                chunk_by_id[result.chunk_id].ordinal,
                result.chunk_id,
            )
        )
        fused = [
            replace(result, rank=rank) for rank, result in enumerate(fused, start=1)
        ]
        if method == "hybrid":
            return tuple(fused[: resolved.top_k])
        if reranking_config is None or not reranking_config.enabled:
            raise ValueError("hybrid_reranked requires an enabled RerankingConfig.")
        remaining = list(fused[: reranking_config.candidate_count])
        selected: list[SearchResult] = []
        while remaining and len(selected) < resolved.top_k:
            rescored: list[tuple[SearchResult, float, dict[str, Any]]] = []
            for result in remaining:
                chunk = chunk_by_id[result.chunk_id]
                document = self._document_by_id[result.document_id]
                overlap = max(
                    (
                        source_range_overlap(
                            chunk,
                            chunk_by_id[chosen.chunk_id],
                        )
                        for chosen in selected
                    ),
                    default=0.0,
                )
                features = reranking_features(
                    query=resolved.raw_text,
                    chunk=chunk,
                    document=document,
                    fusion_details=result.scoring_details["fusion"],
                    document_frequencies=self.document_frequencies,
                    chunk_count=len(self.chunks),
                )
                score, contributions = weighted_reranking_score(
                    features,
                    reranking_config,
                    redundancy_overlap=overlap,
                )
                details = {
                    "config": reranking_config.to_dict(),
                    "features": features,
                    "contributions": contributions,
                    "unclamped_score": score - contributions["clamp_adjustment"],
                    "maximum_selected_source_overlap": overlap,
                    "redundancy_penalty_applied": (
                        overlap >= reranking_config.redundancy_threshold
                    ),
                    "final_score": score,
                    "tie_break": "score_desc_document_id_ordinal_chunk_id",
                }
                rescored.append((result, score, details))
            rescored.sort(
                key=lambda item: (
                    -item[1],
                    item[0].document_id,
                    chunk_by_id[item[0].chunk_id].ordinal,
                    item[0].chunk_id,
                )
            )
            chosen, score, reranking_details = rescored[0]
            if score <= 0.0:
                break
            updated_details = dict(chosen.scoring_details)
            updated_details["reranker"] = reranking_details
            selected.append(
                replace(
                    chosen,
                    rank=len(selected) + 1,
                    score=score,
                    retrieval_method="hybrid_reranked",
                    scoring_details=updated_details,
                )
            )
            remaining = [
                result for result in remaining if result.chunk_id != chosen.chunk_id
            ]
        return tuple(selected)

    def change_reasons(
        self,
        documents: Sequence[Document],
        *,
        chunking_config: ChunkingConfig | None = None,
        lexical_config: LexicalTokenizerConfig | None = None,
        bm25_config: BM25Config | None = None,
    ) -> tuple[str, ...]:
        """Explain why an immutable snapshot would change on full rebuild."""
        new_by_source = {document.source_path: document for document in documents}
        old_by_source = {document.source_path: document for document in self.documents}
        reasons: list[str] = []
        for source in sorted(set(new_by_source) - set(old_by_source)):
            reasons.append(f"source_added:{source}")
        for source in sorted(set(old_by_source) - set(new_by_source)):
            reasons.append(f"source_removed:{source}")
        for source in sorted(set(old_by_source) & set(new_by_source)):
            if (
                old_by_source[source].content_sha256
                != new_by_source[source].content_sha256
            ):
                reasons.append(f"source_content_changed:{source}")
        if chunking_config is not None and chunking_config != self.chunking_config:
            reasons.append("chunking_configuration_changed")
        if lexical_config is not None and lexical_config != self.lexical_config:
            reasons.append("lexical_configuration_changed")
        if bm25_config is not None and bm25_config != self.bm25_config:
            reasons.append("bm25_configuration_changed")
        return tuple(reasons or ["unchanged"])


def highlight_matches(text: str, terms: Sequence[str]) -> str:
    """Return a display-only marked copy while preserving original index text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if isinstance(terms, (str, bytes)) or not isinstance(terms, Sequence):
        raise TypeError("terms must be a sequence of normalized strings.")
    if not all(isinstance(term, str) and term for term in terms):
        raise ValueError("terms must contain non-empty normalized strings.")
    normalized = set(terms)
    if not normalized:
        return text
    spans = [
        (term.start_character, term.end_character)
        for term in lexical_terms(text)
        if term.term in normalized
    ]
    output = text
    for start, end in reversed(spans):
        output = output[:start] + "[[" + output[start:end] + "]]" + output[end:]
    return output
