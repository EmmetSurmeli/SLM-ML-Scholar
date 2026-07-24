"""Immutable deterministic lexical index, search, filtering, and persistence."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

INDEX_FORMAT_VERSION = 1


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
    """One exact ranked passage and transparent lexical scoring evidence."""

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
        if self.retrieval_method not in {"tfidf", "bm25"}:
            raise ValueError("Unknown retrieval method.")
        if self.authors is not None and (
            not isinstance(self.authors, tuple)
            or not all(isinstance(author, str) and author for author in self.authors)
        ):
            raise ValueError("authors must be None or a tuple of strings.")
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
        )

    def _state_without_index_hash(self) -> dict[str, Any]:
        return {
            "index_format_version": INDEX_FORMAT_VERSION,
            "package_version": self.package_version,
            "index_type": "immutable_lexical_snapshot",
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
        expected = {
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
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Retrieval index state keys are malformed.")
        values = dict(state)
        if values["index_format_version"] != INDEX_FORMAT_VERSION:
            raise ValueError("Unsupported retrieval index format version.")
        if values["index_type"] != "immutable_lexical_snapshot":
            raise ValueError("Retrieval index type is incompatible.")
        if values["package_version"] != __version__:
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
        if rebuilt.state_dict() != index.state_dict():
            raise ValueError("Retrieval index statistics do not reconstruct exactly.")
        return index

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
    ) -> tuple[SearchResult, ...]:
        """Return ranked source passages only; this method never generates prose."""
        if method not in {"tfidf", "bm25"}:
            raise ValueError("method must be 'tfidf' or 'bm25'.")
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
            results.append(
                SearchResult(
                    rank=rank,
                    score=score,
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
                    matched_terms=tuple(record["term"] for record in contributions),
                    term_contributions=contributions,
                    scoring_details=details,
                    citation=citation,
                )
            )
        return tuple(results)

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
