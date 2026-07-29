"""Deterministic TF-IDF latent semantic analysis and exact query projection."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from localml_scholar.retrieval.documents import canonical_json
from localml_scholar.retrieval.tfidf import sparse_tfidf_weights

SEMANTIC_INDEX_FORMAT_VERSION = 1


def _sha256_state(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def vocabulary_sha256(vocabulary: Sequence[str]) -> str:
    """Return the deterministic identity of an ordered lexical vocabulary."""
    if isinstance(vocabulary, (str, bytes)) or not isinstance(vocabulary, Sequence):
        raise TypeError("vocabulary must be a sequence of strings.")
    values = tuple(vocabulary)
    if not values or any(not isinstance(term, str) or not term for term in values):
        raise ValueError("vocabulary must contain non-empty strings.")
    if values != tuple(sorted(set(values))):
        raise ValueError("vocabulary must be sorted and unique.")
    return _sha256_state(list(values))


@dataclass(frozen=True)
class SemanticRetrievalConfig:
    """Immutable local LSA construction and normalization policy."""

    method: str = "lsa"
    dimensions: int = 1
    weighting: str = "tfidf"
    center: bool = False
    normalize_embeddings: bool = True
    minimum_singular_value: float = 1e-12
    numerical_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if self.method != "lsa":
            raise ValueError("method must be 'lsa'.")
        if isinstance(self.dimensions, bool) or not isinstance(self.dimensions, int):
            raise TypeError("dimensions must be an integer.")
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive.")
        if self.weighting != "tfidf":
            raise ValueError("weighting must be 'tfidf'.")
        if not isinstance(self.center, bool):
            raise TypeError("center must be boolean.")
        if self.center:
            raise ValueError("Centered LSA is not supported by this baseline.")
        if not isinstance(self.normalize_embeddings, bool):
            raise TypeError("normalize_embeddings must be boolean.")
        for name in ("minimum_singular_value", "numerical_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number.")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, normalized)

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> SemanticRetrievalConfig:
        if not isinstance(state, Mapping) or set(state) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("Semantic retrieval configuration is malformed.")
        return cls(**dict(state))

    @property
    def state_sha256(self) -> str:
        return _sha256_state(self.to_dict())


def build_tfidf_matrix(
    term_frequencies: Sequence[Mapping[str, int]],
    document_frequencies: Mapping[str, int],
    vocabulary: Sequence[str],
) -> np.ndarray:
    """Build an ``(chunks, terms)`` dense matrix without mutating lexical state."""
    if isinstance(term_frequencies, (str, bytes)) or not isinstance(
        term_frequencies, Sequence
    ):
        raise TypeError("term_frequencies must be a sequence of mappings.")
    if not term_frequencies:
        raise ValueError("At least one term-frequency row is required.")
    terms = tuple(vocabulary)
    vocabulary_sha256(terms)
    if not isinstance(document_frequencies, Mapping):
        raise TypeError("document_frequencies must be a mapping.")
    if set(document_frequencies) != set(terms):
        raise ValueError("Document frequencies must align with the vocabulary.")
    chunk_count = len(term_frequencies)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= chunk_count
        for value in document_frequencies.values()
    ):
        raise ValueError("Document frequencies must lie in [0, chunk_count].")
    term_to_column = {term: column for column, term in enumerate(terms)}
    matrix = np.zeros((chunk_count, len(terms)), dtype=np.float64)
    for row_index, frequencies in enumerate(term_frequencies):
        if not isinstance(frequencies, Mapping):
            raise TypeError("Each term-frequency row must be a mapping.")
        if any(
            term not in term_to_column
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for term, count in frequencies.items()
        ):
            raise ValueError(
                "Term-frequency rows must use vocabulary terms and positive counts."
            )
        weights = sparse_tfidf_weights(
            frequencies,
            document_frequencies,
            chunk_count,
        )
        for term, weight in weights.items():
            matrix[row_index, term_to_column[term]] = weight
    if not np.all(np.isfinite(matrix)):
        raise ValueError("TF-IDF matrix contains non-finite values.")
    return matrix


def canonicalize_svd_signs(
    left_vectors: np.ndarray,
    right_vectors_transposed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Canonicalize each component using the first largest-magnitude right entry."""
    left = np.asarray(left_vectors)
    right = np.asarray(right_vectors_transposed)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("SVD factors must both be two-dimensional.")
    if left.shape[1] != right.shape[0]:
        raise ValueError("SVD component dimensions do not align.")
    if not np.issubdtype(left.dtype, np.floating) or not np.issubdtype(
        right.dtype, np.floating
    ):
        raise TypeError("SVD factors must have floating-point dtypes.")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("SVD factors must be finite.")
    canonical_left = left.astype(np.float64, copy=True)
    canonical_right = right.astype(np.float64, copy=True)
    pivots: list[int] = []
    for component in range(canonical_right.shape[0]):
        row = canonical_right[component]
        pivot = int(np.argmax(np.abs(row)))
        pivots.append(pivot)
        if row[pivot] < 0.0:
            canonical_left[:, component] *= -1.0
            canonical_right[component] *= -1.0
    return canonical_left, canonical_right, tuple(pivots)


def _readonly_float_array(
    value: object,
    *,
    name: str,
    dimensions: int,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions.")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have a floating-point dtype.")
    copied = array.astype(np.float64, copy=True)
    if not np.all(np.isfinite(copied)):
        raise ValueError(f"{name} must contain only finite values.")
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class SemanticQueryProjection:
    """One query vector and the transparent lexical state used to create it."""

    embedding: np.ndarray
    raw_norm: float
    known_terms: tuple[str, ...]
    out_of_vocabulary_terms: tuple[str, ...]
    tfidf_weights: dict[str, float]
    term_latent_norms: dict[str, float]

    def __post_init__(self) -> None:
        embedding = _readonly_float_array(
            self.embedding,
            name="query embedding",
            dimensions=1,
        )
        object.__setattr__(self, "embedding", embedding)
        if not math.isfinite(self.raw_norm) or self.raw_norm < 0.0:
            raise ValueError("Query norm must be finite and non-negative.")


class SemanticIndex:
    """Immutable LSA factors aligned to one lexical index snapshot."""

    def __init__(
        self,
        *,
        config: SemanticRetrievalConfig,
        vocabulary: tuple[str, ...],
        chunk_ids: tuple[str, ...],
        right_singular_vectors: np.ndarray,
        chunk_embeddings: np.ndarray,
        chunk_raw_norms: np.ndarray,
        singular_values: np.ndarray,
        effective_rank: int,
        reconstruction_error: float,
        explained_squared_singular_fraction: float,
        zero_row_indices: tuple[int, ...],
        canonical_pivot_indices: tuple[int, ...],
        matrix_shape: tuple[int, int],
        semantic_sha256: str = "pending",
    ) -> None:
        if not isinstance(config, SemanticRetrievalConfig):
            raise TypeError("config must be SemanticRetrievalConfig.")
        vocabulary_sha256(vocabulary)
        if (
            not chunk_ids
            or len(set(chunk_ids)) != len(chunk_ids)
            or any(
                not isinstance(chunk_id, str) or not chunk_id for chunk_id in chunk_ids
            )
        ):
            raise ValueError("chunk_ids must be non-empty and unique.")
        if matrix_shape != (len(chunk_ids), len(vocabulary)):
            raise ValueError("matrix_shape does not match chunks and vocabulary.")
        right = _readonly_float_array(
            right_singular_vectors,
            name="right_singular_vectors",
            dimensions=2,
        )
        embeddings = _readonly_float_array(
            chunk_embeddings,
            name="chunk_embeddings",
            dimensions=2,
        )
        norms = _readonly_float_array(
            chunk_raw_norms,
            name="chunk_raw_norms",
            dimensions=1,
        )
        singular = _readonly_float_array(
            singular_values,
            name="singular_values",
            dimensions=1,
        )
        expected_dimension = config.dimensions
        if right.shape != (expected_dimension, len(vocabulary)):
            raise ValueError("Right singular vectors have an invalid shape.")
        if embeddings.shape != (len(chunk_ids), expected_dimension):
            raise ValueError("Chunk embeddings have an invalid shape.")
        if norms.shape != (len(chunk_ids),):
            raise ValueError("Chunk norms have an invalid shape.")
        if singular.shape != (expected_dimension,) or np.any(singular <= 0.0):
            raise ValueError("Retained singular values must be positive.")
        if np.any(norms < 0.0):
            raise ValueError("Chunk norms must be non-negative.")
        if isinstance(effective_rank, bool) or not isinstance(effective_rank, int):
            raise TypeError("effective_rank must be an integer.")
        if effective_rank < expected_dimension:
            raise ValueError("effective_rank cannot be below the retained dimension.")
        for name, value in (
            ("reconstruction_error", reconstruction_error),
            (
                "explained_squared_singular_fraction",
                explained_squared_singular_fraction,
            ),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if explained_squared_singular_fraction > 1.0 + config.numerical_tolerance:
            raise ValueError("explained_squared_singular_fraction cannot exceed one.")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(chunk_ids)
            for index in zero_row_indices
        ):
            raise ValueError("zero_row_indices contain an invalid row.")
        if len(canonical_pivot_indices) != expected_dimension or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(vocabulary)
            for index in canonical_pivot_indices
        ):
            raise ValueError("Canonical pivot indices are malformed.")
        for component, pivot in enumerate(canonical_pivot_indices):
            if right[component, pivot] < 0.0:
                raise ValueError("SVD signs are not canonical.")
            expected_pivot = int(np.argmax(np.abs(right[component])))
            if expected_pivot != pivot:
                raise ValueError("Canonical SVD pivot is inconsistent.")
        if config.normalize_embeddings:
            stored_norms = np.linalg.norm(embeddings, axis=1)
            nonzero = norms > 0.0
            if not np.allclose(
                stored_norms[nonzero],
                1.0,
                rtol=0.0,
                atol=config.numerical_tolerance,
            ):
                raise ValueError("Stored chunk embeddings are not normalized.")
            if np.any(embeddings[~nonzero] != 0.0):
                raise ValueError("Zero-norm chunks must have zero embeddings.")
        self.config = config
        self.vocabulary = vocabulary
        self.chunk_ids = chunk_ids
        self.right_singular_vectors = right
        self.chunk_embeddings = embeddings
        self.chunk_raw_norms = norms
        self.singular_values = singular
        self.effective_rank = effective_rank
        self.reconstruction_error = float(reconstruction_error)
        self.explained_squared_singular_fraction = float(
            explained_squared_singular_fraction
        )
        self.zero_row_indices = zero_row_indices
        self.canonical_pivot_indices = canonical_pivot_indices
        self.matrix_shape = matrix_shape
        self.vocabulary_sha256 = vocabulary_sha256(vocabulary)
        self.config_sha256 = config.state_sha256
        calculated = self._calculated_sha256()
        if semantic_sha256 != "pending" and semantic_sha256 != calculated:
            raise ValueError("Semantic index hash is inconsistent.")
        self.semantic_sha256 = calculated

    def _state_without_hash(self) -> dict[str, Any]:
        return {
            "semantic_index_format_version": SEMANTIC_INDEX_FORMAT_VERSION,
            "method": self.config.method,
            "config": self.config.to_dict(),
            "config_sha256": self.config_sha256,
            "vocabulary": list(self.vocabulary),
            "vocabulary_sha256": self.vocabulary_sha256,
            "chunk_ids": list(self.chunk_ids),
            "right_singular_vectors": self.right_singular_vectors.tolist(),
            "chunk_embeddings": self.chunk_embeddings.tolist(),
            "chunk_raw_norms": self.chunk_raw_norms.tolist(),
            "singular_values": self.singular_values.tolist(),
            "effective_rank": self.effective_rank,
            "reconstruction_error": self.reconstruction_error,
            "explained_squared_singular_fraction": (
                self.explained_squared_singular_fraction
            ),
            "zero_row_indices": list(self.zero_row_indices),
            "canonicalization": {
                "policy": "largest_absolute_right_entry_nonnegative",
                "tie_break": "lowest_term_index",
                "pivot_indices": list(self.canonical_pivot_indices),
            },
            "matrix_shape": list(self.matrix_shape),
            "dtype": "float64",
        }

    def _calculated_sha256(self) -> str:
        return _sha256_state(self._state_without_hash())

    def to_dict(self) -> dict[str, Any]:
        state = self._state_without_hash()
        state["semantic_sha256"] = self.semantic_sha256
        return state

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> SemanticIndex:
        expected = {
            "semantic_index_format_version",
            "method",
            "config",
            "config_sha256",
            "vocabulary",
            "vocabulary_sha256",
            "chunk_ids",
            "right_singular_vectors",
            "chunk_embeddings",
            "chunk_raw_norms",
            "singular_values",
            "effective_rank",
            "reconstruction_error",
            "explained_squared_singular_fraction",
            "zero_row_indices",
            "canonicalization",
            "matrix_shape",
            "dtype",
            "semantic_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Semantic index state keys are malformed.")
        values = dict(state)
        if values["semantic_index_format_version"] != SEMANTIC_INDEX_FORMAT_VERSION:
            raise ValueError("Unsupported semantic index format version.")
        if values["method"] != "lsa" or values["dtype"] != "float64":
            raise ValueError("Semantic index method or dtype is incompatible.")
        config = SemanticRetrievalConfig.from_dict(values["config"])
        if values["config_sha256"] != config.state_sha256:
            raise ValueError("Semantic configuration hash is inconsistent.")
        for name in (
            "vocabulary",
            "chunk_ids",
            "right_singular_vectors",
            "chunk_embeddings",
            "chunk_raw_norms",
            "singular_values",
            "zero_row_indices",
            "matrix_shape",
        ):
            if not isinstance(values[name], list):
                raise ValueError(f"Semantic index {name} must be a list.")
        vocabulary = tuple(values["vocabulary"])
        if values["vocabulary_sha256"] != vocabulary_sha256(vocabulary):
            raise ValueError("Semantic vocabulary hash is inconsistent.")
        canonicalization = values["canonicalization"]
        if not isinstance(canonicalization, Mapping) or set(canonicalization) != {
            "policy",
            "tie_break",
            "pivot_indices",
        }:
            raise ValueError("Semantic canonicalization state is malformed.")
        if (
            canonicalization["policy"] != "largest_absolute_right_entry_nonnegative"
            or canonicalization["tie_break"] != "lowest_term_index"
            or not isinstance(canonicalization["pivot_indices"], list)
        ):
            raise ValueError("Semantic canonicalization policy is incompatible.")
        if len(values["matrix_shape"]) != 2:
            raise ValueError("Semantic matrix shape is malformed.")
        return cls(
            config=config,
            vocabulary=vocabulary,
            chunk_ids=tuple(values["chunk_ids"]),
            right_singular_vectors=np.asarray(values["right_singular_vectors"]),
            chunk_embeddings=np.asarray(values["chunk_embeddings"]),
            chunk_raw_norms=np.asarray(values["chunk_raw_norms"]),
            singular_values=np.asarray(values["singular_values"]),
            effective_rank=values["effective_rank"],
            reconstruction_error=values["reconstruction_error"],
            explained_squared_singular_fraction=values[
                "explained_squared_singular_fraction"
            ],
            zero_row_indices=tuple(values["zero_row_indices"]),
            canonical_pivot_indices=tuple(canonicalization["pivot_indices"]),
            matrix_shape=tuple(values["matrix_shape"]),
            semantic_sha256=values["semantic_sha256"],
        )

    def project_query(
        self,
        query_terms: Sequence[str],
        document_frequencies: Mapping[str, int],
    ) -> SemanticQueryProjection:
        """Project lexical query TF-IDF into the retained latent coordinates."""
        if isinstance(query_terms, (str, bytes)) or not isinstance(
            query_terms, Sequence
        ):
            raise TypeError("query_terms must be a sequence of strings.")
        if not all(isinstance(term, str) and term for term in query_terms):
            raise ValueError("query_terms must contain non-empty strings.")
        if set(document_frequencies) != set(self.vocabulary):
            raise ValueError("Query vocabulary does not match the semantic index.")
        counts: dict[str, int] = {}
        out_of_vocabulary: list[str] = []
        vocabulary_set = set(self.vocabulary)
        for term in query_terms:
            if term in vocabulary_set:
                counts[term] = counts.get(term, 0) + 1
            elif term not in out_of_vocabulary:
                out_of_vocabulary.append(term)
        weights = sparse_tfidf_weights(
            counts,
            document_frequencies,
            len(self.chunk_ids),
        )
        dense = np.zeros(len(self.vocabulary), dtype=np.float64)
        term_to_column = {term: index for index, term in enumerate(self.vocabulary)}
        term_latent_norms: dict[str, float] = {}
        for term, weight in weights.items():
            column = term_to_column[term]
            dense[column] = weight
            latent_contribution = weight * self.right_singular_vectors[:, column]
            term_latent_norms[term] = float(np.linalg.norm(latent_contribution))
        embedding = dense @ self.right_singular_vectors.T
        raw_norm = float(np.linalg.norm(embedding))
        if self.config.normalize_embeddings and raw_norm > 0.0:
            embedding = embedding / raw_norm
        return SemanticQueryProjection(
            embedding=embedding,
            raw_norm=raw_norm,
            known_terms=tuple(sorted(weights)),
            out_of_vocabulary_terms=tuple(out_of_vocabulary),
            tfidf_weights=dict(sorted(weights.items())),
            term_latent_norms=dict(sorted(term_latent_norms.items())),
        )

    def cosine_scores(
        self,
        query: SemanticQueryProjection,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return exact cosine scores and dot-product numerators for all chunks."""
        if not isinstance(query, SemanticQueryProjection):
            raise TypeError("query must be SemanticQueryProjection.")
        if query.embedding.shape != (self.config.dimensions,):
            raise ValueError("Query latent dimension does not match the index.")
        query_norm = float(np.linalg.norm(query.embedding))
        stored_norms = np.linalg.norm(self.chunk_embeddings, axis=1)
        numerators = self.chunk_embeddings @ query.embedding
        denominators = stored_norms * query_norm
        scores = np.divide(
            numerators,
            denominators,
            out=np.zeros_like(numerators),
            where=denominators > 0.0,
        )
        if not np.all(np.isfinite(scores)):
            raise ValueError("Semantic cosine scores became non-finite.")
        return scores, numerators


def fit_lsa(
    *,
    term_frequencies: Sequence[Mapping[str, int]],
    document_frequencies: Mapping[str, int],
    vocabulary: Sequence[str],
    chunk_ids: Sequence[str],
    config: SemanticRetrievalConfig | None = None,
) -> SemanticIndex:
    """Fit deterministic truncated LSA from existing lexical statistics."""
    resolved = config or SemanticRetrievalConfig()
    if not isinstance(resolved, SemanticRetrievalConfig):
        raise TypeError("config must be SemanticRetrievalConfig.")
    terms = tuple(vocabulary)
    chunks = tuple(chunk_ids)
    if len(chunks) != len(term_frequencies):
        raise ValueError("chunk_ids must align with term-frequency rows.")
    matrix = build_tfidf_matrix(
        term_frequencies,
        document_frequencies,
        terms,
    )
    zero_rows = tuple(
        int(index) for index in np.flatnonzero(np.linalg.norm(matrix, axis=1) == 0.0)
    )
    if not np.any(matrix):
        raise ValueError("Cannot fit LSA to an all-zero term-chunk matrix.")
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    if not (
        np.all(np.isfinite(left))
        and np.all(np.isfinite(singular_values))
        and np.all(np.isfinite(right))
    ):
        raise ValueError("SVD produced non-finite factors.")
    full_reconstruction = (left * singular_values) @ right
    full_error = float(np.linalg.norm(matrix - full_reconstruction))
    full_scale = max(1.0, float(np.linalg.norm(matrix)))
    if full_error > resolved.numerical_tolerance * full_scale:
        raise ValueError(
            "Full SVD reconstruction exceeded the configured numerical tolerance."
        )
    machine_threshold = (
        max(matrix.shape) * np.finfo(np.float64).eps * float(singular_values[0])
    )
    rank_threshold = max(resolved.minimum_singular_value, machine_threshold)
    effective_rank = int(np.sum(singular_values > rank_threshold))
    if resolved.dimensions > effective_rank:
        raise ValueError(
            "Requested semantic dimensions exceed the effective matrix rank "
            f"({resolved.dimensions} requested, rank {effective_rank})."
        )
    dimension = resolved.dimensions
    selected_left, selected_right, pivots = canonicalize_svd_signs(
        left[:, :dimension],
        right[:dimension],
    )
    selected_singular = singular_values[:dimension].astype(np.float64, copy=True)
    raw_embeddings = selected_left * selected_singular
    raw_norms = np.linalg.norm(raw_embeddings, axis=1)
    embeddings = raw_embeddings.copy()
    if resolved.normalize_embeddings:
        np.divide(
            embeddings,
            raw_norms[:, None],
            out=embeddings,
            where=raw_norms[:, None] > 0.0,
        )
    approximation = raw_embeddings @ selected_right
    reconstruction_error = float(np.linalg.norm(matrix - approximation))
    total_squared = float(np.sum(singular_values * singular_values))
    retained_squared = float(np.sum(selected_singular * selected_singular))
    explained = 0.0 if total_squared == 0.0 else retained_squared / total_squared
    return SemanticIndex(
        config=resolved,
        vocabulary=terms,
        chunk_ids=chunks,
        right_singular_vectors=selected_right,
        chunk_embeddings=embeddings,
        chunk_raw_norms=raw_norms,
        singular_values=selected_singular,
        effective_rank=effective_rank,
        reconstruction_error=reconstruction_error,
        explained_squared_singular_fraction=explained,
        zero_row_indices=zero_rows,
        canonical_pivot_indices=pivots,
        matrix_shape=matrix.shape,
    )
