"""Explicit lexical-semantic fusion and transparent deterministic reranking."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from localml_scholar.retrieval.documents import Chunk, Document
from localml_scholar.retrieval.text import tokenize_lexically


@dataclass(frozen=True)
class HybridRetrievalConfig:
    """Candidate depths and exact score-fusion convention."""

    lexical_method: str = "bm25"
    fusion: str = "rrf"
    alpha: float = 0.5
    rrf_rank_constant: int = 60
    lexical_weight: float = 1.0
    semantic_weight: float = 1.0
    lexical_candidate_count: int = 20
    semantic_candidate_count: int = 20

    def __post_init__(self) -> None:
        if self.lexical_method not in {"tfidf", "bm25"}:
            raise ValueError("lexical_method must be 'tfidf' or 'bm25'.")
        if self.fusion not in {"weighted", "rrf"}:
            raise ValueError("fusion must be 'weighted' or 'rrf'.")
        for name in ("alpha", "lexical_weight", "semantic_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number.")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, normalized)
        if self.alpha > 1.0:
            raise ValueError("alpha must lie in [0, 1].")
        if self.lexical_weight == 0.0 and self.semantic_weight == 0.0:
            raise ValueError("At least one RRF component weight must be positive.")
        for name in (
            "rrf_rank_constant",
            "lexical_candidate_count",
            "semantic_candidate_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> HybridRetrievalConfig:
        if not isinstance(state, Mapping) or set(state) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("Hybrid retrieval configuration is malformed.")
        return cls(**dict(state))


@dataclass(frozen=True)
class RerankingConfig:
    """Manually weighted, query-transparent reranking policy."""

    enabled: bool = True
    candidate_count: int = 20
    lexical_weight: float = 1.0
    semantic_weight: float = 1.0
    phrase_weight: float = 0.25
    heading_weight: float = 0.15
    coverage_weight: float = 0.4
    rare_term_weight: float = 0.2
    numerical_overlap_weight: float = 0.15
    identifier_overlap_weight: float = 0.15
    length_penalty: float = 0.05
    redundancy_penalty: float = 0.5
    redundancy_threshold: float = 0.8

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be boolean.")
        if isinstance(self.candidate_count, bool) or not isinstance(
            self.candidate_count, int
        ):
            raise TypeError("candidate_count must be an integer.")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive.")
        for name in (
            "lexical_weight",
            "semantic_weight",
            "phrase_weight",
            "heading_weight",
            "coverage_weight",
            "rare_term_weight",
            "numerical_overlap_weight",
            "identifier_overlap_weight",
            "length_penalty",
            "redundancy_penalty",
            "redundancy_threshold",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number.")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, normalized)
        if self.redundancy_threshold > 1.0:
            raise ValueError("redundancy_threshold must lie in [0, 1].")

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> RerankingConfig:
        if not isinstance(state, Mapping) or set(state) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("Reranking configuration is malformed.")
        return cls(**dict(state))


def maximum_positive_normalization(scores: Mapping[str, float]) -> dict[str, float]:
    """Normalize non-negative component scores by their largest positive value."""
    if not isinstance(scores, Mapping):
        raise TypeError("scores must be a mapping.")
    if any(
        not isinstance(key, str)
        or not key
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0.0
        for key, value in scores.items()
    ):
        raise ValueError("scores must map non-empty IDs to finite non-negative values.")
    maximum = max(scores.values(), default=0.0)
    if maximum <= 0.0:
        return {key: 0.0 for key in sorted(scores)}
    return {key: float(scores[key] / maximum) for key in sorted(scores)}


def fuse_rankings(
    lexical_results: Sequence[Any],
    semantic_results: Sequence[Any],
    *,
    config: HybridRetrievalConfig | None = None,
) -> tuple[dict[str, Any], ...]:
    """Fuse two unique ranked lists and retain every component score and rank."""
    resolved = config or HybridRetrievalConfig()
    if not isinstance(resolved, HybridRetrievalConfig):
        raise TypeError("config must be HybridRetrievalConfig.")
    for name, results in (
        ("lexical_results", lexical_results),
        ("semantic_results", semantic_results),
    ):
        if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
            raise TypeError(f"{name} must be a result sequence.")
        identifiers = [result.chunk_id for result in results]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"{name} must not contain duplicate chunks.")
    lexical = {result.chunk_id: result for result in lexical_results}
    semantic = {result.chunk_id: result for result in semantic_results}
    lexical_scores = {key: float(result.score) for key, result in lexical.items()}
    semantic_scores = {key: float(result.score) for key, result in semantic.items()}
    lexical_normalized = maximum_positive_normalization(lexical_scores)
    semantic_normalized = maximum_positive_normalization(semantic_scores)
    candidates: list[dict[str, Any]] = []
    for chunk_id in sorted(set(lexical) | set(semantic)):
        lexical_result = lexical.get(chunk_id)
        semantic_result = semantic.get(chunk_id)
        lexical_rank = None if lexical_result is None else lexical_result.rank
        semantic_rank = None if semantic_result is None else semantic_result.rank
        if resolved.fusion == "weighted":
            lexical_component = resolved.alpha * lexical_normalized.get(chunk_id, 0.0)
            semantic_component = (1.0 - resolved.alpha) * (
                semantic_normalized.get(chunk_id, 0.0)
            )
            score = lexical_component + semantic_component
            formula = "alpha*max_normalized_lexical+(1-alpha)*max_normalized_semantic"
        else:
            lexical_component = (
                0.0
                if lexical_rank is None
                else resolved.lexical_weight
                / (resolved.rrf_rank_constant + lexical_rank)
            )
            semantic_component = (
                0.0
                if semantic_rank is None
                else resolved.semantic_weight
                / (resolved.rrf_rank_constant + semantic_rank)
            )
            score = lexical_component + semantic_component
            formula = "sum(component_weight/(rank_constant+component_rank))"
        candidates.append(
            {
                "chunk_id": chunk_id,
                "score": float(score),
                "lexical_score": (
                    None if lexical_result is None else float(lexical_result.score)
                ),
                "lexical_rank": lexical_rank,
                "semantic_score": (
                    None if semantic_result is None else float(semantic_result.score)
                ),
                "semantic_rank": semantic_rank,
                "normalized_lexical_score": lexical_normalized.get(chunk_id, 0.0),
                "normalized_semantic_score": semantic_normalized.get(chunk_id, 0.0),
                "lexical_fusion_component": float(lexical_component),
                "semantic_fusion_component": float(semantic_component),
                "retrieved_by": [
                    method
                    for method, result in (
                        (resolved.lexical_method, lexical_result),
                        ("semantic", semantic_result),
                    )
                    if result is not None
                ],
                "fusion_formula": formula,
                "fusion_config": resolved.to_dict(),
            }
        )
    return tuple(candidates)


_NUMBER = re.compile(r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?![\w.])")
_IDENTIFIER = re.compile(
    r"\b(?:[A-Za-z]+_[A-Za-z0-9_]+|[A-Za-z]+[A-Z][A-Za-z0-9]*|[A-Za-z]*\d+[A-Za-z0-9]*)\b"
)


def source_range_overlap(left: Chunk, right: Chunk) -> float:
    """Return shared source-character length divided by the shorter chunk length."""
    if not isinstance(left, Chunk) or not isinstance(right, Chunk):
        raise TypeError("source_range_overlap expects Chunk objects.")
    if left.document_id != right.document_id:
        return 0.0
    shared = max(
        0,
        min(left.end_character, right.end_character)
        - max(left.start_character, right.start_character),
    )
    shorter = min(
        left.end_character - left.start_character,
        right.end_character - right.start_character,
    )
    return 0.0 if shorter <= 0 else shared / shorter


def reranking_features(
    *,
    query: str,
    chunk: Chunk,
    document: Document,
    fusion_details: Mapping[str, Any],
    document_frequencies: Mapping[str, int],
    chunk_count: int,
) -> dict[str, float]:
    """Compute transparent scalar features without inferred user preferences."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must contain non-whitespace text.")
    if not isinstance(chunk, Chunk) or not isinstance(document, Document):
        raise TypeError("chunk and document must use retrieval data models.")
    if chunk.document_id != document.document_id:
        raise ValueError("chunk and document identities do not match.")
    query_terms = tuple(dict.fromkeys(tokenize_lexically(query)))
    chunk_terms = set(tokenize_lexically(chunk.text))
    heading_terms = set(tokenize_lexically(" ".join(chunk.heading_path)))
    shared = set(query_terms) & chunk_terms
    coverage = 0.0 if not query_terms else len(shared) / len(query_terms)
    rare_denominator = math.fsum(
        math.log((chunk_count + 1) / (document_frequencies.get(term, 0) + 1)) + 1.0
        for term in query_terms
    )
    rare_numerator = math.fsum(
        math.log((chunk_count + 1) / (document_frequencies.get(term, 0) + 1)) + 1.0
        for term in shared
    )
    normalized_query = " ".join(tokenize_lexically(query))
    normalized_chunk = " ".join(tokenize_lexically(chunk.text))
    numbers_query = set(_NUMBER.findall(query))
    numbers_chunk = set(_NUMBER.findall(chunk.text))
    identifiers_query = set(_IDENTIFIER.findall(query))
    identifiers_chunk = set(_IDENTIFIER.findall(chunk.text))
    return {
        "lexical": float(fusion_details.get("normalized_lexical_score", 0.0)),
        "semantic": float(fusion_details.get("normalized_semantic_score", 0.0)),
        "exact_phrase": float(
            bool(normalized_query and normalized_query in normalized_chunk)
        ),
        "heading_match": (
            0.0
            if not query_terms
            else len(set(query_terms) & heading_terms) / len(query_terms)
        ),
        "query_term_coverage": coverage,
        "rare_term_coverage": (
            0.0 if rare_denominator == 0.0 else rare_numerator / rare_denominator
        ),
        "numerical_overlap": (
            0.0
            if not numbers_query
            else len(numbers_query & numbers_chunk) / len(numbers_query)
        ),
        "identifier_overlap": (
            0.0
            if not identifiers_query
            else len(identifiers_query & identifiers_chunk) / len(identifiers_query)
        ),
        "length_penalty": min(1.0, chunk.term_count / 500.0),
    }


def weighted_reranking_score(
    features: Mapping[str, float],
    config: RerankingConfig,
    *,
    redundancy_overlap: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """Return a non-negative score and its exact signed feature contributions."""
    if not isinstance(config, RerankingConfig):
        raise TypeError("config must be RerankingConfig.")
    if not 0.0 <= redundancy_overlap <= 1.0:
        raise ValueError("redundancy_overlap must lie in [0, 1].")
    feature_weights = {
        "lexical": config.lexical_weight,
        "semantic": config.semantic_weight,
        "exact_phrase": config.phrase_weight,
        "heading_match": config.heading_weight,
        "query_term_coverage": config.coverage_weight,
        "rare_term_coverage": config.rare_term_weight,
        "numerical_overlap": config.numerical_overlap_weight,
        "identifier_overlap": config.identifier_overlap_weight,
        "length_penalty": -config.length_penalty,
    }
    if set(features) != set(feature_weights):
        raise ValueError("Reranking feature set is malformed.")
    contributions = {
        name: float(features[name] * weight) for name, weight in feature_weights.items()
    }
    applied_redundancy = (
        redundancy_overlap if redundancy_overlap >= config.redundancy_threshold else 0.0
    )
    contributions["redundancy_penalty"] = float(
        -config.redundancy_penalty * applied_redundancy
    )
    raw_score = math.fsum(contributions.values())
    score = max(0.0, raw_score)
    contributions["clamp_adjustment"] = float(score - raw_score)
    return score, contributions
