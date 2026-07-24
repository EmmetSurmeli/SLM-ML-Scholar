"""Independent sparse TF-IDF weighting and cosine primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping


def sublinear_term_frequency(count: int) -> float:
    """Return ``1 + log(count)`` for a positive count, else zero."""
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer.")
    if count < 0:
        raise ValueError("count must be non-negative.")
    return 0.0 if count == 0 else 1.0 + math.log(count)


def smooth_inverse_document_frequency(
    number_of_chunks: int,
    document_frequency: int,
) -> float:
    """Return ``log((N + 1)/(df + 1)) + 1``."""
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (number_of_chunks, document_frequency)
    ):
        raise TypeError("N and df must be integers.")
    if number_of_chunks <= 0:
        raise ValueError("number_of_chunks must be positive.")
    if not 0 <= document_frequency <= number_of_chunks:
        raise ValueError("document_frequency must lie in [0, N].")
    return math.log((number_of_chunks + 1) / (document_frequency + 1)) + 1.0


def sparse_tfidf_weights(
    term_frequencies: Mapping[str, int],
    document_frequencies: Mapping[str, int],
    number_of_chunks: int,
) -> dict[str, float]:
    """Create deterministic nonzero sparse TF-IDF weights."""
    weights: dict[str, float] = {}
    for term in sorted(term_frequencies):
        count = term_frequencies[term]
        if term not in document_frequencies:
            continue
        weight = sublinear_term_frequency(count) * smooth_inverse_document_frequency(
            number_of_chunks,
            document_frequencies[term],
        )
        if weight:
            weights[term] = weight
    return weights


def sparse_norm(weights: Mapping[str, float]) -> float:
    """Return a stable Euclidean norm for a sparse vector."""
    if not isinstance(weights, Mapping):
        raise TypeError("weights must be a mapping.")
    if any(not math.isfinite(value) for value in weights.values()):
        raise ValueError("Sparse weights must be finite.")
    return math.sqrt(math.fsum(value * value for value in weights.values()))


def cosine_score(
    query_weights: Mapping[str, float],
    document_weights: Mapping[str, float],
) -> tuple[float, float, float, float, dict[str, float]]:
    """Return score, numerator, norms, and per-term dot contributions."""
    shared = sorted(set(query_weights) & set(document_weights))
    contributions = {
        term: query_weights[term] * document_weights[term] for term in shared
    }
    numerator = math.fsum(contributions.values())
    query_norm = sparse_norm(query_weights)
    document_norm = sparse_norm(document_weights)
    denominator = query_norm * document_norm
    score = 0.0 if denominator == 0.0 else numerator / denominator
    if not math.isfinite(score):
        raise ValueError("TF-IDF cosine score became non-finite.")
    return score, numerator, query_norm, document_norm, contributions
