"""Validated deterministic exact-ID retrieval metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _validate_ids(retrieved: Sequence[str], relevant: set[str], k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if isinstance(retrieved, (str, bytes)) or not all(
        isinstance(value, str) and value for value in retrieved
    ):
        raise ValueError("retrieved must contain non-empty string IDs.")
    if len(retrieved) != len(set(retrieved)):
        raise ValueError("retrieved IDs must not contain duplicates.")
    if not isinstance(relevant, set) or not all(
        isinstance(value, str) and value for value in relevant
    ):
        raise ValueError("relevant must be a set of non-empty string IDs.")


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Return relevant results in the first k positions divided by k."""
    _validate_ids(retrieved, relevant, k)
    return len(set(retrieved[:k]) & relevant) / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Return relevant results found in the first k divided by all relevant."""
    _validate_ids(retrieved, relevant, k)
    if not relevant:
        raise ValueError("Recall is undefined for a query with no relevant IDs.")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Return inverse rank of the first relevant result, or zero."""
    _validate_ids(retrieved, relevant, 1)
    return next(
        (
            1.0 / rank
            for rank, result in enumerate(retrieved, start=1)
            if result in relevant
        ),
        0.0,
    )


def hit_rate_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Return one when a relevant ID occurs in the first k, else zero."""
    _validate_ids(retrieved, relevant, k)
    return float(bool(set(retrieved[:k]) & relevant))


@dataclass(frozen=True)
class QueryMetrics:
    precision_at_1: float
    precision_at_3: float
    recall_at_3: float
    reciprocal_rank: float
    hit_rate_at_3: float

    def to_dict(self) -> dict[str, float]:
        return dict(vars(self))


@dataclass(frozen=True)
class RetrievalEvaluation:
    per_query: dict[str, QueryMetrics]
    aggregate: QueryMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_query": {
                query: metrics.to_dict()
                for query, metrics in sorted(self.per_query.items())
            },
            "aggregate": self.aggregate.to_dict(),
        }


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    relevance: Mapping[str, Sequence[str]],
    *,
    valid_chunk_ids: set[str] | None = None,
) -> RetrievalEvaluation:
    """Evaluate exact chunk-ID rankings and return per-query/mean metrics."""
    if not isinstance(rankings, Mapping) or not isinstance(relevance, Mapping):
        raise TypeError("rankings and relevance must be mappings.")
    if set(rankings) != set(relevance) or not rankings:
        raise ValueError("rankings and relevance must contain the same queries.")
    per_query: dict[str, QueryMetrics] = {}
    for query in sorted(rankings):
        if not isinstance(query, str) or not query:
            raise ValueError("Evaluation query labels must be non-empty strings.")
        relevant_values = relevance[query]
        if isinstance(relevant_values, (str, bytes)):
            raise TypeError("Relevance values must be ID sequences.")
        relevant_set = set(relevant_values)
        if len(relevant_set) != len(relevant_values):
            raise ValueError("Relevance IDs must not contain duplicates.")
        if not relevant_set:
            raise ValueError("Every evaluated query must have a relevant chunk.")
        if valid_chunk_ids is not None and not relevant_set <= valid_chunk_ids:
            raise ValueError("Relevance contains an unknown chunk ID.")
        retrieved = list(rankings[query])
        per_query[query] = QueryMetrics(
            precision_at_1=precision_at_k(retrieved, relevant_set, 1),
            precision_at_3=precision_at_k(retrieved, relevant_set, 3),
            recall_at_3=recall_at_k(retrieved, relevant_set, 3),
            reciprocal_rank=reciprocal_rank(retrieved, relevant_set),
            hit_rate_at_3=hit_rate_at_k(retrieved, relevant_set, 3),
        )
    count = len(per_query)
    aggregate = QueryMetrics(
        **{
            field: math.fsum(getattr(value, field) for value in per_query.values())
            / count
            for field in QueryMetrics.__dataclass_fields__
        }
    )
    return RetrievalEvaluation(per_query=per_query, aggregate=aggregate)
