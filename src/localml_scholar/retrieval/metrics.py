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


def average_precision(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Return mean precision at each retrieved relevant rank."""
    _validate_ids(retrieved, relevant, 1)
    if not relevant:
        raise ValueError("Average precision is undefined without relevant IDs.")
    hits = 0
    precisions: list[float] = []
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            hits += 1
            precisions.append(hits / rank)
    return math.fsum(precisions) / len(relevant)


def ndcg_at_k(
    retrieved: Sequence[str],
    relevance_grades: Mapping[str, int],
    k: int,
) -> float:
    """Return normalized discounted cumulative gain using ``2**grade - 1``."""
    if not isinstance(relevance_grades, Mapping) or any(
        not isinstance(chunk_id, str)
        or not chunk_id
        or isinstance(grade, bool)
        or not isinstance(grade, int)
        or grade < 0
        for chunk_id, grade in relevance_grades.items()
    ):
        raise ValueError("Relevance grades must map IDs to non-negative integers.")
    relevant = set(relevance_grades)
    _validate_ids(retrieved, relevant, k)

    def discounted_gain(grades: Sequence[int]) -> float:
        return math.fsum(
            (2.0**grade - 1.0) / math.log2(rank + 1.0)
            for rank, grade in enumerate(grades, start=1)
        )

    observed = [relevance_grades.get(chunk_id, 0) for chunk_id in retrieved[:k]]
    ideal = sorted(relevance_grades.values(), reverse=True)[:k]
    ideal_gain = discounted_gain(ideal)
    return 0.0 if ideal_gain == 0.0 else discounted_gain(observed) / ideal_gain


def mean_relevant_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Mean relevant rank, assigning missing relevant IDs ``len(results) + 1``."""
    _validate_ids(retrieved, relevant, 1)
    if not relevant:
        raise ValueError("Mean relevant rank is undefined without relevant IDs.")
    ranks = {chunk_id: rank for rank, chunk_id in enumerate(retrieved, start=1)}
    missing_rank = len(retrieved) + 1
    return math.fsum(ranks.get(chunk_id, missing_rank) for chunk_id in relevant) / len(
        relevant
    )


@dataclass(frozen=True)
class QueryMetrics:
    precision_at_1: float
    precision_at_3: float
    recall_at_3: float
    recall_at_5: float
    reciprocal_rank: float
    hit_rate_at_3: float
    average_precision: float
    ndcg_at_3: float
    ndcg_at_5: float
    mean_relevant_rank: float
    no_relevant_result: float

    def to_dict(self) -> dict[str, float]:
        return dict(vars(self))


@dataclass(frozen=True)
class RetrievalEvaluation:
    per_query: dict[str, QueryMetrics]
    aggregate: QueryMetrics
    per_category: dict[str, QueryMetrics]
    aggregation_policy: str = "macro_average_over_queries"

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_query": {
                query: metrics.to_dict()
                for query, metrics in sorted(self.per_query.items())
            },
            "aggregate": self.aggregate.to_dict(),
            "per_category": {
                category: metrics.to_dict()
                for category, metrics in sorted(self.per_category.items())
            },
            "aggregation_policy": self.aggregation_policy,
        }


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    relevance: Mapping[str, Sequence[str]],
    *,
    valid_chunk_ids: set[str] | None = None,
    grades: Mapping[str, Mapping[str, int]] | None = None,
    categories: Mapping[str, str] | None = None,
) -> RetrievalEvaluation:
    """Evaluate exact chunk-ID rankings and return per-query/mean metrics."""
    if not isinstance(rankings, Mapping) or not isinstance(relevance, Mapping):
        raise TypeError("rankings and relevance must be mappings.")
    if set(rankings) != set(relevance) or not rankings:
        raise ValueError("rankings and relevance must contain the same queries.")
    if grades is not None and set(grades) != set(rankings):
        raise ValueError("grades must contain the same queries as rankings.")
    if categories is not None and set(categories) != set(rankings):
        raise ValueError("categories must contain the same queries as rankings.")
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
        query_grades = (
            {chunk_id: 1 for chunk_id in relevant_set}
            if grades is None
            else dict(grades[query])
        )
        if set(query_grades) != relevant_set or any(
            isinstance(grade, bool) or not isinstance(grade, int) or grade <= 0
            for grade in query_grades.values()
        ):
            raise ValueError(
                "Graded relevance must cover exact relevant IDs with positive grades."
            )
        per_query[query] = QueryMetrics(
            precision_at_1=precision_at_k(retrieved, relevant_set, 1),
            precision_at_3=precision_at_k(retrieved, relevant_set, 3),
            recall_at_3=recall_at_k(retrieved, relevant_set, 3),
            recall_at_5=recall_at_k(retrieved, relevant_set, 5),
            reciprocal_rank=reciprocal_rank(retrieved, relevant_set),
            hit_rate_at_3=hit_rate_at_k(retrieved, relevant_set, 3),
            average_precision=average_precision(retrieved, relevant_set),
            ndcg_at_3=ndcg_at_k(retrieved, query_grades, 3),
            ndcg_at_5=ndcg_at_k(retrieved, query_grades, 5),
            mean_relevant_rank=mean_relevant_rank(retrieved, relevant_set),
            no_relevant_result=float(not (set(retrieved) & relevant_set)),
        )

    def aggregate_metrics(values: Sequence[QueryMetrics]) -> QueryMetrics:
        count = len(values)
        return QueryMetrics(
            **{
                field: math.fsum(getattr(value, field) for value in values) / count
                for field in QueryMetrics.__dataclass_fields__
            }
        )

    category_metrics: dict[str, QueryMetrics] = {}
    if categories is not None:
        for category in sorted(set(categories.values())):
            if not isinstance(category, str) or not category:
                raise ValueError("Category labels must be non-empty strings.")
            values = [
                per_query[query]
                for query in sorted(per_query)
                if categories[query] == category
            ]
            category_metrics[category] = aggregate_metrics(values)
    return RetrievalEvaluation(
        per_query=per_query,
        aggregate=aggregate_metrics(list(per_query.values())),
        per_category=category_metrics,
    )
