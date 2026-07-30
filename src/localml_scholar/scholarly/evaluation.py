"""Transparent metrics for authored scholarly extraction judgments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExtractionMetrics:
    """Set extraction precision/recall/F1 plus exact-value accuracy."""

    precision: float
    recall: float
    f1: float
    exact_value_accuracy: float
    true_positive: int
    false_positive: int
    false_negative: int

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


def extraction_metrics(
    predicted: Iterable[str],
    expected: Iterable[str],
) -> ExtractionMetrics:
    """Compare normalized unique string fields with explicit empty-set semantics."""
    predicted_set = set(predicted)
    expected_set = set(expected)
    if not all(isinstance(item, str) for item in predicted_set | expected_set):
        raise TypeError("Extraction metrics require string values.")
    true_positive = len(predicted_set & expected_set)
    false_positive = len(predicted_set - expected_set)
    false_negative = len(expected_set - predicted_set)
    precision = (
        1.0
        if not predicted_set and not expected_set
        else true_positive / max(1, len(predicted_set))
    )
    recall = 1.0 if not expected_set else true_positive / len(expected_set)
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return ExtractionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        exact_value_accuracy=float(predicted_set == expected_set),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def classification_accuracy(
    predicted: Mapping[str, str],
    expected: Mapping[str, str],
) -> float:
    """Return exact classification accuracy over expected keys."""
    if not expected:
        return 1.0 if not predicted else 0.0
    return sum(predicted.get(key) == value for key, value in expected.items()) / len(
        expected
    )


def citation_coverage(records: Iterable[Mapping[str, Any]]) -> float:
    """Measure serialized records containing at least one citation."""
    materialized = tuple(records)
    if not materialized:
        return 1.0
    return sum(bool(_find_citations(item)) for item in materialized) / len(materialized)


def _find_citations(value: object) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if {
            "document_id",
            "start_character",
            "end_character",
            "source_text_sha256",
        } <= set(value):
            found.append(value)
        for item in value.values():
            found.extend(_find_citations(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_find_citations(item))
    return found
