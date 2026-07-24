"""Independent BM25 configuration, IDF, and term-contribution formulas."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BM25Config:
    """BM25 saturation and document-length normalization settings."""

    k1: float = 1.2
    b: float = 0.75

    def __post_init__(self) -> None:
        for name in ("k1", "b"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if self.k1 <= 0.0:
            raise ValueError("k1 must be positive.")
        if not 0.0 <= self.b <= 1.0:
            raise ValueError("b must lie in [0, 1].")
        object.__setattr__(self, "k1", float(self.k1))
        object.__setattr__(self, "b", float(self.b))

    def to_dict(self) -> dict[str, float]:
        return {"k1": self.k1, "b": self.b}

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> BM25Config:
        if not isinstance(state, dict) or set(state) != {"k1", "b"}:
            raise ValueError("BM25 configuration is malformed.")
        return cls(**state)


def bm25_inverse_document_frequency(
    number_of_chunks: int,
    document_frequency: int,
) -> float:
    """Return ``log(1 + (N - df + 0.5)/(df + 0.5))``."""
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (number_of_chunks, document_frequency)
    ):
        raise TypeError("N and df must be integers.")
    if number_of_chunks <= 0:
        raise ValueError("number_of_chunks must be positive.")
    if not 0 <= document_frequency <= number_of_chunks:
        raise ValueError("document_frequency must lie in [0, N].")
    return math.log(
        1.0 + (number_of_chunks - document_frequency + 0.5) / (document_frequency + 0.5)
    )


def bm25_term_contribution(
    *,
    term_frequency: int,
    document_frequency: int,
    document_length: int,
    average_document_length: float,
    number_of_chunks: int,
    config: BM25Config,
) -> tuple[float, float, float]:
    """Return contribution, IDF, and length-normalization factor."""
    if isinstance(term_frequency, bool) or not isinstance(term_frequency, int):
        raise TypeError("term_frequency must be an integer.")
    if term_frequency < 0:
        raise ValueError("term_frequency must be non-negative.")
    if isinstance(document_length, bool) or not isinstance(document_length, int):
        raise TypeError("document_length must be an integer.")
    if document_length < 0:
        raise ValueError("document_length must be non-negative.")
    if not isinstance(config, BM25Config):
        raise TypeError("config must be BM25Config.")
    if not math.isfinite(average_document_length) or average_document_length <= 0:
        raise ValueError("average_document_length must be finite and positive.")
    inverse_document_frequency = bm25_inverse_document_frequency(
        number_of_chunks,
        document_frequency,
    )
    length_normalization = config.k1 * (
        1.0 - config.b + config.b * document_length / average_document_length
    )
    denominator = term_frequency + length_normalization
    contribution = (
        0.0
        if term_frequency == 0
        else inverse_document_frequency
        * term_frequency
        * (config.k1 + 1.0)
        / denominator
    )
    if not all(
        math.isfinite(value)
        for value in (
            inverse_document_frequency,
            length_normalization,
            contribution,
        )
    ):
        raise ValueError("BM25 computation became non-finite.")
    return contribution, inverse_document_frequency, length_normalization
