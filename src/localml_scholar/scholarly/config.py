"""Immutable configuration for deterministic scholarly analysis."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from localml_scholar.retrieval.documents import canonical_json


@dataclass(frozen=True)
class ScholarlyConfig:
    """Transparent rules controlling conservative paper extraction."""

    definition_window_characters: int = 800
    equation_minimum_operator_count: int = 1
    parse_markdown_tables: bool = True
    parse_delimited_tables: bool = True
    include_inferred_assumptions: bool = False
    equation_aware_retrieval_weight: float = 0.35
    minimum_equation_line_symbol_count: int = 2
    risk_flag_missing_seed: bool = True
    risk_flag_missing_hardware: bool = True
    research_gap_from_missing_ablation: bool = True

    def __post_init__(self) -> None:
        integer_fields = (
            "definition_window_characters",
            "equation_minimum_operator_count",
            "minimum_equation_line_symbol_count",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in (
            "parse_markdown_tables",
            "parse_delimited_tables",
            "include_inferred_assumptions",
            "risk_flag_missing_seed",
            "risk_flag_missing_hardware",
            "research_gap_from_missing_ablation",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")
        weight = self.equation_aware_retrieval_weight
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError("equation_aware_retrieval_weight must be a real number.")
        if not 0.0 <= float(weight) <= 1.0:
            raise ValueError("equation_aware_retrieval_weight must lie in [0, 1].")
        object.__setattr__(self, "equation_aware_retrieval_weight", float(weight))

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible configuration state."""
        return dict(vars(self))

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> ScholarlyConfig:
        """Reconstruct a strictly shaped configuration."""
        if not isinstance(state, Mapping) or set(state) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("Scholarly configuration is malformed.")
        return cls(**dict(state))

    def state_hash(self) -> str:
        """Hash the exact extraction policy."""
        return hashlib.sha256(
            canonical_json(self.to_dict()).encode("utf-8")
        ).hexdigest()
