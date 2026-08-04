"""Review lineage and circular self-training safeguards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def content_sha256(value: object) -> str:
    """Hash one JSON-compatible value using a stable canonical encoding."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value: object, name: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must contain non-whitespace text.")
    return value.strip()


@dataclass(frozen=True)
class ReviewProvenance:
    """Immutable producer, reviewer, correction, source, and ancestry identity."""

    producer_system: str
    producer_version: str
    reviewer_system: str
    reviewer_version: str
    correction_system: str | None
    source_hashes: tuple[str, ...]
    answer_hash: str
    parent_example_ids: tuple[str, ...] = ()
    benchmark_source: str | None = None
    independent_validators: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "producer_system",
            "producer_version",
            "reviewer_system",
            "reviewer_version",
            "answer_hash",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("correction_system", "benchmark_source"):
            value = getattr(self, name)
            object.__setattr__(self, name, _text(value, name, allow_none=True))
        for name in (
            "source_hashes",
            "parent_example_ids",
            "independent_validators",
        ):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                raise TypeError(f"{name} must be a sequence of non-empty strings.")
            cleaned = tuple(item.strip() for item in value)
            if name == "parent_example_ids" and len(set(cleaned)) != len(cleaned):
                raise ValueError("parent_example_ids must not contain duplicates.")
            object.__setattr__(self, name, tuple(dict.fromkeys(cleaned)))
        if not self.source_hashes:
            raise ValueError("source_hashes must contain at least one source identity.")

    @property
    def circular_warnings(self) -> tuple[str, ...]:
        """Return warnings that prevent same-system approval from becoming gold."""
        warnings = []
        if (
            self.producer_system == self.reviewer_system
            and self.producer_version == self.reviewer_version
            and not self.independent_validators
        ):
            warnings.append("same_producer_and_reviewer_without_independent_validation")
        if self.correction_system is not None and self.correction_system in {
            self.producer_system,
            self.reviewer_system,
        }:
            warnings.append("correction_system_reuses_producer_or_reviewer")
        return tuple(warnings)

    def validate_source_hashes(self, current_hashes: tuple[str, ...]) -> None:
        """Reject review reuse against a different source snapshot."""
        if not isinstance(current_hashes, tuple):
            raise TypeError("current_hashes must be a tuple.")
        if tuple(sorted(current_hashes)) != tuple(sorted(self.source_hashes)):
            raise ValueError(
                "Review provenance source hashes do not match current sources."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer_system": self.producer_system,
            "producer_version": self.producer_version,
            "reviewer_system": self.reviewer_system,
            "reviewer_version": self.reviewer_version,
            "correction_system": self.correction_system,
            "source_hashes": list(self.source_hashes),
            "answer_hash": self.answer_hash,
            "parent_example_ids": list(self.parent_example_ids),
            "benchmark_source": self.benchmark_source,
            "independent_validators": list(self.independent_validators),
            "circular_warnings": list(self.circular_warnings),
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> ReviewProvenance:
        if not isinstance(state, dict):
            raise TypeError("ReviewProvenance JSON must be an object.")
        return cls(
            producer_system=state["producer_system"],
            producer_version=state["producer_version"],
            reviewer_system=state["reviewer_system"],
            reviewer_version=state["reviewer_version"],
            correction_system=state.get("correction_system"),
            source_hashes=tuple(state["source_hashes"]),
            answer_hash=state["answer_hash"],
            parent_example_ids=tuple(state.get("parent_example_ids", [])),
            benchmark_source=state.get("benchmark_source"),
            independent_validators=tuple(state.get("independent_validators", [])),
        )
