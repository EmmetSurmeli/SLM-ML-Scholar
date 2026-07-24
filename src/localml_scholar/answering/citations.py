"""Machine-parseable inline citation labels for grounded answers."""

from __future__ import annotations

import re
from dataclasses import dataclass

_LABEL = r"C[1-9]\d*"
_CITATION_PATTERN = re.compile(rf"\[(?P<labels>{_LABEL}(?:\s*,\s*{_LABEL})*)\]")
_CITATION_LIKE_PATTERN = re.compile(r"\[[^\]\n]*C\d*[^\]\n]*\]|\[\s*C\d*")


@dataclass(frozen=True)
class CitationOccurrence:
    """One valid citation group and its exact answer-text span."""

    labels: tuple[str, ...]
    start_character: int
    end_character: int
    raw_text: str


class CitationSyntaxError(ValueError):
    """Raised when answer text contains malformed citation-like syntax."""


def parse_inline_citations(
    text: str,
    *,
    strict: bool = True,
) -> tuple[CitationOccurrence, ...]:
    """Parse ``[C1]`` and ``[C1, C3]`` groups in deterministic source order."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not isinstance(strict, bool):
        raise TypeError("strict must be boolean.")
    occurrences: list[CitationOccurrence] = []
    valid_spans: list[tuple[int, int]] = []
    for match in _CITATION_PATTERN.finditer(text):
        labels = tuple(dict.fromkeys(re.split(r"\s*,\s*", match.group("labels"))))
        occurrences.append(
            CitationOccurrence(
                labels=labels,
                start_character=match.start(),
                end_character=match.end(),
                raw_text=match.group(0),
            )
        )
        valid_spans.append(match.span())
    if strict:
        for candidate in _CITATION_LIKE_PATTERN.finditer(text):
            if not any(
                start <= candidate.start() and candidate.end() <= end
                for start, end in valid_spans
            ):
                raise CitationSyntaxError(
                    "Malformed inline citation syntax at character "
                    f"{candidate.start()}: {candidate.group(0)!r}."
                )
    return tuple(occurrences)


def citation_labels(text: str, *, strict: bool = True) -> tuple[str, ...]:
    """Return unique labels in first-occurrence order."""
    return tuple(
        dict.fromkeys(
            label
            for occurrence in parse_inline_citations(text, strict=strict)
            for label in occurrence.labels
        )
    )


def strip_inline_citations(text: str) -> str:
    """Remove valid citation groups without rewriting any other content."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    return _CITATION_PATTERN.sub("", text)


def format_inline_citation(labels: tuple[str, ...]) -> str:
    """Format normalized labels as one deterministic inline group."""
    if not isinstance(labels, tuple) or not labels:
        raise ValueError("labels must be a non-empty tuple.")
    normalized = tuple(dict.fromkeys(labels))
    if not all(re.fullmatch(_LABEL, label) for label in normalized):
        raise ValueError("Every citation label must use the form C1, C2, ....")
    return "[" + ", ".join(normalized) + "]"
