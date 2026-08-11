"""Canonical deterministic question concepts shared by retrieval and curation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from localml_scholar.retrieval.text import normalize_query_terms, tokenize_lexically

_INTENT_TERMS = frozenset(
    {
        "describe",
        "describes",
        "establish",
        "explain",
        "identify",
        "introduce",
        "list",
        "long",
        "main",
        "most",
        "paper",
        "performed",
        "prior",
        "proposed",
        "report",
        "reported",
        "result",
        "show",
        "shows",
        "state",
        "stated",
        "step",
        "steps",
        "strongly",
        "support",
        "supports",
        "take",
        "use",
        "used",
        "uses",
    }
)


@dataclass(frozen=True)
class QuestionConcepts:
    """Essential and optional concepts for one question."""

    essential_terms: tuple[str, ...]
    optional_terms: tuple[str, ...]
    entities: tuple[str, ...]
    numbers: tuple[str, ...]
    question_intent: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "essential_terms": list(self.essential_terms),
            "optional_terms": list(self.optional_terms),
            "entities": list(self.entities),
            "numbers": list(self.numbers),
            "question_intent": self.question_intent,
        }


def question_concepts(question: str) -> QuestionConcepts:
    """Extract stable essential concepts without function-word pollution."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must contain non-whitespace text.")
    normalized = normalize_query_terms(question)
    optional = tuple(term for term in normalized if term in _INTENT_TERMS)
    essential = tuple(term for term in normalized if term not in _INTENT_TERMS)
    if not essential:
        essential, optional = normalized, ()
    entities = tuple(
        dict.fromkeys(
            token.casefold()
            for token in re.findall(r"\b[A-Z][A-Za-z0-9+.-]*\b", question)
            if token.casefold() not in {"what", "which", "how", "why"}
        )
    )
    numbers = tuple(dict.fromkeys(re.findall(r"\b\d+(?:\.\d+)?\b", question)))
    raw = set(tokenize_lexically(question))
    intent = (
        "ablation"
        if {"ablation", "ablated", "ablate"} & set(normalized)
        else "mechanism"
        if {"how", "mechanism", "work"} & raw
        else "numerical"
        if numbers
        else "lookup"
    )
    return QuestionConcepts(essential, optional, entities, numbers, intent)
