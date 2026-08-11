"""Transparent retrieval-only lexical normalization and term spans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_TERM_PATTERN = re.compile(
    r"[^\W\d_]+(?:['’][^\W\d_]+)+|[^\W\d_]\w*|\d+(?:\.\d+)?",
    flags=re.UNICODE,
)

# Query-only normalization. Document indexing intentionally keeps every lexical
# token so phrase/source diagnostics remain lossless; queries and all downstream
# coverage checks share this exact stopword policy.
QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "approaches",
        "are",
        "as",
        "at",
        "authors",
        "be",
        "been",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "identify",
        "in",
        "is",
        "it",
        "its",
        "long",
        "main",
        "most",
        "of",
        "on",
        "or",
        "paper",
        "paper's",
        "prior",
        "proposed",
        "reason",
        "section",
        "simply",
        "state",
        "step",
        "steps",
        "strongest",
        "support",
        "supports",
        "that",
        "the",
        "their",
        "this",
        "to",
        "take",
        "use",
        "used",
        "uses",
        "using",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
    }
)

_QUERY_ALIASES = {
    "idea": "method",
    "skeptical": "limitations",
}


@dataclass(frozen=True)
class LexicalTokenizerConfig:
    """Explicit retrieval normalization, separate from LM tokenization."""

    casefold: bool = True
    normalization: str = "none"
    split_hyphens: bool = True
    preserve_apostrophes: bool = True
    preserve_underscores: bool = True
    split_camel_case: bool = False

    def __post_init__(self) -> None:
        for name in (
            "casefold",
            "split_hyphens",
            "preserve_apostrophes",
            "preserve_underscores",
            "split_camel_case",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")
        if self.normalization != "none":
            raise ValueError("Only retrieval normalization='none' is supported.")
        if not self.split_hyphens:
            raise ValueError("The reference lexical tokenizer always splits hyphens.")
        if not self.preserve_apostrophes or not self.preserve_underscores:
            raise ValueError(
                "The reference lexical tokenizer preserves apostrophes/underscores."
            )
        if self.split_camel_case:
            raise ValueError("Camel-case splitting is not implemented.")

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> LexicalTokenizerConfig:
        if not isinstance(state, dict) or set(state) != set(cls.__dataclass_fields__):
            raise ValueError("Lexical tokenizer configuration is malformed.")
        return cls(**state)


@dataclass(frozen=True)
class LexicalTerm:
    """One normalized term and its exact span in original text."""

    term: str
    start_character: int
    end_character: int
    position: int
    original: str


def lexical_terms(
    text: str,
    config: LexicalTokenizerConfig | None = None,
) -> tuple[LexicalTerm, ...]:
    """Return deterministic Unicode-aware word/code terms with source spans."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    resolved = config or LexicalTokenizerConfig()
    if not isinstance(resolved, LexicalTokenizerConfig):
        raise TypeError("config must be LexicalTokenizerConfig.")
    terms: list[LexicalTerm] = []
    for match in _TERM_PATTERN.finditer(text):
        original = match.group(0)
        normalized = original.casefold() if resolved.casefold else original
        terms.append(
            LexicalTerm(
                term=normalized,
                start_character=match.start(),
                end_character=match.end(),
                position=len(terms),
                original=original,
            )
        )
    return tuple(terms)


def tokenize_lexically(
    text: str,
    config: LexicalTokenizerConfig | None = None,
) -> tuple[str, ...]:
    """Return normalized retrieval terms only."""
    return tuple(term.term for term in lexical_terms(text, config))


def normalize_query_terms(
    text: str,
    config: LexicalTokenizerConfig | None = None,
) -> tuple[str, ...]:
    """Return unique non-stop query terms using the canonical lexical policy.

    If a query consists entirely of stopwords, unique lexical terms are retained
    so callers can produce an explicit low-information diagnostic rather than an
    invalid empty query.
    """
    terms = tokenize_lexically(text, config)
    unique = tuple(dict.fromkeys(_QUERY_ALIASES.get(term, term) for term in terms))
    meaningful = tuple(term for term in unique if term not in QUERY_STOPWORDS)
    return meaningful or unique
