"""Validated immutable models for evidence-grounded local answers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from localml_scholar.retrieval.documents import (
    Citation,
    canonical_json,
    sha256_text,
    stable_identifier,
)

_LABEL_PATTERN = re.compile(r"C[1-9]\d*")
_ANSWER_METHODS = {
    "top_passage",
    "extractive",
    "generative",
    "generative_with_extractive_fallback",
}


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _optional_nonempty_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, name)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1].")
    return normalized


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{name} must be a tuple of non-empty strings.")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class EvidenceItem:
    """One selected exact source slice bound to one retrieval result."""

    evidence_id: str
    label: str
    chunk_id: str
    document_id: str
    source_name: str
    title: str | None
    heading_path: tuple[str, ...]
    selected_text: str
    start_character: int
    end_character: int
    start_line: int
    end_line: int
    page_start: int | None
    page_end: int | None
    citation: Citation
    retrieval_rank: int
    retrieval_score: float
    retrieval_method: str
    matched_terms: tuple[str, ...]
    token_count: int | None
    character_count: int
    truncated: bool
    selected_text_sha256: str
    index_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "evidence_id",
            "chunk_id",
            "document_id",
            "source_name",
        ):
            _nonempty_string(getattr(self, name), name)
        if not _LABEL_PATTERN.fullmatch(self.label):
            raise ValueError("Evidence label must use the form C1, C2, ....")
        _optional_nonempty_string(self.title, "title")
        _string_tuple(self.heading_path, "heading_path")
        _nonempty_string(self.selected_text, "selected_text")
        for name in (
            "start_character",
            "end_character",
            "start_line",
            "end_line",
            "retrieval_rank",
            "character_count",
        ):
            _nonnegative_integer(getattr(self, name), name)
        if self.start_character >= self.end_character:
            raise ValueError(
                "Evidence character offsets must be non-empty and ordered."
            )
        if self.start_line < 1 or self.start_line > self.end_line:
            raise ValueError("Evidence line ranges must be positive and ordered.")
        if self.retrieval_rank < 1:
            raise ValueError("retrieval_rank starts at one.")
        if self.character_count != len(self.selected_text):
            raise ValueError("character_count must equal the selected text length.")
        if self.token_count is not None:
            _nonnegative_integer(self.token_count, "token_count")
        if isinstance(self.retrieval_score, bool) or not isinstance(
            self.retrieval_score, (int, float)
        ):
            raise TypeError("retrieval_score must be a real number.")
        if not math.isfinite(float(self.retrieval_score)) or self.retrieval_score < 0.0:
            raise ValueError("retrieval_score must be finite and non-negative.")
        if self.retrieval_method not in {"tfidf", "bm25"}:
            raise ValueError("retrieval_method must be 'tfidf' or 'bm25'.")
        _string_tuple(self.matched_terms, "matched_terms")
        if not isinstance(self.truncated, bool):
            raise TypeError("truncated must be boolean.")
        if self.selected_text_sha256 != sha256_text(self.selected_text):
            raise ValueError("selected_text_sha256 does not match selected_text.")
        if not _is_sha256(self.index_sha256):
            raise ValueError("index_sha256 must be a lowercase SHA-256 digest.")
        if self.citation.chunk_id != self.chunk_id:
            raise ValueError("Evidence citation must link to the exact chunk.")
        if self.citation.document_id != self.document_id:
            raise ValueError("Evidence citation must link to the exact document.")
        if (
            self.citation.source_name != self.source_name
            or self.citation.title != self.title
            or self.citation.heading_path != self.heading_path
        ):
            raise ValueError("Evidence citation metadata must match the evidence.")
        if (
            self.citation.start_line != self.start_line
            or self.citation.end_line != self.end_line
            or self.citation.page_start != self.page_start
            or self.citation.page_end != self.page_end
        ):
            raise ValueError(
                "Evidence citation location must match the selected slice."
            )

    @classmethod
    def create(
        cls,
        *,
        label: str,
        chunk_id: str,
        document_id: str,
        source_name: str,
        title: str | None,
        heading_path: tuple[str, ...],
        selected_text: str,
        start_character: int,
        end_character: int,
        start_line: int,
        end_line: int,
        page_start: int | None,
        page_end: int | None,
        retrieval_rank: int,
        retrieval_score: float,
        retrieval_method: str,
        matched_terms: tuple[str, ...],
        token_count: int | None,
        truncated: bool,
        index_sha256: str,
    ) -> EvidenceItem:
        """Construct deterministic evidence and citation identities."""
        citation = Citation(
            document_id=document_id,
            source_name=source_name,
            title=title,
            heading_path=heading_path,
            page_start=page_start,
            page_end=page_end,
            start_line=start_line,
            end_line=end_line,
            chunk_id=chunk_id,
        )
        evidence_id = stable_identifier(
            "ev",
            index_sha256,
            chunk_id,
            start_character,
            end_character,
            sha256_text(selected_text),
        )
        return cls(
            evidence_id=evidence_id,
            label=label,
            chunk_id=chunk_id,
            document_id=document_id,
            source_name=source_name,
            title=title,
            heading_path=heading_path,
            selected_text=selected_text,
            start_character=start_character,
            end_character=end_character,
            start_line=start_line,
            end_line=end_line,
            page_start=page_start,
            page_end=page_end,
            citation=citation,
            retrieval_rank=retrieval_rank,
            retrieval_score=float(retrieval_score),
            retrieval_method=retrieval_method,
            matched_terms=matched_terms,
            token_count=token_count,
            character_count=len(selected_text),
            truncated=truncated,
            selected_text_sha256=sha256_text(selected_text),
            index_sha256=index_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        state = dict(vars(self))
        state["heading_path"] = list(self.heading_path)
        state["matched_terms"] = list(self.matched_terms)
        state["citation"] = self.citation.to_dict()
        return state

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> EvidenceItem:
        expected = set(cls.__dataclass_fields__)
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Evidence item state keys are malformed.")
        values = dict(state)
        for name in ("heading_path", "matched_terms"):
            if not isinstance(values[name], list):
                raise ValueError(f"Serialized {name} must be a list.")
            values[name] = tuple(values[name])
        values["citation"] = Citation.from_dict(values["citation"])
        return cls(**values)


@dataclass(frozen=True)
class CitationBinding:
    """A deterministic answer-local label bound to structured evidence."""

    label: str
    evidence_id: str
    citation: Citation

    def __post_init__(self) -> None:
        if not _LABEL_PATTERN.fullmatch(self.label):
            raise ValueError("Citation label must use the form C1, C2, ....")
        _nonempty_string(self.evidence_id, "evidence_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "evidence_id": self.evidence_id,
            "citation": self.citation.to_dict(),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> CitationBinding:
        if not isinstance(state, Mapping) or set(state) != {
            "label",
            "evidence_id",
            "citation",
        }:
            raise ValueError("Citation binding state keys are malformed.")
        return cls(
            label=state["label"],
            evidence_id=state["evidence_id"],
            citation=Citation.from_dict(state["citation"]),
        )


@dataclass(frozen=True)
class EvidenceSufficiency:
    """Transparent lexical heuristic controlling answer attempts."""

    sufficient: bool
    score: float
    reasons: tuple[str, ...]
    matched_query_terms: tuple[str, ...]
    unmatched_query_terms: tuple[str, ...]
    evidence_count: int
    unique_source_count: int
    query_term_coverage: float
    top_retrieval_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.sufficient, bool):
            raise TypeError("sufficient must be boolean.")
        _fraction(self.score, "score")
        _string_tuple(self.reasons, "reasons")
        for name in ("matched_query_terms", "unmatched_query_terms"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not all(
                isinstance(item, str) and item for item in value
            ):
                raise ValueError(f"{name} must contain non-empty strings.")
        _nonnegative_integer(self.evidence_count, "evidence_count")
        _nonnegative_integer(self.unique_source_count, "unique_source_count")
        _fraction(self.query_term_coverage, "query_term_coverage")
        if (
            not math.isfinite(self.top_retrieval_score)
            or self.top_retrieval_score < 0.0
        ):
            raise ValueError("top_retrieval_score must be finite and non-negative.")

    def to_dict(self) -> dict[str, Any]:
        state = dict(vars(self))
        for name in ("reasons", "matched_query_terms", "unmatched_query_terms"):
            state[name] = list(state[name])
        return state

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> EvidenceSufficiency:
        if not isinstance(state, Mapping) or set(state) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("Evidence sufficiency state keys are malformed.")
        values = dict(state)
        for name in ("reasons", "matched_query_terms", "unmatched_query_terms"):
            if not isinstance(values[name], list):
                raise ValueError(f"Serialized {name} must be a list.")
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class ClaimSupport:
    """Conservative lexical support signals; this is not entailment."""

    supported: bool
    score: float
    reasons: tuple[str, ...]
    overlapping_terms: tuple[str, ...]
    missing_key_terms: tuple[str, ...]
    number_mismatches: tuple[str, ...]
    equation_symbol_mismatches: tuple[str, ...]
    identifier_mismatches: tuple[str, ...]
    negation_warning: bool
    exact_phrase_overlap: bool
    exact_quoted_characters: int
    longest_exact_quote: int
    quote_evidence_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.supported, bool):
            raise TypeError("supported must be boolean.")
        _fraction(self.score, "score")
        for name in (
            "reasons",
            "overlapping_terms",
            "missing_key_terms",
            "number_mismatches",
            "equation_symbol_mismatches",
            "identifier_mismatches",
        ):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not all(
                isinstance(item, str) and item for item in value
            ):
                raise ValueError(f"{name} must contain non-empty strings.")
        if not isinstance(self.negation_warning, bool) or not isinstance(
            self.exact_phrase_overlap, bool
        ):
            raise TypeError("Claim warning flags must be boolean.")
        _nonnegative_integer(self.exact_quoted_characters, "exact_quoted_characters")
        _nonnegative_integer(self.longest_exact_quote, "longest_exact_quote")
        _optional_nonempty_string(self.quote_evidence_id, "quote_evidence_id")

    def to_dict(self) -> dict[str, Any]:
        state = dict(vars(self))
        for name in (
            "reasons",
            "overlapping_terms",
            "missing_key_terms",
            "number_mismatches",
            "equation_symbol_mismatches",
            "identifier_mismatches",
        ):
            state[name] = list(state[name])
        return state

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> ClaimSupport:
        if not isinstance(state, Mapping) or set(state) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("Claim support state keys are malformed.")
        values = dict(state)
        for name in (
            "reasons",
            "overlapping_terms",
            "missing_key_terms",
            "number_mismatches",
            "equation_symbol_mismatches",
            "identifier_mismatches",
        ):
            if not isinstance(values[name], list):
                raise ValueError(f"Serialized {name} must be a list.")
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class GroundedClaim:
    """One deterministic substantive answer claim and attached citations."""

    claim_id: str
    text: str
    citation_labels: tuple[str, ...]
    sentence_index: int
    substantive: bool
    supported: bool
    support: ClaimSupport | None

    def __post_init__(self) -> None:
        _nonempty_string(self.claim_id, "claim_id")
        _nonempty_string(self.text, "text")
        if not isinstance(self.citation_labels, tuple) or not all(
            _LABEL_PATTERN.fullmatch(label) for label in self.citation_labels
        ):
            raise ValueError("citation_labels must contain valid C# labels.")
        if len(set(self.citation_labels)) != len(self.citation_labels):
            raise ValueError("citation_labels must be normalized and unique.")
        _nonnegative_integer(self.sentence_index, "sentence_index")
        if not isinstance(self.substantive, bool) or not isinstance(
            self.supported, bool
        ):
            raise TypeError("Claim flags must be boolean.")
        if self.substantive and self.support is None:
            raise ValueError("A substantive claim requires support diagnostics.")
        if self.support is not None and self.support.supported != self.supported:
            raise ValueError("Claim supported flag must match support diagnostics.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "citation_labels": list(self.citation_labels),
            "sentence_index": self.sentence_index,
            "substantive": self.substantive,
            "supported": self.supported,
            "support": None if self.support is None else self.support.to_dict(),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> GroundedClaim:
        if not isinstance(state, Mapping) or set(state) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("Grounded claim state keys are malformed.")
        labels = state["citation_labels"]
        if not isinstance(labels, list):
            raise ValueError("Serialized citation_labels must be a list.")
        support = state["support"]
        return cls(
            claim_id=state["claim_id"],
            text=state["text"],
            citation_labels=tuple(labels),
            sentence_index=state["sentence_index"],
            substantive=state["substantive"],
            supported=state["supported"],
            support=None if support is None else ClaimSupport.from_dict(support),
        )


@dataclass(frozen=True)
class AnswerValidation:
    """Structural citation, coverage, and conservative support diagnostics."""

    accepted: bool
    citations_valid: bool
    citation_coverage: float
    unsupported_claim_count: int
    unknown_citation_count: int
    uncited_claim_count: int
    malformed_citation_count: int
    numerical_mismatch_count: int
    negation_warning_count: int
    exact_quote_characters: int
    longest_exact_quote: int
    quote_evidence_id: str | None
    evidence_hash: str
    index_sha256: str
    rejection_reasons: tuple[str, ...]
    unknown_citation_labels: tuple[str, ...]
    uncited_claim_ids: tuple[str, ...]
    unsupported_claim_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool) or not isinstance(
            self.citations_valid, bool
        ):
            raise TypeError("Validation acceptance flags must be boolean.")
        _fraction(self.citation_coverage, "citation_coverage")
        for name in (
            "unsupported_claim_count",
            "unknown_citation_count",
            "uncited_claim_count",
            "malformed_citation_count",
            "numerical_mismatch_count",
            "negation_warning_count",
            "exact_quote_characters",
            "longest_exact_quote",
        ):
            _nonnegative_integer(getattr(self, name), name)
        _optional_nonempty_string(self.quote_evidence_id, "quote_evidence_id")
        if not _is_sha256(self.evidence_hash) or not _is_sha256(self.index_sha256):
            raise ValueError("Validation identities must be SHA-256 digests.")
        for name in (
            "rejection_reasons",
            "unknown_citation_labels",
            "uncited_claim_ids",
            "unsupported_claim_ids",
        ):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not all(
                isinstance(item, str) and item for item in value
            ):
                raise ValueError(f"{name} must contain non-empty strings.")
        if self.unknown_citation_count != len(self.unknown_citation_labels):
            raise ValueError("unknown_citation_count is inconsistent.")
        if self.uncited_claim_count != len(self.uncited_claim_ids):
            raise ValueError("uncited_claim_count is inconsistent.")
        if self.unsupported_claim_count != len(self.unsupported_claim_ids):
            raise ValueError("unsupported_claim_count is inconsistent.")

    def to_dict(self) -> dict[str, Any]:
        state = dict(vars(self))
        for name in (
            "rejection_reasons",
            "unknown_citation_labels",
            "uncited_claim_ids",
            "unsupported_claim_ids",
        ):
            state[name] = list(state[name])
        return state

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> AnswerValidation:
        if not isinstance(state, Mapping) or set(state) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("Answer validation state keys are malformed.")
        values = dict(state)
        for name in (
            "rejection_reasons",
            "unknown_citation_labels",
            "uncited_claim_ids",
            "unsupported_claim_ids",
        ):
            if not isinstance(values[name], list):
                raise ValueError(f"Serialized {name} must be a list.")
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class GroundedAnswer:
    """Complete answer, evidence, validation, and explicit failure state."""

    question: str
    method: str
    answer_text: str
    raw_generated_text: str | None
    processed_generated_text: str | None
    claims: tuple[GroundedClaim, ...]
    evidence: tuple[EvidenceItem, ...]
    citations: tuple[CitationBinding, ...]
    sufficiency: EvidenceSufficiency
    abstained: bool
    abstention_reason: str | None
    validation: AnswerValidation
    fallback_used: bool
    fallback_reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty_string(self.question, "question")
        if self.method not in _ANSWER_METHODS:
            raise ValueError(f"Unknown answer method {self.method!r}.")
        _nonempty_string(self.answer_text, "answer_text")
        for name in ("raw_generated_text", "processed_generated_text"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be None or a string.")
        if not all(isinstance(claim, GroundedClaim) for claim in self.claims):
            raise TypeError("claims must contain GroundedClaim objects.")
        if not all(isinstance(item, EvidenceItem) for item in self.evidence):
            raise TypeError("evidence must contain EvidenceItem objects.")
        if not all(isinstance(item, CitationBinding) for item in self.citations):
            raise TypeError("citations must contain CitationBinding objects.")
        if len({item.evidence_id for item in self.evidence}) != len(self.evidence):
            raise ValueError("Evidence IDs must be unique.")
        if [item.label for item in self.evidence] != [
            f"C{index}" for index in range(1, len(self.evidence) + 1)
        ]:
            raise ValueError("Evidence labels must be contiguous in evidence order.")
        evidence_by_label = {item.label: item for item in self.evidence}
        if len(self.citations) != len(self.evidence):
            raise ValueError("Every evidence item requires one citation binding.")
        for binding in self.citations:
            evidence = evidence_by_label.get(binding.label)
            if (
                evidence is None
                or binding.evidence_id != evidence.evidence_id
                or binding.citation != evidence.citation
            ):
                raise ValueError("Citation bindings must match exact evidence items.")
        if self.validation.evidence_hash != evidence_set_hash(self.evidence):
            raise ValueError("Validation evidence hash does not match answer evidence.")
        if any(
            item.index_sha256 != self.validation.index_sha256 for item in self.evidence
        ):
            raise ValueError("Answer evidence and validation index identities differ.")
        if not isinstance(self.abstained, bool):
            raise TypeError("abstained must be boolean.")
        if self.abstained:
            _nonempty_string(self.abstention_reason, "abstention_reason")
            if self.claims:
                raise ValueError("An abstention cannot contain substantive claims.")
        elif self.abstention_reason is not None:
            raise ValueError("A non-abstention cannot have an abstention reason.")
        if not isinstance(self.fallback_used, bool):
            raise TypeError("fallback_used must be boolean.")
        if self.fallback_used:
            _nonempty_string(self.fallback_reason, "fallback_reason")
            if self.method != "generative_with_extractive_fallback":
                raise ValueError("Fallback is only valid for the fallback method.")
        elif self.fallback_reason is not None:
            raise ValueError("Unused fallback cannot have a fallback reason.")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")
        canonical_json(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "method": self.method,
            "answer_text": self.answer_text,
            "raw_generated_text": self.raw_generated_text,
            "processed_generated_text": self.processed_generated_text,
            "claims": [claim.to_dict() for claim in self.claims],
            "evidence": [item.to_dict() for item in self.evidence],
            "citations": [binding.to_dict() for binding in self.citations],
            "sufficiency": self.sufficiency.to_dict(),
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "validation": self.validation.to_dict(),
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> GroundedAnswer:
        if not isinstance(state, Mapping) or set(state) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("Grounded answer state keys are malformed.")
        values = dict(state)
        for name in ("claims", "evidence", "citations"):
            if not isinstance(values[name], list):
                raise ValueError(f"Serialized {name} must be a list.")
        values["claims"] = tuple(
            GroundedClaim.from_dict(item) for item in values["claims"]
        )
        values["evidence"] = tuple(
            EvidenceItem.from_dict(item) for item in values["evidence"]
        )
        values["citations"] = tuple(
            CitationBinding.from_dict(item) for item in values["citations"]
        )
        values["sufficiency"] = EvidenceSufficiency.from_dict(values["sufficiency"])
        values["validation"] = AnswerValidation.from_dict(values["validation"])
        return cls(**values)


def evidence_set_hash(evidence: tuple[EvidenceItem, ...]) -> str:
    """Hash answer-local evidence identity and exact selected text."""
    state = [
        {
            "label": item.label,
            "evidence_id": item.evidence_id,
            "chunk_id": item.chunk_id,
            "selected_text_sha256": item.selected_text_sha256,
            "citation": item.citation.to_dict(),
        }
        for item in evidence
    ]
    return sha256_text(canonical_json(state))
