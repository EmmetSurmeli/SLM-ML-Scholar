"""Citation, claim coverage, and conservative lexical support validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from localml_scholar.answering.citations import (
    CitationSyntaxError,
    parse_inline_citations,
    strip_inline_citations,
)
from localml_scholar.answering.models import (
    AnswerValidation,
    ClaimSupport,
    EvidenceItem,
    GroundedClaim,
    evidence_set_hash,
)
from localml_scholar.answering.segmentation import segment_answer_claims
from localml_scholar.retrieval import RetrievalIndex, tokenize_lexically
from localml_scholar.retrieval.documents import stable_identifier

_CLAIM_STOP_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)
_NEGATIONS = frozenset({"no", "not", "never", "none", "without", "cannot"})
_NUMBER_PATTERN = re.compile(r"(?<!\w)[+-]?(?:\d+(?:\.\d+)?|\.\d+)%?")
_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:[A-Za-z]+_[A-Za-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]*)\b"
)
_EQUATION_SYMBOL_PATTERN = re.compile(r"<=|>=|==|!=|[=<>±×÷∑√λ]")


@dataclass(frozen=True)
class AnswerAcceptanceConfig:
    """Conservative generated-answer acceptance policy."""

    require_all_citations_valid: bool = True
    minimum_citation_coverage: float = 1.0
    minimum_claim_support_score: float = 0.45
    reject_unknown_citations: bool = True
    reject_uncited_claims: bool = True
    reject_unsupported_claims: bool = True
    reject_number_mismatches: bool = True
    reject_negation_warnings: bool = True
    allow_abstention: bool = True

    def __post_init__(self) -> None:
        for name in (
            "minimum_citation_coverage",
            "minimum_claim_support_score",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number.")
            normalized = float(value)
            if not 0.0 <= normalized <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
            object.__setattr__(self, name, normalized)
        for name in (
            "require_all_citations_valid",
            "reject_unknown_citations",
            "reject_uncited_claims",
            "reject_unsupported_claims",
            "reject_number_mismatches",
            "reject_negation_warnings",
            "allow_abstention",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> AnswerAcceptanceConfig:
        if not isinstance(state, Mapping) or set(state) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("Answer acceptance configuration is malformed.")
        return cls(**dict(state))


def _claim_terms(text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            term for term in tokenize_lexically(text) if term not in _CLAIM_STOP_TERMS
        )
    )


def _contains_negation(text: str) -> bool:
    return bool(set(tokenize_lexically(text)) & _NEGATIONS)


def _phrase_overlap(claim_terms: tuple[str, ...], evidence_text: str) -> bool:
    evidence_terms = tokenize_lexically(evidence_text)
    if len(claim_terms) < 3:
        return False
    triples = {claim_terms[index : index + 3] for index in range(len(claim_terms) - 2)}
    evidence_triples = {
        evidence_terms[index : index + 3]
        for index in range(max(0, len(evidence_terms) - 2))
    }
    return bool(triples & evidence_triples)


def _quote_statistics(
    claim: str,
    evidence: tuple[EvidenceItem, ...],
) -> tuple[int, int, str | None]:
    total_best = 0
    longest = 0
    source: str | None = None
    for item in evidence:
        matcher = SequenceMatcher(None, claim, item.selected_text, autojunk=False)
        blocks = [
            block.size for block in matcher.get_matching_blocks() if block.size >= 20
        ]
        copied = sum(blocks)
        candidate_longest = max(blocks, default=0)
        if candidate_longest > longest:
            longest = candidate_longest
            source = item.evidence_id
        total_best = max(total_best, copied)
    return total_best, longest, source


def assess_claim_support(
    claim_text: str,
    cited_evidence: tuple[EvidenceItem, ...],
    *,
    minimum_score: float = 0.45,
) -> ClaimSupport:
    """Apply lexical, number, identifier, symbol, and negation diagnostics."""
    if not isinstance(claim_text, str) or not claim_text.strip():
        raise ValueError("claim_text must be non-empty.")
    if not isinstance(cited_evidence, tuple) or not all(
        isinstance(item, EvidenceItem) for item in cited_evidence
    ):
        raise TypeError("cited_evidence must contain EvidenceItem objects.")
    if isinstance(minimum_score, bool) or not isinstance(minimum_score, (int, float)):
        raise TypeError("minimum_score must be a real number.")
    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must lie in [0, 1].")
    clean_claim = strip_inline_citations(claim_text).strip()
    evidence_text = "\n".join(item.selected_text for item in cited_evidence)
    exact_source_quote = any(
        clean_claim in item.selected_text for item in cited_evidence
    )
    claim_terms = _claim_terms(clean_claim)
    evidence_terms = set(_claim_terms(evidence_text))
    overlapping = tuple(term for term in claim_terms if term in evidence_terms)
    missing = tuple(term for term in claim_terms if term not in evidence_terms)
    term_score = len(overlapping) / max(1, len(claim_terms))
    claim_numbers = tuple(dict.fromkeys(_NUMBER_PATTERN.findall(clean_claim)))
    evidence_numbers = set(_NUMBER_PATTERN.findall(evidence_text))
    number_mismatches = tuple(
        number for number in claim_numbers if number not in evidence_numbers
    )
    claim_identifiers = tuple(dict.fromkeys(_IDENTIFIER_PATTERN.findall(clean_claim)))
    evidence_identifiers = set(_IDENTIFIER_PATTERN.findall(evidence_text))
    identifier_mismatches = tuple(
        value for value in claim_identifiers if value not in evidence_identifiers
    )
    claim_symbols = tuple(dict.fromkeys(_EQUATION_SYMBOL_PATTERN.findall(clean_claim)))
    evidence_symbols = set(_EQUATION_SYMBOL_PATTERN.findall(evidence_text))
    symbol_mismatches = tuple(
        value for value in claim_symbols if value not in evidence_symbols
    )
    negation_warning = (
        bool(cited_evidence)
        and not exact_source_quote
        and (_contains_negation(clean_claim) != _contains_negation(evidence_text))
    )
    phrase_overlap = _phrase_overlap(claim_terms, evidence_text)
    copied, longest, quote_source = _quote_statistics(clean_claim, cited_evidence)
    exact_signal = 1.0 if phrase_overlap or longest >= 20 else 0.0
    identifier_score = (
        1.0
        if not claim_identifiers
        else 1.0 - len(identifier_mismatches) / len(claim_identifiers)
    )
    symbol_score = (
        1.0 if not claim_symbols else 1.0 - len(symbol_mismatches) / len(claim_symbols)
    )
    score = (
        0.60 * term_score
        + 0.20 * exact_signal
        + 0.10 * identifier_score
        + 0.10 * symbol_score
    )
    failures: list[str] = []
    if not cited_evidence:
        failures.append("no_valid_cited_evidence")
    if score < minimum_score:
        failures.append("lexical_support_below_threshold")
    if number_mismatches:
        failures.append("number_mismatch")
    if identifier_mismatches:
        failures.append("identifier_mismatch")
    if symbol_mismatches:
        failures.append("equation_symbol_mismatch")
    if negation_warning:
        failures.append("negation_mismatch_warning")
    return ClaimSupport(
        supported=not failures,
        score=score,
        reasons=tuple(failures or ["support_thresholds_satisfied"]),
        overlapping_terms=overlapping,
        missing_key_terms=missing,
        number_mismatches=number_mismatches,
        equation_symbol_mismatches=symbol_mismatches,
        identifier_mismatches=identifier_mismatches,
        negation_warning=negation_warning,
        exact_phrase_overlap=phrase_overlap,
        exact_quoted_characters=copied,
        longest_exact_quote=longest,
        quote_evidence_id=quote_source,
    )


def _validate_evidence_against_index(
    index: RetrievalIndex,
    evidence: tuple[EvidenceItem, ...],
) -> tuple[str, ...]:
    errors: list[str] = []
    chunks = {chunk.chunk_id: chunk for chunk in index.chunks}
    documents = {document.document_id: document for document in index.documents}
    for item in evidence:
        if item.index_sha256 != index.index_sha256:
            errors.append(f"index_hash_mismatch:{item.label}")
            continue
        chunk = chunks.get(item.chunk_id)
        document = documents.get(item.document_id)
        if chunk is None:
            errors.append(f"missing_chunk:{item.label}")
            continue
        if document is None or chunk.document_id != item.document_id:
            errors.append(f"missing_document:{item.label}")
            continue
        if (
            item.source_name != document.source_name
            or item.title != document.title
            or item.heading_path != chunk.heading_path
            or item.page_start != chunk.page_start
            or item.page_end != chunk.page_end
        ):
            errors.append(f"evidence_metadata_mismatch:{item.label}")
            continue
        if not (
            chunk.start_character
            <= item.start_character
            < item.end_character
            <= chunk.end_character
        ):
            errors.append(f"evidence_range_outside_chunk:{item.label}")
            continue
        expected_start_line = document.text.count("\n", 0, item.start_character) + 1
        expected_end_line = document.text.count("\n", 0, item.end_character - 1) + 1
        if item.start_line != expected_start_line or item.end_line != expected_end_line:
            errors.append(f"evidence_line_mismatch:{item.label}")
            continue
        if (
            document.text[item.start_character : item.end_character]
            != item.selected_text
        ):
            errors.append(f"evidence_text_mismatch:{item.label}")
    return tuple(errors)


def validate_answer_text(
    index: RetrievalIndex,
    answer_text: str,
    evidence: tuple[EvidenceItem, ...],
    *,
    config: AnswerAcceptanceConfig | None = None,
    abstained: bool = False,
) -> tuple[tuple[GroundedClaim, ...], AnswerValidation]:
    """Validate citations and claim support against this exact index snapshot."""
    if not isinstance(index, RetrievalIndex):
        raise TypeError("index must be a RetrievalIndex.")
    if not isinstance(answer_text, str) or not answer_text.strip():
        raise ValueError("answer_text must be non-empty.")
    if not isinstance(evidence, tuple) or not all(
        isinstance(item, EvidenceItem) for item in evidence
    ):
        raise TypeError("evidence must contain EvidenceItem objects.")
    resolved = config or AnswerAcceptanceConfig()
    malformed_count = 0
    try:
        all_occurrences = parse_inline_citations(answer_text, strict=True)
    except CitationSyntaxError:
        malformed_count = 1
        all_occurrences = parse_inline_citations(answer_text, strict=False)
    known = {item.label: item for item in evidence}
    unknown_labels = tuple(
        dict.fromkeys(
            label
            for occurrence in all_occurrences
            for label in occurrence.labels
            if label not in known
        )
    )
    evidence_errors = _validate_evidence_against_index(index, evidence)
    claims: list[GroundedClaim] = []
    for sentence_index, claim_text in enumerate(segment_answer_claims(answer_text)):
        clean = strip_inline_citations(claim_text).strip()
        terms = _claim_terms(clean)
        substantive = len(terms) >= 2 or bool(
            _NUMBER_PATTERN.search(clean)
            or _IDENTIFIER_PATTERN.search(clean)
            or _EQUATION_SYMBOL_PATTERN.search(clean)
        )
        occurrences = parse_inline_citations(claim_text, strict=False)
        attached: list[str] = []
        for occurrence in occurrences:
            prefix = claim_text[: occurrence.start_character]
            if tokenize_lexically(strip_inline_citations(prefix)):
                attached.extend(occurrence.labels)
        labels = tuple(dict.fromkeys(attached))
        cited = tuple(known[label] for label in labels if label in known)
        support = (
            assess_claim_support(
                clean,
                cited,
                minimum_score=resolved.minimum_claim_support_score,
            )
            if substantive
            else None
        )
        supported = True if support is None else support.supported
        claims.append(
            GroundedClaim(
                claim_id=stable_identifier(
                    "claim",
                    answer_text,
                    sentence_index,
                    clean,
                ),
                text=clean,
                citation_labels=labels,
                sentence_index=sentence_index,
                substantive=substantive,
                supported=supported,
                support=support,
            )
        )
    substantive_claims = tuple(claim for claim in claims if claim.substantive)
    uncited = tuple(
        claim.claim_id
        for claim in substantive_claims
        if not any(label in known for label in claim.citation_labels)
    )
    unsupported = tuple(
        claim.claim_id for claim in substantive_claims if not claim.supported
    )
    if substantive_claims:
        covered = len(substantive_claims) - len(uncited)
        coverage = covered / len(substantive_claims)
    else:
        coverage = 1.0 if abstained else 0.0
    number_mismatch_count = sum(
        len(claim.support.number_mismatches)
        for claim in substantive_claims
        if claim.support is not None
    )
    negation_count = sum(
        int(claim.support.negation_warning)
        for claim in substantive_claims
        if claim.support is not None
    )
    quote_claim = max(
        (claim for claim in substantive_claims if claim.support is not None),
        key=lambda claim: claim.support.longest_exact_quote,
        default=None,
    )
    exact_quote_characters = sum(
        claim.support.exact_quoted_characters
        for claim in substantive_claims
        if claim.support is not None
    )
    longest_quote = (
        0 if quote_claim is None else quote_claim.support.longest_exact_quote
    )
    quote_evidence = (
        None if quote_claim is None else quote_claim.support.quote_evidence_id
    )
    citations_valid = (
        malformed_count == 0 and not unknown_labels and not evidence_errors
    )
    rejection_reasons: list[str] = list(evidence_errors)
    if resolved.require_all_citations_valid and not citations_valid:
        rejection_reasons.append("citations_invalid")
    if resolved.reject_unknown_citations and unknown_labels:
        rejection_reasons.append("unknown_citations")
    if resolved.reject_uncited_claims and uncited:
        rejection_reasons.append("uncited_claims")
    if coverage < resolved.minimum_citation_coverage:
        rejection_reasons.append("citation_coverage_below_threshold")
    if resolved.reject_unsupported_claims and unsupported:
        rejection_reasons.append("unsupported_claims")
    if resolved.reject_number_mismatches and number_mismatch_count:
        rejection_reasons.append("numerical_mismatches")
    if resolved.reject_negation_warnings and negation_count:
        rejection_reasons.append("negation_warnings")
    if abstained and not resolved.allow_abstention:
        rejection_reasons.append("abstention_not_allowed")
    if abstained and substantive_claims:
        rejection_reasons.append("abstention_contains_claims")
    return tuple(claims), AnswerValidation(
        accepted=not rejection_reasons,
        citations_valid=citations_valid,
        citation_coverage=coverage,
        unsupported_claim_count=len(unsupported),
        unknown_citation_count=len(unknown_labels),
        uncited_claim_count=len(uncited),
        malformed_citation_count=malformed_count,
        numerical_mismatch_count=number_mismatch_count,
        negation_warning_count=negation_count,
        exact_quote_characters=exact_quote_characters,
        longest_exact_quote=longest_quote,
        quote_evidence_id=quote_evidence,
        evidence_hash=evidence_set_hash(evidence),
        index_sha256=index.index_sha256,
        rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
        unknown_citation_labels=unknown_labels,
        uncited_claim_ids=uncited,
        unsupported_claim_ids=unsupported,
    )
