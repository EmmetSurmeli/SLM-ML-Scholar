"""Deterministic retrieval, redundancy control, and sufficiency heuristics."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from localml_scholar.answering.models import EvidenceItem, EvidenceSufficiency
from localml_scholar.retrieval import (
    RetrievalIndex,
    SearchFilters,
    SearchResult,
    lexical_terms,
    tokenize_lexically,
)
from localml_scholar.tokenizer import Tokenizer

_STOP_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "why",
        "with",
    }
)


def meaningful_query_terms(text: str) -> tuple[str, ...]:
    """Return unique non-stop lexical query terms in source order."""
    terms = tokenize_lexically(text)
    meaningful = tuple(dict.fromkeys(term for term in terms if term not in _STOP_TERMS))
    return meaningful or tuple(dict.fromkeys(terms))


@dataclass(frozen=True)
class EvidenceSelectionConfig:
    """Validated lexical retrieval, selection, and sufficiency policy."""

    retrieval_method: str = "bm25"
    retrieval_top_k: int = 8
    evidence_top_k: int = 4
    maximum_evidence_characters: int = 4000
    minimum_score: float = 0.0
    require_positive_score: bool = True
    diversify_documents: bool = True
    maximum_chunks_per_document: int | None = 2
    maximum_source_overlap: float = 0.8
    minimum_top_score: float = 0.0
    minimum_query_term_coverage: float = 0.2
    minimum_unique_matched_terms: int = 1
    minimum_evidence_count: int = 1

    def __post_init__(self) -> None:
        if self.retrieval_method not in {"tfidf", "bm25"}:
            raise ValueError("retrieval_method must be 'tfidf' or 'bm25'.")
        for name in (
            "retrieval_top_k",
            "evidence_top_k",
            "maximum_evidence_characters",
            "minimum_unique_matched_terms",
            "minimum_evidence_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.evidence_top_k > self.retrieval_top_k:
            raise ValueError("evidence_top_k cannot exceed retrieval_top_k.")
        if self.maximum_chunks_per_document is not None:
            if isinstance(self.maximum_chunks_per_document, bool) or not isinstance(
                self.maximum_chunks_per_document, int
            ):
                raise TypeError("maximum_chunks_per_document must be None or integer.")
            if self.maximum_chunks_per_document <= 0:
                raise ValueError("maximum_chunks_per_document must be positive.")
        for name in ("minimum_score", "minimum_top_score"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, float(value))
        for name in ("maximum_source_overlap", "minimum_query_term_coverage"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number.")
            normalized = float(value)
            if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
            object.__setattr__(self, name, normalized)
        for name in ("require_positive_score", "diversify_documents"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> EvidenceSelectionConfig:
        if not isinstance(state, Mapping) or set(state) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("Evidence selection configuration is malformed.")
        return cls(**dict(state))


@dataclass(frozen=True)
class EvidenceSelection:
    """Selected evidence and the complete retrieval result set inspected."""

    evidence: tuple[EvidenceItem, ...]
    retrieval_results: tuple[SearchResult, ...]
    suppressed_chunk_ids: tuple[str, ...]


def _source_overlap(left: Any, right: Any) -> float:
    if left.document_id != right.document_id:
        return 0.0
    intersection = max(
        0,
        min(left.end_character, right.end_character)
        - max(left.start_character, right.start_character),
    )
    shorter = min(
        left.end_character - left.start_character,
        right.end_character - right.start_character,
    )
    return 0.0 if shorter <= 0 else intersection / shorter


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _slice_boundary(text: str, limit: int, *, from_start: bool) -> int:
    if limit <= 0 or limit >= len(text):
        return max(0, min(limit, len(text)))
    if from_start:
        candidates = [
            match.start()
            for match in re.finditer(r"(?:\n\s*\n|(?<=[.!?])\s+|\s+)", text)
            if match.start() >= limit
        ]
        return candidates[0] if candidates else limit
    candidates = [
        match.end()
        for match in re.finditer(r"(?:\n\s*\n|(?<=[.!?])\s+|\s+)", text[:limit])
    ]
    return candidates[-1] if candidates else limit


def _truncate_result(
    index: RetrievalIndex,
    result: SearchResult,
    maximum_characters: int,
) -> tuple[str, int, int, int, int, bool]:
    chunk = next(chunk for chunk in index.chunks if chunk.chunk_id == result.chunk_id)
    if len(result.text) <= maximum_characters:
        return (
            result.text,
            chunk.start_character,
            chunk.end_character,
            result.start_line,
            result.end_line,
            False,
        )
    matches = [
        term
        for term in lexical_terms(result.text, index.lexical_config)
        if term.term in set(result.matched_terms)
    ]
    anchor = matches[0].start_character if matches else 0
    local_start = max(0, anchor - maximum_characters // 4)
    if local_start:
        local_start = _slice_boundary(result.text, local_start, from_start=True)
    local_end = min(len(result.text), local_start + maximum_characters)
    if local_end < len(result.text):
        local_end = _slice_boundary(result.text, local_end, from_start=False)
    if local_end <= local_start:
        local_start = max(0, min(anchor, len(result.text) - maximum_characters))
        local_end = min(len(result.text), local_start + maximum_characters)
    selected = result.text[local_start:local_end]
    absolute_start = chunk.start_character + local_start
    absolute_end = chunk.start_character + local_end
    document = next(
        document
        for document in index.documents
        if document.document_id == result.document_id
    )
    return (
        selected,
        absolute_start,
        absolute_end,
        _line_number(document.text, absolute_start),
        _line_number(document.text, absolute_end - 1),
        True,
    )


def _make_evidence(
    index: RetrievalIndex,
    result: SearchResult,
    *,
    label: str,
    maximum_characters: int,
    tokenizer: Tokenizer | None,
) -> EvidenceItem:
    selected, start, end, start_line, end_line, truncated = _truncate_result(
        index,
        result,
        maximum_characters,
    )
    token_count = None if tokenizer is None else int(tokenizer.encode(selected).size)
    return EvidenceItem.create(
        label=label,
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        source_name=result.source_name,
        title=result.title,
        heading_path=result.heading_path,
        selected_text=selected,
        start_character=start,
        end_character=end,
        start_line=start_line,
        end_line=end_line,
        page_start=result.page_start,
        page_end=result.page_end,
        retrieval_rank=result.rank,
        retrieval_score=result.score,
        retrieval_method=result.retrieval_method,
        matched_terms=result.matched_terms,
        token_count=token_count,
        truncated=truncated,
        index_sha256=index.index_sha256,
    )


def select_evidence(
    index: RetrievalIndex,
    question: str,
    *,
    config: EvidenceSelectionConfig | None = None,
    filters: SearchFilters | None = None,
    tokenizer: Tokenizer | None = None,
) -> EvidenceSelection:
    """Retrieve and select exact passages with transparent range deduplication."""
    if not isinstance(index, RetrievalIndex):
        raise TypeError("index must be a RetrievalIndex.")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must contain non-whitespace text.")
    resolved = config or EvidenceSelectionConfig()
    if not isinstance(resolved, EvidenceSelectionConfig):
        raise TypeError("config must be EvidenceSelectionConfig.")
    if tokenizer is not None and not isinstance(tokenizer, Tokenizer):
        raise TypeError("tokenizer must implement the Tokenizer interface.")
    results = index.search(
        question,
        method=resolved.retrieval_method,
        top_k=resolved.retrieval_top_k,
        filters=filters,
    )
    content_query_terms = set(meaningful_query_terms(question))
    eligible = [
        result
        for result in results
        if result.score >= resolved.minimum_score
        and (not resolved.require_positive_score or result.score > 0.0)
        and bool(content_query_terms & set(result.matched_terms))
    ]
    if resolved.diversify_documents:
        first_per_document: list[SearchResult] = []
        remaining: list[SearchResult] = []
        seen: set[str] = set()
        for result in eligible:
            if result.document_id in seen:
                remaining.append(result)
            else:
                first_per_document.append(result)
                seen.add(result.document_id)
        candidates = first_per_document + remaining
    else:
        candidates = eligible
    chunk_by_id = {chunk.chunk_id: chunk for chunk in index.chunks}
    selected_results: list[SearchResult] = []
    counts: Counter[str] = Counter()
    suppressed: list[str] = []
    for result in candidates:
        if len(selected_results) >= resolved.evidence_top_k:
            break
        if (
            resolved.maximum_chunks_per_document is not None
            and counts[result.document_id] >= resolved.maximum_chunks_per_document
        ):
            suppressed.append(result.chunk_id)
            continue
        candidate_chunk = chunk_by_id[result.chunk_id]
        if any(
            _source_overlap(candidate_chunk, chunk_by_id[chosen.chunk_id])
            > resolved.maximum_source_overlap
            for chosen in selected_results
        ):
            suppressed.append(result.chunk_id)
            continue
        selected_results.append(result)
        counts[result.document_id] += 1
    selected_results.sort(key=lambda result: result.rank)
    remaining_characters = resolved.maximum_evidence_characters
    evidence: list[EvidenceItem] = []
    for result in selected_results:
        if remaining_characters <= 0:
            suppressed.append(result.chunk_id)
            continue
        maximum = min(len(result.text), remaining_characters)
        if maximum <= 0:
            continue
        item = _make_evidence(
            index,
            result,
            label=f"C{len(evidence) + 1}",
            maximum_characters=maximum,
            tokenizer=tokenizer,
        )
        evidence.append(item)
        remaining_characters -= len(item.selected_text)
    return EvidenceSelection(
        evidence=tuple(evidence),
        retrieval_results=results,
        suppressed_chunk_ids=tuple(dict.fromkeys(suppressed)),
    )


def assess_evidence_sufficiency(
    question: str,
    evidence: tuple[EvidenceItem, ...],
    *,
    config: EvidenceSelectionConfig | None = None,
) -> EvidenceSufficiency:
    """Apply explicit lexical thresholds; this does not prove factual support."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must contain non-whitespace text.")
    if not isinstance(evidence, tuple) or not all(
        isinstance(item, EvidenceItem) for item in evidence
    ):
        raise TypeError("evidence must be a tuple of EvidenceItem objects.")
    resolved = config or EvidenceSelectionConfig()
    query_terms = meaningful_query_terms(question)
    matched = tuple(
        term
        for term in query_terms
        if any(term in item.matched_terms for item in evidence)
    )
    unmatched = tuple(term for term in query_terms if term not in set(matched))
    coverage = 0.0 if not query_terms else len(matched) / len(query_terms)
    top_score = max((item.retrieval_score for item in evidence), default=0.0)
    source_count = len({item.document_id for item in evidence})
    content_present = any(
        tokenize_lexically(
            "\n".join(
                line
                for line in item.selected_text.splitlines()
                if not re.match(r"^\s*#{1,6}\s+", line)
            )
        )
        for item in evidence
    )
    failures: list[str] = []
    if len(evidence) < resolved.minimum_evidence_count:
        failures.append("too_few_evidence_items")
    if top_score < resolved.minimum_top_score:
        failures.append("top_score_below_threshold")
    if len(matched) < resolved.minimum_unique_matched_terms:
        failures.append("too_few_unique_query_terms")
    if coverage < resolved.minimum_query_term_coverage:
        failures.append("query_term_coverage_below_threshold")
    if not content_present:
        failures.append("heading_only_or_empty_evidence")
    score_signal = top_score / (1.0 + top_score) if top_score > 0.0 else 0.0
    count_signal = min(1.0, len(evidence) / resolved.minimum_evidence_count)
    heuristic_score = (coverage + score_signal + count_signal) / 3.0
    return EvidenceSufficiency(
        sufficient=not failures,
        score=heuristic_score,
        reasons=tuple(failures or ["thresholds_satisfied"]),
        matched_query_terms=matched,
        unmatched_query_terms=unmatched,
        evidence_count=len(evidence),
        unique_source_count=source_count,
        query_term_coverage=coverage,
        top_retrieval_score=top_score,
    )


def relabel_evidence(evidence: tuple[EvidenceItem, ...]) -> tuple[EvidenceItem, ...]:
    """Return identical evidence slices with contiguous answer-local labels."""
    return tuple(
        EvidenceItem.create(
            label=f"C{index}",
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            source_name=item.source_name,
            title=item.title,
            heading_path=item.heading_path,
            selected_text=item.selected_text,
            start_character=item.start_character,
            end_character=item.end_character,
            start_line=item.start_line,
            end_line=item.end_line,
            page_start=item.page_start,
            page_end=item.page_end,
            retrieval_rank=item.retrieval_rank,
            retrieval_score=item.retrieval_score,
            retrieval_method=item.retrieval_method,
            matched_terms=item.matched_terms,
            token_count=item.token_count,
            truncated=item.truncated,
            index_sha256=item.index_sha256,
        )
        for index, item in enumerate(evidence, start=1)
    )


def truncate_evidence_item(
    index: RetrievalIndex,
    item: EvidenceItem,
    *,
    maximum_characters: int,
    tokenizer: Tokenizer,
) -> EvidenceItem:
    """Shorten one selected source slice without losing source/citation linkage."""
    if not isinstance(index, RetrievalIndex):
        raise TypeError("index must be a RetrievalIndex.")
    if not isinstance(item, EvidenceItem):
        raise TypeError("item must be an EvidenceItem.")
    if isinstance(maximum_characters, bool) or not isinstance(maximum_characters, int):
        raise TypeError("maximum_characters must be an integer.")
    if maximum_characters <= 0:
        raise ValueError("maximum_characters must be positive.")
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError("tokenizer must implement the Tokenizer interface.")
    if len(item.selected_text) <= maximum_characters:
        return item
    local_end = _slice_boundary(
        item.selected_text,
        maximum_characters,
        from_start=False,
    )
    if local_end <= 0:
        local_end = maximum_characters
    selected = item.selected_text[:local_end]
    absolute_end = item.start_character + local_end
    document = next(
        (
            document
            for document in index.documents
            if document.document_id == item.document_id
        ),
        None,
    )
    if document is None:
        raise ValueError("Evidence document is not present in the retrieval index.")
    if document.text[item.start_character : absolute_end] != selected:
        raise ValueError("Evidence text no longer matches the retrieval index.")
    return EvidenceItem.create(
        label=item.label,
        chunk_id=item.chunk_id,
        document_id=item.document_id,
        source_name=item.source_name,
        title=item.title,
        heading_path=item.heading_path,
        selected_text=selected,
        start_character=item.start_character,
        end_character=absolute_end,
        start_line=item.start_line,
        end_line=_line_number(document.text, absolute_end - 1),
        page_start=item.page_start,
        page_end=item.page_end,
        retrieval_rank=item.retrieval_rank,
        retrieval_score=item.retrieval_score,
        retrieval_method=item.retrieval_method,
        matched_terms=item.matched_terms,
        token_count=int(tokenizer.encode(selected).size),
        truncated=True,
        index_sha256=item.index_sha256,
    )
