"""Optional equation-aware reranking layered over unchanged retrieval methods."""

from __future__ import annotations

import re

from localml_scholar.retrieval import RetrievalIndex, SearchFilters
from localml_scholar.scholarly.config import ScholarlyConfig
from localml_scholar.scholarly.equations import symbol_spans
from localml_scholar.scholarly.models import ScholarlySearchResult

_EQUATION_NUMBER = re.compile(r"\b(?:equation|eq\.?)\s*(?P<number>\d+[a-z]?)", re.I)
_OPERATORS = re.compile(r"<=|>=|≤|≥|=|∑|∫|∂|argmax|argmin|\blog\b")


def equation_aware_search(
    index: RetrievalIndex,
    query: str,
    *,
    document_id: str,
    method: str = "bm25",
    top_k: int = 5,
    candidate_count: int = 20,
    config: ScholarlyConfig | None = None,
) -> tuple[ScholarlySearchResult, ...]:
    """Rerank exact cited chunks using disclosed equation-text signals."""
    resolved = config or ScholarlyConfig()
    if method not in {"bm25", "tfidf", "semantic", "hybrid", "hybrid_reranked"}:
        raise ValueError("Unsupported base retrieval method.")
    if candidate_count < top_k or top_k <= 0:
        raise ValueError("candidate_count must be at least positive top_k.")
    results = index.search(
        query,
        method=method,
        top_k=candidate_count,
        filters=SearchFilters(document_id=document_id),
    )
    if not results:
        return ()
    maximum = max(item.score for item in results)
    query_symbols = {item[0] for item in symbol_spans(query)}
    query_operators = set(_OPERATORS.findall(query))
    number_match = _EQUATION_NUMBER.search(query)
    query_number = None if number_match is None else number_match.group("number")
    weight = resolved.equation_aware_retrieval_weight
    scored: list[tuple[float, str, ScholarlySearchResult]] = []
    for result in results:
        text_symbols = {item[0] for item in symbol_spans(result.text)}
        symbol_overlap = tuple(sorted(query_symbols & text_symbols))
        operator_overlap = tuple(
            sorted(query_operators & set(_OPERATORS.findall(result.text)))
        )
        number_match_score = (
            1.0
            if query_number is not None
            and re.search(
                rf"(?:\\tag\{{{re.escape(query_number)}\}}|\({re.escape(query_number)}\))",
                result.text,
            )
            else 0.0
        )
        symbol_score = len(symbol_overlap) / max(1, len(query_symbols))
        operator_score = len(operator_overlap) / max(1, len(query_operators))
        equation_signal = min(
            1.0,
            0.55 * symbol_score + 0.30 * number_match_score + 0.15 * operator_score,
        )
        base_score = result.score / maximum if maximum > 0.0 else 0.0
        score = (1.0 - weight) * base_score + weight * equation_signal
        wrapper = ScholarlySearchResult(
            base_result=result.to_dict(),
            scholarly_score=score,
            equation_signals={
                "base_method": method,
                "base_normalized_score": base_score,
                "equation_weight": weight,
                "symbol_overlap": list(symbol_overlap),
                "symbol_overlap_score": symbol_score,
                "equation_number": query_number,
                "equation_number_match": bool(number_match_score),
                "operator_overlap": list(operator_overlap),
                "operator_overlap_score": operator_score,
                "symbolic_equivalence_claimed": False,
            },
        )
        scored.append((score, result.chunk_id, wrapper))
    return tuple(
        item[2]
        for item in sorted(scored, key=lambda value: (-value[0], value[1]))[:top_k]
    )
