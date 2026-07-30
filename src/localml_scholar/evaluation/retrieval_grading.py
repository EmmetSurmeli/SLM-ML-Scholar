"""Question-aware retrieval relevance, section, boilerplate, and redundancy grades."""

from __future__ import annotations

from collections.abc import Sequence

from localml_scholar.evaluation.schemas import BenchmarkQuestion, RetrievalGrade
from localml_scholar.retrieval import RetrievalIndex, SearchResult, ndcg_at_k

_BOILERPLATE_HEADINGS = frozenset(
    {
        "acknowledgment",
        "acknowledgments",
        "acknowledgements",
        "author contributions",
        "bibliography",
        "copyright",
        "funding",
        "license",
        "permissions",
        "references",
        "supplementary material",
    }
)
_QUESTION_SECTION_POLICY = {
    "metadata": (("title", "abstract"), ("references", "acknowledgments")),
    "motivation": (("abstract", "introduction"), ("references", "license")),
    "main_method": (("method", "methodology", "architecture"), ("references",)),
    "architecture": (("architecture", "method", "model"), ("references",)),
    "equation": (("method", "theory", "equation", "appendix"), ("references",)),
    "notation": (("method", "theory", "appendix"), ("references",)),
    "assumption": (("method", "theory", "discussion"), ("references",)),
    "methodology": (("method", "methodology"), ("references",)),
    "hyperparameter": (("experiment", "training", "implementation"), ("references",)),
    "experiment": (("experiment", "evaluation"), ("references",)),
    "result": (("result", "experiment"), ("references",)),
    "ablation": (("ablation", "experiment"), ("references",)),
    "limitation": (("limitation", "discussion", "conclusion"), ("references",)),
    "reproduction": (
        ("method", "training", "experiment", "implementation"),
        ("references",),
    ),
}


def _normalized_heading(result: SearchResult) -> str:
    return " ".join(result.heading_path).casefold()


def is_boilerplate_result(
    result: SearchResult,
    *,
    question_type: str,
) -> bool:
    """Return context-aware low-value section status."""
    if not isinstance(result, SearchResult):
        raise TypeError("result must be SearchResult.")
    heading = _normalized_heading(result)
    matched = any(label in heading for label in _BOILERPLATE_HEADINGS)
    if not matched:
        return False
    if question_type == "metadata" and "author contributions" in heading:
        return False
    return not (
        question_type in {"historical_impact", "comparison"}
        and ("reference" in heading or "bibliograph" in heading)
    )


def question_section_policy(
    question: BenchmarkQuestion,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve benchmark overrides over conservative question-type defaults."""
    if not isinstance(question, BenchmarkQuestion):
        raise TypeError("question must be BenchmarkQuestion.")
    defaults = _QUESTION_SECTION_POLICY.get(question.question_type, ((), ()))
    expected = question.expected_sections or defaults[0]
    forbidden = question.forbidden_sections or defaults[1]
    return expected, forbidden


def _heading_matches(result: SearchResult, labels: tuple[str, ...]) -> bool:
    heading = _normalized_heading(result)
    return any(label.casefold() in heading for label in labels)


def _is_title_page(result: SearchResult) -> bool:
    heading = _normalized_heading(result)
    return "title" in heading or result.start_line <= 20 or result.page_start == 1


def _source_overlap(left, right) -> float:
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


def grade_retrieval(
    question: BenchmarkQuestion,
    results: Sequence[SearchResult],
    *,
    index: RetrievalIndex,
    k: int = 5,
) -> RetrievalGrade:
    """Grade exact retrieval independently of any answer generation."""
    if not isinstance(question, BenchmarkQuestion):
        raise TypeError("question must be BenchmarkQuestion.")
    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise TypeError("results must be a sequence.")
    materialized = tuple(results)
    if not all(isinstance(item, SearchResult) for item in materialized):
        raise TypeError("results must contain SearchResult objects.")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    chunk_ids = [item.chunk_id for item in materialized]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Retrieval results must have unique chunk IDs.")
    valid_ids = {chunk.chunk_id for chunk in index.chunks}
    if any(item not in valid_ids for item in chunk_ids):
        raise ValueError("Retrieval results contain a chunk absent from the index.")
    relevant = set(question.relevant_chunk_ids)
    expected, forbidden = question_section_policy(question)

    def recall_at(limit: int) -> float:
        if not relevant:
            return 1.0
        return len(set(chunk_ids[:limit]) & relevant) / len(relevant)

    first_relevant = next(
        (
            position
            for position, chunk_id in enumerate(chunk_ids, start=1)
            if chunk_id in relevant
        ),
        None,
    )
    grades = {item.chunk_id: item.relevance_grade for item in question.gold_evidence}
    grades.update(
        {
            chunk_id: 1
            for chunk_id in question.acceptable_chunk_ids
            if chunk_id not in grades
        }
    )
    ndcg = 1.0 if not grades else ndcg_at_k(chunk_ids, grades, k)
    wrong_section = tuple(
        item.chunk_id
        for item in materialized
        if _heading_matches(item, forbidden)
        or bool(expected)
        and not _heading_matches(item, expected)
        and item.chunk_id not in relevant
    )
    boilerplate = tuple(
        item.chunk_id
        for item in materialized
        if is_boilerplate_result(item, question_type=question.question_type)
    )
    redundant_pairs = 0
    total_pairs = 0
    chunk_by_id = {item.chunk_id: item for item in index.chunks}
    for left_index, left in enumerate(materialized):
        for right in materialized[left_index + 1 :]:
            total_pairs += 1
            if (
                _source_overlap(
                    chunk_by_id[left.chunk_id],
                    chunk_by_id[right.chunk_id],
                )
                >= 0.8
            ):
                redundant_pairs += 1
    expected_hit = (
        1.0
        if not expected
        else float(any(_heading_matches(item, expected) for item in materialized[:k]))
    )
    positive = tuple(item for item in materialized if item.score > 0.0)
    irrelevant_positive = (
        0.0
        if not positive
        else sum(item.chunk_id not in relevant for item in positive) / len(positive)
    )
    return RetrievalGrade(
        recall_at_1=recall_at(1),
        recall_at_3=recall_at(3),
        recall_at_5=recall_at(5),
        reciprocal_rank=0.0 if first_relevant is None else 1.0 / first_relevant,
        hit_rate_at_k=float(bool(set(chunk_ids[:k]) & relevant)) if relevant else 1.0,
        ndcg_at_k=ndcg,
        expected_section_hit=expected_hit,
        title_page_hit=float(any(_is_title_page(item) for item in materialized[:k])),
        motivation_source_hit=float(
            any(
                _heading_matches(item, ("abstract", "introduction"))
                for item in materialized[:k]
            )
        ),
        forbidden_section_rate=(
            0.0
            if not materialized
            else sum(_heading_matches(item, forbidden) for item in materialized)
            / len(materialized)
        ),
        boilerplate_rate=(
            0.0 if not materialized else len(boilerplate) / len(materialized)
        ),
        evidence_redundancy=(
            0.0 if total_pairs == 0 else redundant_pairs / total_pairs
        ),
        irrelevant_positive_score_rate=irrelevant_positive,
        retrieved_chunk_ids=tuple(chunk_ids),
        missed_gold_chunk_ids=tuple(
            item for item in question.relevant_chunk_ids if item not in chunk_ids[:k]
        ),
        wrong_section_chunk_ids=wrong_section,
        boilerplate_chunk_ids=boilerplate,
    )
