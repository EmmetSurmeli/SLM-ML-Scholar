"""Transparent sufficiency, relevance, concept, claim, and citation grading."""

from __future__ import annotations

import re

from localml_scholar.answering import GroundedAnswer
from localml_scholar.answering.citations import (
    CitationSyntaxError,
    parse_inline_citations,
    strip_inline_citations,
)
from localml_scholar.evaluation.retrieval_grading import (
    is_boilerplate_result,
    question_section_policy,
)
from localml_scholar.evaluation.schemas import (
    AnswerGrade,
    BenchmarkQuestion,
    CitationGrade,
    ConceptCoverage,
    ConceptGroup,
    SufficiencyGrade,
)
from localml_scholar.retrieval import RetrievalIndex, SearchResult, tokenize_lexically

_MECHANISM_MARKERS = frozenset(
    {
        "allows",
        "because",
        "blocks",
        "by",
        "enables",
        "prevents",
        "so",
        "through",
        "using",
        "uses",
    }
)
_QUALIFICATION_MARKERS = frozenset(
    {
        "although",
        "however",
        "limited",
        "may",
        "might",
        "only",
        "not",
        "suggests",
        "under",
    }
)
_TRADEOFF_MARKERS = frozenset(
    {"but", "cost", "however", "tradeoff", "while", "whereas", "memory", "compute"}
)
_EXPERIMENT_MARKERS = frozenset(
    {"dataset", "experiment", "evaluation", "metric", "bleu", "accuracy", "result"}
)
_ASSUMPTION_MARKERS = frozenset({"assume", "assumption", "provided", "requires"})
_LIMITATION_MARKERS = frozenset(
    {"limitation", "limited", "cannot", "unclear", "unknown", "future", "not"}
)
_EVIDENCE_MARKERS = frozenset(
    {"empirical", "experiment", "measured", "reported", "theoretical", "proof"}
)
_PERSON_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


def _normalized_phrase(value: str) -> str:
    return " ".join(value.casefold().split())


def _phrase_present(text: str, phrase: str) -> bool:
    normalized_text = _normalized_phrase(strip_inline_citations(text))
    normalized_phrase = _normalized_phrase(phrase)
    if normalized_phrase in normalized_text:
        return True
    phrase_terms = set(tokenize_lexically(phrase))
    return bool(phrase_terms) and phrase_terms <= set(tokenize_lexically(text))


def _concept_matches(text: str, concept: ConceptGroup) -> bool:
    return any(
        _phrase_present(text, candidate)
        for candidate in (concept.concept, *concept.aliases)
    )


def grade_concepts(
    question: BenchmarkQuestion,
    answer: GroundedAnswer,
) -> ConceptCoverage:
    """Measure manually specified concept presence, citation, and support."""
    if not isinstance(question, BenchmarkQuestion):
        raise TypeError("question must be BenchmarkQuestion.")
    if not isinstance(answer, GroundedAnswer):
        raise TypeError("answer must be GroundedAnswer.")

    present_required: list[str] = []
    missing_required: list[str] = []
    uncited_required: list[str] = []
    unsupported_required: list[str] = []
    supported_count = 0
    for concept in question.required_concepts:
        matching_claims = tuple(
            claim for claim in answer.claims if _concept_matches(claim.text, concept)
        )
        if not matching_claims and not _concept_matches(answer.answer_text, concept):
            missing_required.append(concept.concept)
            continue
        present_required.append(concept.concept)
        if not any(claim.citation_labels for claim in matching_claims):
            uncited_required.append(concept.concept)
        if not any(claim.supported for claim in matching_claims):
            unsupported_required.append(concept.concept)
        else:
            supported_count += 1
    present_optional = tuple(
        concept.concept
        for concept in question.optional_concepts
        if _concept_matches(answer.answer_text, concept)
    )
    required_recall = (
        1.0
        if not question.required_concepts
        else supported_count / len(question.required_concepts)
    )
    optional_recall = (
        1.0
        if not question.optional_concepts
        else len(present_optional) / len(question.optional_concepts)
    )
    return ConceptCoverage(
        required_recall=required_recall,
        optional_recall=optional_recall,
        present_required=tuple(present_required),
        missing_required=tuple(missing_required),
        present_optional=present_optional,
        uncited_required=tuple(uncited_required),
        unsupported_required=tuple(unsupported_required),
    )


def prohibited_claims_present(
    question: BenchmarkQuestion,
    answer_text: str,
) -> tuple[str, ...]:
    """Return prohibited benchmark claims detected by transparent lexical rules."""
    if not isinstance(question, BenchmarkQuestion):
        raise TypeError("question must be BenchmarkQuestion.")
    if not isinstance(answer_text, str):
        raise TypeError("answer_text must be a string.")
    return tuple(
        claim
        for claim in question.prohibited_claims
        if _phrase_present(answer_text, claim)
    )


def grade_sufficiency(
    question: BenchmarkQuestion,
    answer: GroundedAnswer,
) -> SufficiencyGrade:
    """Compare the paper-only answer decision with human gold sufficiency."""
    if not isinstance(question, BenchmarkQuestion):
        raise TypeError("question must be BenchmarkQuestion.")
    if not isinstance(answer, GroundedAnswer):
        raise TypeError("answer must be GroundedAnswer.")
    answer_terms = set(tokenize_lexically(answer.answer_text))
    qualified = answer.abstained or bool(answer_terms & _QUALIFICATION_MARKERS)
    external_recognized = answer.abstained or (
        "external" in answer_terms
        or "outside" in answer_terms
        or "later" in answer_terms
    )
    if answer.abstained:
        predicted = "insufficient"
    elif external_recognized and question.paper_sufficiency == "external_required":
        predicted = "external_required"
    elif qualified and question.paper_sufficiency == "partially_sufficient":
        predicted = "partially_sufficient"
    else:
        predicted = "sufficient"
    correct = predicted == question.paper_sufficiency
    if question.paper_sufficiency == "partially_sufficient" and predicted in {
        "partially_sufficient",
        "sufficient",
    }:
        correct = qualified
    false_answer = (
        question.paper_sufficiency in {"insufficient", "external_required"}
        and not answer.abstained
        and not (question.paper_sufficiency == "external_required" and qualified)
    )
    false_abstention = question.paper_sufficiency == "sufficient" and answer.abstained
    return SufficiencyGrade(
        gold_label=question.paper_sufficiency,
        predicted_label=predicted,
        correct=float(correct),
        false_answer=float(false_answer),
        false_abstention=float(false_abstention),
        external_context_recognized=float(
            question.paper_sufficiency != "external_required" or external_recognized
        ),
        correctly_qualified=float(
            question.paper_sufficiency
            not in {"partially_sufficient", "external_required"}
            or qualified
        ),
    )


def _interrogative_relevance(
    question: BenchmarkQuestion,
    answer_text: str,
    concepts: ConceptCoverage,
) -> tuple[float, tuple[str, ...], float]:
    clean_answer = strip_inline_citations(answer_text)
    terms = set(tokenize_lexically(clean_answer))
    prompt_terms = tokenize_lexically(question.question)
    first = prompt_terms[0] if prompt_terms else ""
    reasons: list[str] = []
    wrong_entity = False
    matches = True
    if first == "who" or (
        question.question_type == "metadata" and "author" in prompt_terms
    ):
        matches = bool(_PERSON_PATTERN.search(clean_answer)) or any(
            _phrase_present(clean_answer, item)
            for item in question.expected_identifiers
        )
        wrong_entity = not matches
        if not matches:
            reasons.append("who_answer_missing_person")
    elif first == "when":
        matches = bool(_YEAR_PATTERN.search(clean_answer)) or any(
            item in clean_answer for item in question.expected_numbers
        )
        wrong_entity = not matches
        if not matches:
            reasons.append("when_answer_missing_date")
    elif "dataset" in prompt_terms or question.question_type == "experiment":
        matches = "dataset" in terms or any(
            _phrase_present(clean_answer, item)
            for item in question.expected_identifiers
        )
        if not matches:
            reasons.append("dataset_answer_missing_dataset_entity")
    elif first in {"how", "why"} or question.question_type in {
        "motivation",
        "main_method",
        "architecture",
        "methodology",
    }:
        matches = bool(terms & _MECHANISM_MARKERS)
        if not matches:
            reasons.append("mechanism_or_rationale_missing")
    concept_signal = concepts.required_recall
    lexical_signal = len(set(tokenize_lexically(question.question)) & terms) / max(
        1, len(set(tokenize_lexically(question.question)))
    )
    relevance = (
        0.55 * float(matches)
        + 0.35 * concept_signal
        + 0.10 * min(1.0, lexical_signal * 2.0)
    )
    if question.answerability in {"unanswerable", "external_sources_required"}:
        relevance = max(relevance, float("not" in terms or "cannot" in terms))
    return relevance, tuple(reasons), float(wrong_entity)


def _requirement_present(requirement: str, answer_text: str) -> bool:
    terms = set(tokenize_lexically(answer_text))
    rules = {
        "direct_answer": bool(strip_inline_citations(answer_text).strip()),
        "mechanism": bool(terms & _MECHANISM_MARKERS),
        "tradeoff": bool(terms & _TRADEOFF_MARKERS),
        "experiment_connection": bool(terms & _EXPERIMENT_MARKERS),
        "architecture_connection": bool(
            terms & {"architecture", "encoder", "decoder", "layer", "attention"}
        ),
        "assumptions": bool(terms & _ASSUMPTION_MARKERS),
        "qualification": bool(terms & _QUALIFICATION_MARKERS),
        "limitations": bool(terms & _LIMITATION_MARKERS),
        "evidence_type": bool(terms & _EVIDENCE_MARKERS),
        "what_not_established": "not" in terms or "cannot" in terms or "doesn" in terms,
    }
    return rules.get(requirement, _phrase_present(answer_text, requirement))


def _default_completeness_requirements(question: BenchmarkQuestion) -> tuple[str, ...]:
    if question.audience_level == "beginner":
        return ("direct_answer", "mechanism", "qualification")
    if question.audience_level == "undergraduate":
        return (
            "direct_answer",
            "mechanism",
            "tradeoff",
            "architecture_connection",
        )
    return (
        "direct_answer",
        "mechanism",
        "assumptions",
        "evidence_type",
        "limitations",
        "what_not_established",
    )


def grade_answer(
    question: BenchmarkQuestion,
    answer: GroundedAnswer,
    concepts: ConceptCoverage,
) -> AnswerGrade:
    """Grade relevance and completeness separately from citation validity."""
    if not isinstance(question, BenchmarkQuestion):
        raise TypeError("question must be BenchmarkQuestion.")
    if not isinstance(answer, GroundedAnswer):
        raise TypeError("answer must be GroundedAnswer.")
    if not isinstance(concepts, ConceptCoverage):
        raise TypeError("concepts must be ConceptCoverage.")
    prohibited = prohibited_claims_present(question, answer.answer_text)
    relevance, reasons, wrong_entity = _interrogative_relevance(
        question, answer.answer_text, concepts
    )
    requirements = (
        question.completeness_requirements
        or _default_completeness_requirements(question)
    )
    missing = tuple(
        item
        for item in requirements
        if not _requirement_present(item, answer.answer_text)
    )
    completeness = 1.0 - len(missing) / max(1, len(requirements))
    number_matches = sum(
        _phrase_present(answer.answer_text, value)
        for value in question.expected_numbers
    )
    numerical_accuracy = (
        1.0
        if not question.expected_numbers
        else number_matches / len(question.expected_numbers)
    )
    false_premise_accepted = (
        question.question_type == "false_premise"
        and not answer.abstained
        and not bool(set(tokenize_lexically(answer.answer_text)) & {"not", "false"})
    )
    expected_attempt = question.answerability == "paper_answerable"
    answerability_correct = expected_attempt != answer.abstained
    if question.answerability in {
        "external_sources_required",
        "unanswerable",
    }:
        answerability_correct = answer.abstained or bool(
            set(tokenize_lexically(answer.answer_text)) & _QUALIFICATION_MARKERS
        )
    return AnswerGrade(
        relevance=relevance,
        completeness=completeness,
        answerability_correct=float(answerability_correct),
        numerical_accuracy=numerical_accuracy,
        prohibited_claim_count=len(prohibited),
        false_premise_accepted=float(false_premise_accepted),
        wrong_entity_type=wrong_entity,
        missing_requirements=missing,
        relevance_reasons=reasons,
    )


def grade_citations(
    question: BenchmarkQuestion,
    answer: GroundedAnswer,
    *,
    index: RetrievalIndex,
) -> CitationGrade:
    """Separate syntax, existence, location, support, relevance, and coverage."""
    if not isinstance(question, BenchmarkQuestion):
        raise TypeError("question must be BenchmarkQuestion.")
    if not isinstance(answer, GroundedAnswer):
        raise TypeError("answer must be GroundedAnswer.")
    try:
        parse_inline_citations(answer.answer_text, strict=True)
        syntax_valid = True
    except CitationSyntaxError:
        syntax_valid = False
    evidence_by_label = {item.label: item for item in answer.evidence}
    cited_labels = {label for claim in answer.claims for label in claim.citation_labels}
    cited = tuple(
        evidence_by_label[label]
        for label in sorted(cited_labels)
        if label in evidence_by_label
    )
    existence_valid = len(cited_labels) == len(cited)
    valid_chunks = {item.chunk_id for item in index.chunks}
    location_correct = all(item.chunk_id in valid_chunks for item in cited)
    substantive = tuple(item for item in answer.claims if item.substantive)
    support_rate = (
        1.0
        if not substantive
        else sum(item.supported for item in substantive) / len(substantive)
    )
    relevant = set(question.relevant_chunk_ids)
    cited_ids = {item.chunk_id for item in cited}
    relevance_rate = (
        1.0
        if not cited_ids and not relevant
        else len(cited_ids & relevant) / max(1, len(cited_ids))
    )
    precision = relevance_rate
    recall = 1.0 if not relevant else len(cited_ids & relevant) / len(relevant)
    expected, forbidden = question_section_policy(question)
    wrong_section = sum(
        bool(forbidden)
        and any(
            label.casefold() in " ".join(item.heading_path).casefold()
            for label in forbidden
        )
        or bool(expected)
        and not any(
            label.casefold() in " ".join(item.heading_path).casefold()
            for label in expected
        )
        for item in cited
    )
    results = tuple(
        SearchResult(
            rank=item.retrieval_rank,
            score=item.retrieval_score,
            retrieval_method=item.retrieval_method,
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            source_name=item.source_name,
            title=item.title,
            authors=None,
            heading_path=item.heading_path,
            page_start=item.page_start,
            page_end=item.page_end,
            start_line=item.start_line,
            end_line=item.end_line,
            text=item.selected_text,
            matched_terms=item.matched_terms,
            semantic_query_terms=item.semantic_query_terms,
            term_contributions=(),
            scoring_details={},
            citation=item.citation,
        )
        for item in cited
    )
    boilerplate = sum(
        is_boilerplate_result(item, question_type=question.question_type)
        for item in results
    )
    return CitationGrade(
        syntax_valid=float(syntax_valid),
        existence_valid=float(existence_valid),
        source_location_correct=float(location_correct),
        support_rate=support_rate,
        relevance_rate=relevance_rate,
        precision=precision,
        recall=recall,
        coverage=answer.validation.citation_coverage,
        wrong_section_count=wrong_section,
        boilerplate_count=boilerplate,
    )
