"""Multi-label failure taxonomy and cautious likely root-cause attribution."""

from __future__ import annotations

from localml_scholar.answering import GroundedAnswer
from localml_scholar.evaluation.schemas import (
    AnswerGrade,
    AudienceGrade,
    BenchmarkQuestion,
    CitationGrade,
    ConceptCoverage,
    RetrievalGrade,
    RootCauseAttribution,
    SufficiencyGrade,
)

FAILURE_CATEGORIES = frozenset(
    {
        "retrieval_wrong_section",
        "retrieval_irrelevant_passage",
        "retrieval_missed_gold_evidence",
        "retrieval_boilerplate",
        "retrieval_redundant",
        "sufficiency_false_positive",
        "sufficiency_false_negative",
        "answer_not_relevant",
        "answer_incomplete",
        "answer_overgeneralized",
        "answer_wrong_entity_type",
        "answer_wrong_number",
        "answer_false_premise_accepted",
        "citation_invalid",
        "citation_irrelevant",
        "citation_wrong_location",
        "citation_incomplete",
        "required_concept_missing",
        "prohibited_claim_present",
        "audience_too_technical",
        "audience_too_shallow",
        "external_context_required",
        "generation_malformed",
        "fallback_used",
        "correct_abstention",
    }
)


def categorize_failures(
    question: BenchmarkQuestion,
    retrieval: RetrievalGrade,
    *,
    answer_object: GroundedAnswer | None,
    sufficiency: SufficiencyGrade | None,
    answer: AnswerGrade | None,
    concepts: ConceptCoverage | None,
    citations: CitationGrade | None,
    audience: AudienceGrade | None,
) -> tuple[str, ...]:
    """Assign every applicable deterministic category in stable policy order."""
    if not isinstance(question, BenchmarkQuestion):
        raise TypeError("question must be BenchmarkQuestion.")
    if not isinstance(retrieval, RetrievalGrade):
        raise TypeError("retrieval must be RetrievalGrade.")
    categories: list[str] = []
    if retrieval.wrong_section_chunk_ids:
        categories.append("retrieval_wrong_section")
    if retrieval.irrelevant_positive_score_rate > 0.5:
        categories.append("retrieval_irrelevant_passage")
    if retrieval.missed_gold_chunk_ids:
        categories.append("retrieval_missed_gold_evidence")
    if retrieval.boilerplate_chunk_ids:
        categories.append("retrieval_boilerplate")
    if retrieval.evidence_redundancy > 0.0:
        categories.append("retrieval_redundant")
    if sufficiency is not None:
        if sufficiency.false_answer:
            categories.append("sufficiency_false_positive")
        if sufficiency.false_abstention:
            categories.append("sufficiency_false_negative")
    if answer is not None:
        if answer.relevance < 0.5:
            categories.append("answer_not_relevant")
        if answer.completeness < 0.75:
            categories.append("answer_incomplete")
        if answer.prohibited_claim_count:
            categories.extend(["answer_overgeneralized", "prohibited_claim_present"])
        if answer.wrong_entity_type:
            categories.append("answer_wrong_entity_type")
        if answer.numerical_accuracy < 1.0:
            categories.append("answer_wrong_number")
        if answer.false_premise_accepted:
            categories.append("answer_false_premise_accepted")
    if concepts is not None and (
        concepts.missing_required
        or concepts.uncited_required
        or concepts.unsupported_required
    ):
        categories.append("required_concept_missing")
    if citations is not None:
        if not citations.syntax_valid or not citations.existence_valid:
            categories.append("citation_invalid")
        if citations.relevance_rate < 1.0:
            categories.append("citation_irrelevant")
        if not citations.source_location_correct or citations.wrong_section_count:
            categories.append("citation_wrong_location")
        if citations.coverage < 1.0 or citations.recall < 1.0:
            categories.append("citation_incomplete")
    if audience is not None and audience.appropriateness < 0.6:
        if question.audience_level == "beginner":
            categories.append("audience_too_technical")
        else:
            categories.append("audience_too_shallow")
    if question.answerability == "external_sources_required":
        categories.append("external_context_required")
    if answer_object is not None:
        if (
            answer_object.method
            in {
                "generative",
                "generative_with_extractive_fallback",
            }
            and not answer_object.validation.accepted
        ):
            categories.append("generation_malformed")
        if answer_object.fallback_used:
            categories.append("fallback_used")
        if (
            answer_object.abstained
            and question.paper_sufficiency in {"insufficient", "external_required"}
            and sufficiency is not None
            and sufficiency.correct
        ):
            categories.append("correct_abstention")
    return tuple(dict.fromkeys(categories))


def root_cause_attribution(
    categories: tuple[str, ...],
) -> RootCauseAttribution:
    """Assign a likely cause without representing heuristic attribution as fact."""
    if not isinstance(categories, tuple) or any(
        item not in FAILURE_CATEGORIES for item in categories
    ):
        raise ValueError("categories must contain recognized failure labels.")
    groups = [
        (
            "retrieval",
            {
                "retrieval_wrong_section",
                "retrieval_irrelevant_passage",
                "retrieval_missed_gold_evidence",
                "retrieval_boilerplate",
                "retrieval_redundant",
            },
            (
                "Inspect chunking, retrieval terms, fusion, and "
                "question-specific section policy."
            ),
        ),
        (
            "sufficiency",
            {"sufficiency_false_positive", "sufficiency_false_negative"},
            "Calibrate evidence sufficiency against approved paper-only labels.",
        ),
        (
            "answer_construction",
            {
                "answer_not_relevant",
                "answer_incomplete",
                "answer_overgeneralized",
                "answer_wrong_entity_type",
                "answer_wrong_number",
                "answer_false_premise_accepted",
                "required_concept_missing",
                "prohibited_claim_present",
            },
            (
                "Revise structured answer construction and required/prohibited "
                "content checks."
            ),
        ),
        (
            "citation_validation",
            {
                "citation_invalid",
                "citation_irrelevant",
                "citation_wrong_location",
                "citation_incomplete",
            },
            "Inspect claim-to-evidence attachment and citation relevance separately.",
        ),
        (
            "audience_rendering",
            {"audience_too_technical", "audience_too_shallow"},
            "Revise the deterministic renderer without changing its factual target.",
        ),
        (
            "generation",
            {"generation_malformed", "fallback_used"},
            "Inspect raw local generation and keep validated fallback enabled.",
        ),
    ]
    matched = [
        (cause, labels & set(categories), action)
        for cause, labels, action in groups
        if labels & set(categories)
    ]
    if not matched:
        if "external_context_required" in categories:
            return RootCauseAttribution(
                primary_cause="benchmark_ambiguity",
                secondary_causes=(),
                reasons=("paper_boundary_requires_external_context",),
                confidence="high",
                recommended_next_action=(
                    "Retain qualification or abstention and review with external "
                    "evidence only in a separate benchmark."
                ),
            )
        return RootCauseAttribution(
            primary_cause="none",
            secondary_causes=(),
            reasons=("no_automatic_failure_detected",),
            confidence="medium",
            recommended_next_action="Retain a deterministic sample for human review.",
        )
    primary, primary_labels, action = matched[0]
    secondary = tuple(item[0] for item in matched[1:])
    confidence = "high" if len(matched) == 1 else "medium"
    return RootCauseAttribution(
        primary_cause=primary,
        secondary_causes=secondary,
        reasons=tuple(sorted(primary_labels)),
        confidence=confidence,
        recommended_next_action=action,
    )


def is_automatic_pass(categories: tuple[str, ...]) -> bool:
    """Return whether no failure category is present."""
    informational = {"correct_abstention"}
    return not (set(categories) - informational)
