from __future__ import annotations

from dataclasses import replace

import pytest

from localml_scholar.answering import GroundedAnswerPipeline
from localml_scholar.evaluation.answer_grading import (
    grade_answer,
    grade_citations,
    grade_concepts,
    grade_sufficiency,
    prohibited_claims_present,
)
from localml_scholar.evaluation.audience import (
    factual_basis_is_preserved,
    grade_audience,
    render_beginner_answer,
    render_for_audience,
    render_researcher_answer,
    render_undergraduate_answer,
    structured_target_from_answer,
)
from localml_scholar.evaluation.retrieval_grading import (
    grade_retrieval,
    is_boilerplate_result,
)
from localml_scholar.evaluation.schemas import (
    CitedAnswerPoint,
    ConceptGroup,
    StructuredAnswerTarget,
)
from tests.test_real_paper_evaluation_schemas import evaluation_question


def _ranked(results):
    return tuple(
        replace(item, rank=position) for position, item in enumerate(results, 1)
    )


def test_retrieval_metrics_gold_at_rank_one_and_three(grounded_index):
    question = evaluation_question(grounded_index)
    results = grounded_index.search(
        "causal mask future positions",
        method="bm25",
        top_k=5,
    )
    gold = next(
        item for item in results if item.chunk_id in question.relevant_chunk_ids
    )
    others = tuple(item for item in results if item.chunk_id != gold.chunk_id)
    at_one = grade_retrieval(
        question,
        _ranked((gold, *others)),
        index=grounded_index,
    )
    distractors = tuple(grounded_index.search("gradient probability learning", top_k=3))
    at_three = grade_retrieval(
        question,
        _ranked((*distractors[:2], gold, *others)),
        index=grounded_index,
    )
    assert at_one.recall_at_1 == 1.0
    assert at_one.reciprocal_rank == 1.0
    assert at_three.recall_at_1 == 0.0
    assert at_three.recall_at_3 == 1.0
    assert at_three.reciprocal_rank == pytest.approx(1 / 3)


def test_missing_gold_and_acceptable_alternate(grounded_index):
    question = evaluation_question(grounded_index)
    alternate = next(
        item
        for item in grounded_index.chunks
        if item.text.startswith("g\nautoregressive prediction")
    )
    question = replace(
        question,
        acceptable_chunk_ids=(alternate.chunk_id,),
    )
    results = tuple(
        replace(item, rank=1)
        for item in grounded_index.search("autoregressive prediction", top_k=3)
        if item.chunk_id == alternate.chunk_id
    )
    assert len(results) == 1
    grade = grade_retrieval(question, results, index=grounded_index)
    assert grade.hit_rate_at_k == 1.0
    assert question.gold_evidence[0].chunk_id in grade.missed_gold_chunk_ids


def test_forbidden_boilerplate_and_question_context(grounded_index):
    result = grounded_index.search("reproducibility configuration", top_k=1)[0]
    question = evaluation_question(
        grounded_index,
        forbidden_sections=("reproducibility",),
    )
    grade = grade_retrieval(question, (result,), index=grounded_index)
    assert grade.forbidden_section_rate == 1.0
    assert grade.wrong_section_chunk_ids == (result.chunk_id,)

    reference_like = replace(
        result,
        heading_path=("Paper", "References"),
    )
    assert is_boilerplate_result(reference_like, question_type="architecture")
    assert not is_boilerplate_result(
        reference_like,
        question_type="historical_impact",
    )


def test_metadata_and_motivation_section_signals(grounded_index):
    title = next(item for item in grounded_index.chunks if item.start_line == 1)
    title_result = grounded_index.search("causal attention", top_k=1)[0]
    title_result = replace(
        title_result,
        chunk_id=title.chunk_id,
        document_id=title.document_id,
        heading_path=title.heading_path,
        start_line=title.start_line,
        end_line=title.end_line,
        text=title.text,
        citation=replace(
            title_result.citation,
            chunk_id=title.chunk_id,
            document_id=title.document_id,
            heading_path=title.heading_path,
            start_line=title.start_line,
            end_line=title.end_line,
        ),
    )
    question = evaluation_question(
        grounded_index,
        question="What is the paper title?",
        question_type="metadata",
        expected_sections=("causal attention",),
        gold_evidence=(),
        acceptable_chunk_ids=(title.chunk_id,),
    )
    grade = grade_retrieval(question, (title_result,), index=grounded_index)
    assert grade.title_page_hit == 1.0


def test_graded_relevance_ndcg_prefers_higher_grade(grounded_index):
    question = evaluation_question(grounded_index)
    results = grounded_index.search("causal mask future positions", top_k=5)
    gold = next(
        item for item in results if item.chunk_id in question.relevant_chunk_ids
    )
    other = next(item for item in results if item.chunk_id != gold.chunk_id)
    alternate_question = replace(
        question,
        acceptable_chunk_ids=(other.chunk_id,),
    )
    better = grade_retrieval(
        alternate_question,
        _ranked((gold, other)),
        index=grounded_index,
        k=2,
    )
    worse = grade_retrieval(
        alternate_question,
        _ranked((other, gold)),
        index=grounded_index,
        k=2,
    )
    assert better.ndcg_at_k > worse.ndcg_at_k


def test_concept_coverage_aliases_citations_and_support(grounded_index):
    answer = GroundedAnswerPipeline(grounded_index).answer(
        "How does a causal mask block future positions?",
        method="top_passage",
        top_k=4,
    )
    question = evaluation_question(
        grounded_index,
        required_concepts=(
            ConceptGroup("future tokens", aliases=("future positions",)),
        ),
        optional_concepts=(ConceptGroup("softmax"),),
    )
    coverage = grade_concepts(question, answer)
    assert coverage.required_recall == 1.0
    assert coverage.present_required == ("future tokens",)
    assert coverage.uncited_required == ()
    assert coverage.unsupported_required == ()


def test_empty_concept_requirements_have_defined_score(grounded_index):
    answer = GroundedAnswerPipeline(grounded_index).answer(
        "How does a causal mask work?",
        method="top_passage",
    )
    coverage = grade_concepts(
        evaluation_question(
            grounded_index,
            required_concepts=(),
            optional_concepts=(),
        ),
        answer,
    )
    assert coverage.required_recall == coverage.optional_recall == 1.0


def test_prohibited_claim_detection_is_explicit(grounded_index):
    question = evaluation_question(
        grounded_index,
        prohibited_claims=("Transformers are always superior",),
    )
    assert prohibited_claims_present(
        question,
        "Transformers are always superior.",
    ) == ("Transformers are always superior",)
    assert prohibited_claims_present(question, "The evidence is limited.") == ()


@pytest.mark.parametrize(
    ("prompt", "expected_minimum", "wrong_entity"),
    [
        ("How does a causal mask block future positions?", 0.5, 0.0),
        ("Who wrote the paper?", 0.0, 1.0),
    ],
)
def test_interrogative_answer_relevance(
    grounded_index,
    prompt,
    expected_minimum,
    wrong_entity,
):
    answer = GroundedAnswerPipeline(grounded_index).answer(
        "How does a causal mask block future positions?",
        method="top_passage",
    )
    question = evaluation_question(
        grounded_index,
        question=prompt,
        required_concepts=(),
        completeness_requirements=("direct_answer",),
    )
    concepts = grade_concepts(question, answer)
    grade = grade_answer(question, answer, concepts)
    assert grade.relevance >= expected_minimum
    assert grade.wrong_entity_type == wrong_entity


def test_numerical_and_sufficiency_grades(grounded_index):
    answer = GroundedAnswerPipeline(grounded_index).answer(
        "What context length and learning rate are used?",
        method="top_passage",
    )
    question = evaluation_question(
        grounded_index,
        question="What context length and learning rate are used?",
        question_type="hyperparameter",
        expected_numbers=("128", "0.01"),
        required_concepts=(),
    )
    grade = grade_answer(question, answer, grade_concepts(question, answer))
    assert grade.numerical_accuracy == 1.0
    assert grade_sufficiency(question, answer).correct == 1.0


def test_citation_grade_keeps_validity_distinct_from_relevance(grounded_index):
    answer = GroundedAnswerPipeline(grounded_index).answer(
        "What context length and learning rate are used?",
        method="top_passage",
    )
    unrelated_question = evaluation_question(
        grounded_index,
        question="Who wrote the paper?",
        question_type="metadata",
        required_concepts=(),
        gold_evidence=(),
        acceptable_chunk_ids=(
            next(
                item.chunk_id
                for item in grounded_index.chunks
                if "Causal Attention" in item.text
            ),
        ),
    )
    grade = grade_citations(
        unrelated_question,
        answer,
        index=grounded_index,
    )
    assert grade.syntax_valid == 1.0
    assert grade.existence_valid == 1.0
    assert grade.relevance_rate == 0.0


def _target() -> StructuredAnswerTarget:
    return StructuredAnswerTarget(
        core_answer=CitedAnswerPoint(
            "A causal mask means later positions are blocked.",
            ("C1",),
        ),
        supporting_points=(
            CitedAnswerPoint(
                "It works by preventing attention to future tokens.",
                ("C1",),
            ),
        ),
        assumptions=(
            CitedAnswerPoint(
                "This assumes an autoregressive decoding order.",
                ("C2",),
            ),
        ),
        qualifications=(
            CitedAnswerPoint(
                "The claim only concerns causal information flow.",
                ("C2",),
            ),
        ),
        limitations=(
            CitedAnswerPoint(
                "It does not establish broader model quality.",
                ("C2",),
            ),
        ),
    )


def test_audience_renderers_preserve_one_factual_basis_and_citations():
    target = _target()
    rendered = (
        render_beginner_answer(target),
        render_undergraduate_answer(target),
        render_researcher_answer(target),
    )
    assert factual_basis_is_preserved(target, rendered)
    assert all("[C1]" in item for item in rendered)
    assert rendered == tuple(
        render_for_audience(target, level)
        for level in ("beginner", "undergraduate", "researcher")
    )
    assert len(rendered[0]) < len(rendered[2])


@pytest.mark.parametrize(
    ("level", "text", "expected_reason"),
    [
        (
            "beginner",
            "Attention uses a tensor representation parameterization.",
            "beginner_definition_missing",
        ),
        (
            "undergraduate",
            "Attention exists.",
            "undergraduate_mechanism_missing",
        ),
        (
            "researcher",
            "The mechanism works by masking values.",
            "researcher_qualification_missing",
        ),
    ],
)
def test_audience_grade_exposes_transparent_failures(
    grounded_index,
    level,
    text,
    expected_reason,
):
    question = evaluation_question(
        grounded_index,
        audience_level=level,
    )
    assert expected_reason in grade_audience(question, text).reasons


def test_structured_target_comes_only_from_supported_cited_claims(grounded_index):
    answer = GroundedAnswerPipeline(grounded_index).answer(
        "How does a causal mask block future positions?",
        method="top_passage",
    )
    target = structured_target_from_answer(answer)
    assert target.core_answer.citations
    assert factual_basis_is_preserved(
        target,
        tuple(
            render_for_audience(target, level)
            for level in ("beginner", "undergraduate", "researcher")
        ),
    )
