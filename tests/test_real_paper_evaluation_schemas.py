from __future__ import annotations

import json
from dataclasses import replace

import pytest

from localml_scholar.evaluation import (
    Benchmark,
    BenchmarkQuestion,
    ConceptGroup,
    EvaluationConfig,
    GoldEvidence,
)
from localml_scholar.evaluation.benchmark import apply_review_decisions
from localml_scholar.evaluation.serialization import (
    load_benchmark,
    save_benchmark,
)


def evaluation_question(grounded_index, **changes) -> BenchmarkQuestion:
    chunk = next(
        item
        for item in grounded_index.chunks
        if "causal mask before softmax" in item.text
    )
    defaults = {
        "paper_id": chunk.document_id,
        "question": "How does a causal mask prevent future-token leakage?",
        "question_type": "architecture",
        "audience_level": "undergraduate",
        "answerability": "paper_answerable",
        "paper_sufficiency": "sufficient",
        "expected_sections": ("masking future positions",),
        "forbidden_sections": ("references",),
        "gold_evidence": (
            GoldEvidence(
                chunk_id=chunk.chunk_id,
                section_id=chunk.section_id,
                relevance_grade=3,
            ),
        ),
        "required_concepts": (
            ConceptGroup("causal mask"),
            ConceptGroup("future positions", aliases=("future tokens",)),
        ),
        "optional_concepts": (ConceptGroup("softmax"),),
        "prohibited_claims": ("future positions are visible",),
        "completeness_requirements": ("direct_answer", "mechanism"),
        "gold_core_answer": (
            "The causal mask blocks attention from each position to later positions."
        ),
        "gold_notes": "Authored from the masking section.",
        "review_status": "approved",
    }
    defaults.update(changes)
    return BenchmarkQuestion.create(**defaults)


def evaluation_benchmark(grounded_index, *questions) -> Benchmark:
    selected = questions or (evaluation_question(grounded_index),)
    documents = {item.document_id: item for item in grounded_index.documents}
    paper_ids = {item.paper_id for item in selected}
    return Benchmark(
        name="Authored causal-attention benchmark",
        benchmark_version="1.0",
        index_sha256=grounded_index.index_sha256,
        document_hashes={
            paper_id: documents[paper_id].content_sha256
            for paper_id in sorted(paper_ids)
        },
        questions=tuple(selected),
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("paper_id", "", "paper_id"),
        ("audience_level", "expert", "audience_level"),
        ("question_type", "trivia", "question_type"),
        ("answerability", "maybe", "answerability"),
        ("paper_sufficiency", "mostly", "paper_sufficiency"),
    ],
)
def test_question_rejects_invalid_canonical_fields(
    grounded_index,
    field,
    value,
    message,
):
    with pytest.raises(ValueError, match=message):
        evaluation_question(grounded_index, **{field: value})


def test_answerability_and_paper_sufficiency_cannot_contradict(grounded_index):
    with pytest.raises(ValueError, match="contradictory"):
        evaluation_question(
            grounded_index,
            answerability="external_sources_required",
            paper_sufficiency="sufficient",
        )


def test_approved_paper_question_requires_evidence_and_notes(grounded_index):
    with pytest.raises(ValueError, match="gold evidence"):
        evaluation_question(grounded_index, gold_evidence=())
    with pytest.raises(ValueError, match="gold_notes"):
        evaluation_question(grounded_index, gold_notes=None)


def test_proposed_question_is_not_official_gold(grounded_index):
    question = evaluation_question(
        grounded_index,
        review_status="proposed",
        gold_notes=None,
    )
    benchmark = evaluation_benchmark(grounded_index, question)
    assert benchmark.approved_questions == ()
    assert question.prohibited_claims == ("future positions are visible",)


def test_duplicate_question_ids_are_rejected(grounded_index):
    question = evaluation_question(grounded_index)
    with pytest.raises(ValueError, match="unique"):
        evaluation_benchmark(grounded_index, question, question)


def test_benchmark_round_trip_and_hash_tampering(grounded_index, tmp_path):
    benchmark = evaluation_benchmark(grounded_index)
    path = save_benchmark(benchmark, tmp_path / "benchmark.json")

    loaded = load_benchmark(path, index=grounded_index)
    assert loaded == benchmark
    assert loaded.benchmark_sha256 == benchmark.benchmark_sha256

    state = json.loads(path.read_text(encoding="utf-8"))
    state["payload"]["name"] = "tampered"
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_benchmark(path)


def test_stale_chunk_and_source_are_rejected(grounded_index):
    benchmark = evaluation_benchmark(grounded_index)
    question = replace(
        benchmark.questions[0],
        gold_evidence=(GoldEvidence("chk_" + "0" * 24),),
    )
    stale_chunk = replace(benchmark, questions=(question,))
    with pytest.raises(ValueError, match="stale gold chunk"):
        stale_chunk.validate_against_index(grounded_index)
    stale_source = replace(
        benchmark,
        document_hashes={
            benchmark.questions[0].paper_id: "0" * 64,
        },
    )
    with pytest.raises(ValueError, match="source document hash"):
        stale_source.validate_against_index(grounded_index)


def test_review_decisions_can_replace_gold_evidence(grounded_index):
    proposed = evaluation_question(
        grounded_index,
        review_status="proposed",
        gold_evidence=(),
        gold_notes=None,
    )
    benchmark = evaluation_benchmark(grounded_index, proposed)
    chunk = next(item for item in grounded_index.chunks if "Scaled Scores" in item.text)
    reviewed = apply_review_decisions(
        benchmark,
        {
            proposed.question_id: {
                "status": "edited",
                "gold_notes": "Reviewer selected exact scaled-score evidence.",
                "edits": {
                    "gold_evidence": [
                        GoldEvidence(
                            chunk_id=chunk.chunk_id,
                            section_id=chunk.section_id,
                        ).to_dict()
                    ],
                    "required_concepts": [ConceptGroup("score scaling").to_dict()],
                },
            }
        },
    )
    assert reviewed.approved_questions[0].gold_evidence[0].chunk_id == chunk.chunk_id
    assert reviewed.approved_questions[0].review_status == "edited"


def test_review_omissions_never_auto_approve(grounded_index):
    proposed = evaluation_question(
        grounded_index,
        review_status="proposed",
        gold_notes=None,
    )
    benchmark = evaluation_benchmark(grounded_index, proposed)
    assert apply_review_decisions(benchmark, {}).approved_questions == ()


def test_evaluation_configuration_records_comparison_controls():
    config = EvaluationConfig(
        retrieval_parameters={"top_k": 5, "fusion": "rrf"},
        evidence_selection_settings={"evidence_top_k": 3},
        sufficiency_settings={"minimum_score": 0.5},
        acceptance_policy={"require_citations": True},
    )
    assert EvaluationConfig.from_dict(config.to_dict()) == config
    with pytest.raises(TypeError, match="retrieval_parameters"):
        EvaluationConfig(retrieval_parameters=[])  # type: ignore[arg-type]
