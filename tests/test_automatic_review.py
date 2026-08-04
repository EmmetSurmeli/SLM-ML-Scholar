"""Tests for transparent deterministic automatic-review proposals."""

from __future__ import annotations

from localml_scholar.review_app.automation import (
    propose_automatic_failure_review,
    propose_automatic_review,
    summarize_automatic_reviews,
)
from localml_scholar.training_data import QuestionCandidate


def _candidate(question_type: str, **kwargs) -> QuestionCandidate:
    return QuestionCandidate.create(
        paper_ids=("paper-1",),
        question="What does the method do?",
        question_type=question_type,
        **kwargs,
    )


def _interaction(
    *,
    abstained: bool = False,
    accepted: bool = True,
    citation_coverage: float = 1.0,
    query_coverage: float = 1.0,
    unsupported: int = 0,
) -> dict:
    answer_text = (
        "The indexed sources state:\n"
        "- The method computes query-key scores. [C1]\n"
        "- It applies a softmax. [C1]"
    )
    return {
        "interaction_id": "interaction-1",
        "question": "What does the method do?",
        "paper_ids": ["paper-1"],
        "answer": {
            "answer_text": answer_text,
            "abstained": abstained,
            "evidence": [
                {
                    "evidence_id": "evidence-1",
                    "label": "C1",
                    "selected_text": "The method computes query-key scores.",
                }
            ],
            "validation": {
                "accepted": accepted,
                "citation_coverage": citation_coverage,
                "unsupported_claim_count": unsupported,
            },
            "sufficiency": {"query_term_coverage": query_coverage},
        },
        "diagnostics": {
            "accepted": accepted,
            "citation_coverage": citation_coverage,
            "query_term_coverage": query_coverage,
        },
        "comparison": {"requested": False, "complete": True},
    }


def test_explanatory_answer_is_cautiously_proposed_as_partial() -> None:
    review = propose_automatic_review(
        _interaction(), _candidate("teaching"), batch_id="batch-1"
    )
    assert review["proposed_label"] == "partial"
    assert review["needs_answer_edit"] is True
    assert review["decision"] == "pending_user_review"
    assert review["semantic_judge_used"] is False
    assert review["proposed_evidence_ids"] == ["evidence-1"]
    assert review["proposed_required_facts"] == [
        "The method computes query-key scores.",
        "It applies a softmax.",
    ]


def test_expected_abstention_is_reviewed_by_observed_behavior() -> None:
    answered = propose_automatic_review(
        _interaction(abstained=False),
        _candidate("false_premise"),
        batch_id="batch-1",
    )
    abstained = propose_automatic_review(
        _interaction(abstained=True),
        _candidate("insufficient_evidence"),
        batch_id="batch-1",
    )
    assert answered["proposed_label"] == "should_abstain"
    assert "insufficient" in answered["proposed_corrected_answer"].lower()
    assert abstained["proposed_label"] == "correct"


def test_failed_claim_validation_is_proposed_as_incorrect() -> None:
    review = propose_automatic_review(
        _interaction(accepted=False, unsupported=1),
        _candidate("fact_extraction"),
        batch_id="batch-1",
    )
    assert review["proposed_label"] == "incorrect"
    assert review["proposed_confidence"] == 0.84


def test_review_summary_keeps_automatic_and_user_states_distinct() -> None:
    pending = propose_automatic_review(
        _interaction(), _candidate("teaching"), batch_id="batch-1"
    )
    saved = dict(pending, review_id="saved", decision="saved_as_user_review")
    excluded = dict(pending, review_id="excluded", decision="excluded_by_user")
    summary = summarize_automatic_reviews([pending, saved, excluded])
    assert summary["review_count"] == 3
    assert summary["pending_user_review_count"] == 1
    assert summary["saved_review_count"] == 1
    assert summary["excluded_count"] == 1


def test_failed_answer_attempt_becomes_non_saveable_review_item() -> None:
    review = propose_automatic_failure_review(
        _candidate("metadata"),
        batch_id="batch-1",
        error=RuntimeError("citation validation failed"),
    )
    summary = summarize_automatic_reviews([review])

    assert review["proposed_label"] == "incorrect"
    assert review["saveable"] is False
    assert review["default_selected"] is False
    assert review["interaction_id"] is None
    assert review["diagnostics"]["execution_error"] == "citation validation failed"
    assert summary["execution_error_count"] == 1
