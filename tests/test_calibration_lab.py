"""Focused Milestone 12A.2 calibration-lab regression tests."""

from __future__ import annotations

import json

import pytest

from localml_scholar.review_app.automation import propose_automatic_review
from localml_scholar.review_app.service import ReviewService
from localml_scholar.training_data import (
    CalibrationPolicy,
    PaperAcquisitionItem,
    calibration_report,
    confidence_bucket,
    select_calibration_sample,
)
from localml_scholar.training_data.schemas import QuestionCandidate


def _review(
    identity: str,
    *,
    paper: str = "paper-a",
    question_type: str = "method",
    label: str = "correct",
    confidence: float = 0.96,
    approved: bool = True,
    mandatory: tuple[str, ...] = (),
) -> dict:
    return {
        "review_id": identity,
        "batch_id": "batch-a",
        "question_id": f"question-{identity}",
        "interaction_id": f"interaction-{identity}",
        "paper_ids": [paper],
        "question": f"Question {identity}?",
        "question_type": question_type,
        "answer": {
            "answer_text": "The paper states the method. [C1]",
            "abstained": False,
            "evidence": [
                {
                    "evidence_id": "chunk-a",
                    "label": "C1",
                    "selected_text": "The paper states the method.",
                }
            ],
        },
        "diagnostics": {},
        "proposed_label": label,
        "proposed_confidence": confidence,
        "proposed_required_facts": ["The paper states the method."],
        "proposed_prohibited_claims": [],
        "proposed_evidence_ids": ["chunk-a"],
        "second_pass": {
            "confidence": confidence,
            "review_status": "codex_approved" if approved else "needs_human_review",
            "would_approve_if_enabled": approved,
            "mandatory_human_categories": list(mandatory),
            "reviewer_results": [
                {
                    "reviewer_profile": "grounding",
                    "gates": {"citations_valid": True},
                }
            ],
        },
    }


def test_documented_confidence_buckets_cover_boundaries():
    assert confidence_bucket(0.50) == "0.50-0.69"
    assert confidence_bucket(0.70) == "0.70-0.79"
    assert confidence_bucket(0.95) == "0.95-0.97"
    assert confidence_bucket(1.0) == "0.98-1.00"
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        confidence_bucket(1.1)


def test_calibration_sample_is_deterministic_and_seeks_coverage():
    reviews = [
        _review("a", paper="paper-a", question_type="method", confidence=0.60),
        _review("b", paper="paper-b", question_type="experiment", confidence=0.75),
        _review("c", paper="paper-a", question_type="explanation", confidence=0.85),
        _review("d", paper="paper-b", question_type="metadata", confidence=0.96),
    ]
    first = select_calibration_sample(reviews, target_count=4, seed=7)
    second = select_calibration_sample(reviews, target_count=4, seed=7)
    assert first == second
    assert first["selected_count"] == 4
    assert first["coverage_gaps"] == []
    assert {item["question_type"] for item in first["items"]} == {
        "method",
        "experiment",
        "explanation",
        "metadata",
    }
    insufficient = select_calibration_sample(reviews, target_count=50, seed=7)
    assert insufficient["selected_count"] == 4
    assert "smaller" in " ".join(insufficient["warnings"])


def test_readiness_emphasizes_false_approvals_and_integrity():
    records = [
        {
            "confidence": 0.99,
            "automated_approved": True,
            "human_approved": position != 0,
        }
        for position in range(50)
    ]
    report = calibration_report(records)
    assert report["false_approval_count"] == 1
    assert report["false_approval_rate"] == pytest.approx(0.02)
    assert report["auto_approval_precision"] == pytest.approx(0.98)
    assert report["metrics_pass"]
    blocked = calibration_report(records, integrity={"source_hash_errors": 1})
    assert not blocked["checks"]["source_hashes"]
    assert "Source-hash" in " ".join(blocked["reasons"])


def test_readiness_suspends_excessive_false_approvals():
    records = [
        {
            "confidence": 0.99,
            "automated_approved": True,
            "human_approved": position >= 5,
        }
        for position in range(50)
    ]
    report = calibration_report(
        records,
        policy=CalibrationPolicy(maximum_brier_score=1.0),
        explicit_enable=True,
    )
    assert report["state"] == "auto_approval_suspended"
    assert not report["checks"]["false_approval_rate"]


def test_service_sample_and_one_click_decision_are_separate_from_training(tmp_path):
    service = ReviewService(tmp_path)
    review = _review("one")
    service._save_automatic_review_batches(  # noqa: SLF001 - persistence fixture
        [{"batch_id": "batch-a", "reviews": [review]}]
    )
    sample = service.create_calibration_sample(target_count=1, seed=2)
    assert sample["review_ids"] == ["one"]
    pair = service.record_calibration_decision(
        review_id="one", action="approve_auto", reviewer="human-a"
    )
    assert pair["human_label"] == "correct"
    assert pair["training_approved"] is False
    assert not service.corrections_path.exists()
    state = service.state()
    assert state["calibration_pairs"][0]["pair_id"] == pair["pair_id"]
    with pytest.raises(ValueError, match="already has"):
        service.record_calibration_decision(
            review_id="one", action="override_partial", reviewer="human-a"
        )


def test_calibration_edit_preserves_before_after_hashes(tmp_path):
    service = ReviewService(tmp_path)
    service._save_automatic_review_batches(  # noqa: SLF001 - persistence fixture
        [{"batch_id": "batch-a", "reviews": [_review("edit")]}]
    )
    service.create_calibration_sample(target_count=1)
    pair = service.record_calibration_decision(
        review_id="edit",
        action="override_partial",
        reviewer="human-a",
        edits={
            "answer_text": "A clearer explanation. [C1]",
            "required_facts": ["Explain the mechanism."],
        },
    )
    assert pair["edited"]
    assert pair["correction_revalidated"]
    assert pair["revalidation"] is not None
    assert pair["original_snapshot_hash"] != pair["reviewed_snapshot_hash"]
    assert pair["reviewed_snapshot"]["answer"]["answer_text"].startswith("A clearer")


@pytest.mark.parametrize(
    ("action", "expected_label", "expected_status"),
    (
        ("override_correct", "correct", "finalized"),
        ("override_partial", "partial", "finalized"),
        ("override_incorrect", "incorrect", "finalized"),
        ("override_should_abstain", "should_abstain", "finalized"),
        ("benchmark_problem", "benchmark_problem", "finalized"),
        ("skip", None, "skipped"),
    ),
)
def test_every_one_click_human_action_is_persisted(
    tmp_path, action, expected_label, expected_status
):
    service = ReviewService(tmp_path)
    service._save_automatic_review_batches(  # noqa: SLF001 - persistence fixture
        [{"batch_id": "batch-a", "reviews": [_review(action)]}]
    )
    result = service.record_calibration_decision(
        review_id=action, action=action, reviewer="human-a"
    )
    assert result["status"] == expected_status
    if expected_label is not None:
        assert result["human_label"] == expected_label


def test_zero_denominator_metrics_are_finite_and_explicit():
    records = [
        {
            "confidence": 0.6,
            "automated_approved": False,
            "human_approved": False,
        }
    ]
    report = calibration_report(records)
    assert report["auto_approval_precision"] == 0.0
    assert report["auto_approval_recall"] == 0.0
    assert report["false_approval_rate"] == 0.0
    assert report["rejection_agreement"] == 1.0


def test_mandatory_human_and_integrity_failures_block_readiness():
    records = [
        {
            "confidence": 0.99,
            "automated_approved": True,
            "human_approved": True,
            "mandatory_human_categories": ["inferred_derivation"],
        }
        for _ in range(50)
    ]
    report = calibration_report(
        records,
        integrity={"test_leakage_errors": 1, "provenance_errors": 1},
    )
    assert not report["checks"]["mandatory_human_routing"]
    assert not report["checks"]["test_leakage"]
    assert not report["checks"]["provenance"]
    assert not report["metrics_pass"]


def test_manual_acquisition_queue_never_fetches_and_deduplicates(tmp_path):
    service = ReviewService(tmp_path)
    item = service.add_acquisition_item(
        title="A Useful Paper",
        doi="10.1/example",
        arxiv_id=None,
        citation="Author (2025)",
        reason="Adds optimizer coverage.",
        category="optimization",
    )
    assert item["fetch_performed"] is False
    assert (
        service.update_acquisition_item(item_id=item["item_id"], status="obtained")[
            "status"
        ]
        == "obtained"
    )
    with pytest.raises(ValueError, match="already"):
        service.add_acquisition_item(
            title="A Useful Paper",
            doi="10.1/example",
            arxiv_id=None,
            citation="Author (2025)",
            reason="Duplicate.",
            category="optimization",
        )
    persisted = json.loads(service.acquisition_queue_path.read_text(encoding="utf-8"))
    assert len(persisted) == 1


def test_acquisition_schema_rejects_invalid_status():
    with pytest.raises(ValueError, match="status"):
        PaperAcquisitionItem(
            title="Paper", reason="Coverage", category="method", status="download"
        )


def test_bulk_review_remains_locked_before_explicit_enable(tmp_path):
    service = ReviewService(tmp_path)
    with pytest.raises(ValueError, match="locked"):
        service.bulk_auto_review(eligible_only=True)


def test_historical_rerun_is_linked_and_does_not_mutate_original(tmp_path):
    service = ReviewService(tmp_path)
    paper = service.add_paper(
        filename="paper.md",
        payload=b"# Paper\n\n## Method\nTraining uses Adam optimizer.\n",
    )
    candidate_state = service.add_question(
        question="Which optimizer is used?",
        paper_ids=(paper["document_id"],),
        question_type="metadata",
    )
    candidate = QuestionCandidate.from_dict(candidate_state)
    interaction = service.run_question(candidate.question_id)
    original = propose_automatic_review(
        interaction, candidate, batch_id="historical-batch"
    )
    service._save_automatic_review_batches(  # noqa: SLF001 - persistence fixture
        [{"batch_id": "historical-batch", "reviews": [original]}]
    )
    before = service.automatic_reviews_path.read_bytes()
    result = service.rerun_historical_reviews(review_ids=(original["review_id"],))
    assert result["rerun_count"] == 1
    assert service.automatic_reviews_path.read_bytes() == before
    rerun = result["reruns"][0]
    assert rerun["original_review_id"] == original["review_id"]
    assert rerun["original_snapshot"] == original
    assert rerun["non_destructive"] is True
    assert rerun["original_snapshot_hash"] != ""
    assert rerun["new_snapshot_hash"] != ""
