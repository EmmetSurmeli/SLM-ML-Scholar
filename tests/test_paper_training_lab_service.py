"""End-to-end service tests for the Milestone 12A Paper Training Lab."""

from __future__ import annotations

import json

import pytest

from localml_scholar.review_app.service import ReviewService
from localml_scholar.review_app.storage import atomic_write_json
from localml_scholar.training_data import load_dataset

_PAPER_A = b"""# Alpha Attention

## Method
The Alpha method computes attention weights from query-key dot products.
Training uses Adam with learning rate 0.001.

## Limitations
Alpha was evaluated only on synthetic sequences.
"""

_PAPER_B = b"""# Beta Memory

## Method
The Beta method uses blockwise memory to reduce storage during attention.
Training uses stochastic gradient descent.

## Results
Beta reports lower peak memory on long sequences.
"""


def _service_with_papers(tmp_path):
    service = ReviewService(tmp_path)
    alpha = service.add_paper(filename="alpha.md", payload=_PAPER_A)
    beta = service.add_paper(filename="beta.md", payload=_PAPER_B)
    return service, alpha, beta


def test_session_is_memory_only_unless_persistence_is_explicit(tmp_path):
    service, alpha, _beta = _service_with_papers(tmp_path)
    ephemeral = service.create_session(
        selected_paper_ids=(alpha["document_id"],),
        preferences={"verbosity": "concise"},
    )
    assert service.get_session(ephemeral["session_id"]).preferences == {
        "verbosity": "concise"
    }
    assert not service.sessions_path.exists()

    persisted = service.create_session(
        preferences={"use_analogy": True}, persist_preferences=True
    )
    state = json.loads(service.sessions_path.read_text(encoding="utf-8"))
    assert state[0]["session_id"] == persisted["session_id"]
    assert state[0]["preferences"] == {"use_analogy": True}


def test_state_normalizes_legacy_single_document_interactions(tmp_path):
    service, alpha, _beta = _service_with_papers(tmp_path)
    atomic_write_json(
        service.interactions_path,
        [
            {
                "interaction_id": "legacy-interaction",
                "document_id": alpha["document_id"],
                "question": "Legacy question",
                "answer": {"answer_text": "Legacy answer", "evidence": []},
            }
        ],
    )
    interaction = service.state()["interactions"][0]
    assert interaction["paper_ids"] == [alpha["document_id"]]


def test_adaptive_multi_turn_ask_records_profile_and_context(tmp_path):
    service, alpha, _beta = _service_with_papers(tmp_path)
    session = service.create_session(selected_paper_ids=(alpha["document_id"],))
    first = service.ask(
        question="As an undergrad, explain the optimizer concisely.",
        session_id=session["session_id"],
    )
    second = service.ask(
        question="I don't understand. Explain that more simply.",
        session_id=session["session_id"],
    )
    assert first["instruction_profile"]["canonical_audience"] == "undergraduate"
    assert second["instruction_profile"]["simplify_previous"] is True
    assert len(service.get_session(session["session_id"]).turns) == 4
    assert second["evidence_selection"]["independent_of_instruction_profile"] is True


def test_cross_paper_question_reports_source_coverage_without_fabrication(tmp_path):
    service, alpha, beta = _service_with_papers(tmp_path)
    interaction = service.ask(
        question="Compare the optimizer in Alpha with Beta.",
        document_ids=(alpha["document_id"], beta["document_id"]),
    )
    assert interaction["instruction_profile"]["include_comparison"] is True
    assert interaction["comparison"]["requested"] is True
    assert set(interaction["paper_ids"]) == {
        alpha["document_id"],
        beta["document_id"],
    }
    if not interaction["comparison"]["complete"]:
        assert "incomplete" in interaction["comparison"]["warning"]

    single_source = service.ask(
        question="Compare Alpha with a missing external method.",
        document_id=alpha["document_id"],
    )
    assert single_source["comparison"]["complete"] is False
    assert single_source["comparison"]["missing_source_count"] == 1


def test_question_generation_manual_questions_review_and_variations(tmp_path):
    service, alpha, beta = _service_with_papers(tmp_path)
    generated = service.generate_questions(paper_id=alpha["document_id"], count=40)
    assert len(generated) == 40
    assert all(item["review_status"] == "proposed" for item in generated)
    manual = service.add_question(
        question="How do Alpha and Beta differ?",
        paper_ids=(alpha["document_id"], beta["document_id"]),
        question_type="comparison",
    )
    approved = service.review_question(
        question_id=manual["question_id"],
        review_status="human_approved",
        required_concepts=("optimizer",),
    )
    assert approved["required_concepts"] == ["optimizer"]
    variations = service.propose_question_variations(manual["question_id"])
    assert len(variations) == 4
    assert all(item["review_status"] == "proposed" for item in variations)
    assert service.list_questions()[0]["question_type"] in {
        "false_premise",
        "insufficient_evidence",
        "derivation",
        "critical_reasoning",
    }


def test_review_correction_approval_and_dataset_export(tmp_path):
    service, alpha, _beta = _service_with_papers(tmp_path)
    interaction = service.ask(
        question="Which optimizer and learning rate are used?",
        document_id=alpha["document_id"],
    )
    evidence_id = interaction["answer"]["evidence"][0]["evidence_id"]
    candidate = service.review_interaction(
        interaction_id=interaction["interaction_id"],
        review_label="partial",
        corrected_answer="Alpha uses Adam with learning rate 0.001. [C1]",
        required_facts=("Alpha uses Adam.",),
        prohibited_claims=("Alpha uses SGD.",),
        replacement_evidence_ids=(evidence_id,),
        notes="The learning rate was omitted.",
    )
    assert candidate["review_status"] == "proposed"
    with pytest.raises(ValueError, match="No human-approved"):
        ReviewService(tmp_path).export_training_dataset()

    approved = service.approve_correction(
        example_id=candidate["example_id"], reviewer="reviewer-1"
    )
    assert approved["review_status"] == "human_approved"
    exported = service.export_training_dataset(
        manual_paper_splits={alpha["document_id"]: "test"}
    )
    dataset = load_dataset(exported["output"])
    assert len(dataset.examples) == 1
    assert dataset.examples[0].review_status == "human_approved"
    assert dataset.examples[0].split == "test"


def test_invalid_paper_scopes_and_review_evidence_fail_clearly(tmp_path):
    service, alpha, _beta = _service_with_papers(tmp_path)
    with pytest.raises(ValueError, match="Use document_id or document_ids"):
        service.ask(
            question="Question",
            document_id=alpha["document_id"],
            document_ids=(alpha["document_id"],),
        )
    interaction = service.ask(
        question="What optimizer?", document_id=alpha["document_id"]
    )
    with pytest.raises(ValueError, match="replacement evidence"):
        service.review_interaction(
            interaction_id=interaction["interaction_id"],
            review_label="incorrect",
            corrected_answer="Corrected.",
            replacement_evidence_ids=("missing",),
        )


def test_correction_suggestions_can_be_edited_or_rejected_without_approval(tmp_path):
    service, alpha, _beta = _service_with_papers(tmp_path)
    interaction = service.ask(
        question="Which optimizer and learning rate are used?",
        document_id=alpha["document_id"],
    )
    first = service.review_interaction(
        interaction_id=interaction["interaction_id"],
        review_label="partial",
        corrected_answer="Alpha uses Adam. [C1]",
    )
    edited = service.edit_correction(
        example_id=first["example_id"],
        final_answer="Alpha uses Adam with learning rate 0.001. [C1]",
    )
    assert edited["review_status"] == "proposed"
    assert "0.001" in edited["final_answer"]
    rejected = service.reject_correction(
        example_id=edited["example_id"],
        reviewer="reviewer",
        reason="Evidence did not support the wording.",
    )
    assert rejected["review_status"] == "human_rejected"
    assert rejected["metadata"]["rejected_by"] == "reviewer"


def test_automatic_batch_runs_questions_and_waits_for_user_confirmation(tmp_path):
    service, alpha, _beta = _service_with_papers(tmp_path)
    question = service.add_question(
        question="Explain how Alpha computes attention.",
        paper_ids=(alpha["document_id"],),
        question_type="teaching",
    )
    batch = service.run_automatic_review_batch(
        paper_ids=(alpha["document_id"],),
        question_ids=(question["question_id"],),
        generate_if_empty=False,
    )
    assert batch["status"] == "awaiting_user_review"
    assert batch["semantic_judge_used"] is False
    assert batch["human_confirmation_required"] is True
    assert batch["summary"]["pending_user_review_count"] == 1
    assert batch["reviews"][0]["proposed_label"] == "partial"
    assert service.state()["automatic_review_batch_count"] == 1
    assert service.state()["correction_count"] == 0


def test_automatic_batch_accepts_edits_as_proposed_corrections(tmp_path):
    service, alpha, _beta = _service_with_papers(tmp_path)
    question = service.add_question(
        question="Which optimizer and learning rate are used?",
        paper_ids=(alpha["document_id"],),
        question_type="fact_extraction",
    )
    batch = service.run_automatic_review_batch(
        paper_ids=(alpha["document_id"],),
        question_ids=(question["question_id"],),
    )
    review = batch["reviews"][0]
    edited_answer = "Alpha uses Adam with learning rate 0.001. [C1]"
    result = service.finalize_automatic_review_batch(
        batch_id=batch["batch_id"],
        reviewer="reviewer-1",
        decisions=[
            {
                "review_id": review["review_id"],
                "accepted": True,
                "review_label": "correct",
                "corrected_answer": edited_answer,
                "required_facts": ["Alpha uses Adam at learning rate 0.001."],
                "prohibited_claims": ["Alpha uses SGD."],
                "evidence_ids": review["proposed_evidence_ids"],
            }
        ],
    )
    saved_review = result["batch"]["reviews"][0]
    assert result["batch"]["status"] == "saved"
    assert result["saved_review_count"] == 1
    assert result["corrections_remain_proposed"] is True
    assert saved_review["decision"] == "saved_as_user_review"
    assert saved_review["user_edited"] is True
    assert service.state()["corrections"][0]["review_status"] == "proposed"
    assert service.state()["approved_example_count"] == 0


def test_automatic_batch_can_be_excluded_and_rejects_malformed_decisions(tmp_path):
    service, alpha, _beta = _service_with_papers(tmp_path)
    question = service.add_question(
        question="What limitation is reported?",
        paper_ids=(alpha["document_id"],),
        question_type="limitation",
    )
    batch = service.run_automatic_review_batch(
        paper_ids=(alpha["document_id"],),
        question_ids=(question["question_id"],),
    )
    review_id = batch["reviews"][0]["review_id"]
    with pytest.raises(ValueError, match="review_label"):
        service.finalize_automatic_review_batch(
            batch_id=batch["batch_id"],
            reviewer="reviewer-1",
            decisions=[
                {
                    "review_id": review_id,
                    "accepted": True,
                    "review_label": "looks_good",
                }
            ],
        )
    assert service.state()["correction_count"] == 0

    result = service.finalize_automatic_review_batch(
        batch_id=batch["batch_id"],
        reviewer="reviewer-1",
        decisions=[{"review_id": review_id, "accepted": False}],
    )
    assert result["excluded_count"] == 1
    assert result["batch"]["reviews"][0]["decision"] == "excluded_by_user"
    assert service.state()["correction_count"] == 0


def test_automatic_batch_preserves_one_question_failure_and_continues(
    tmp_path, monkeypatch
):
    service, alpha, _beta = _service_with_papers(tmp_path)
    first = service.add_question(
        question="What does Alpha compute?",
        paper_ids=(alpha["document_id"],),
        question_type="method",
    )
    second = service.add_question(
        question="Which optimizer is used?",
        paper_ids=(alpha["document_id"],),
        question_type="reproduction",
    )
    original = service.run_question

    def fail_one(question_id, *, session_id=None):
        if question_id == first["question_id"]:
            raise RuntimeError("synthetic answer failure")
        return original(question_id, session_id=session_id)

    monkeypatch.setattr(service, "run_question", fail_one)
    batch = service.run_automatic_review_batch(
        paper_ids=(alpha["document_id"],),
        question_ids=(first["question_id"], second["question_id"]),
    )

    assert batch["status"] == "awaiting_user_review"
    assert batch["summary"]["review_count"] == 2
    assert batch["summary"]["execution_error_count"] == 1
    failed = next(item for item in batch["reviews"] if item.get("saveable") is False)
    assert failed["question_id"] == first["question_id"]
    assert "synthetic answer failure" in failed["rationale"][-1]
    with pytest.raises(ValueError, match="failed answer attempt"):
        service.finalize_automatic_review_batch(
            batch_id=batch["batch_id"],
            reviewer="reviewer-1",
            decisions=[{"review_id": failed["review_id"], "accepted": True}],
        )


def test_automatic_batch_can_resume_legacy_failed_state(tmp_path):
    service, alpha, _beta = _service_with_papers(tmp_path)
    question = service.add_question(
        question="Which optimizer is used?",
        paper_ids=(alpha["document_id"],),
        question_type="reproduction",
    )
    legacy_batch = {
        "batch_id": "auto_batch_legacy",
        "created_at": "2026-08-01T00:00:00+00:00",
        "completed_at": None,
        "paper_ids": [alpha["document_id"]],
        "question_ids": [question["question_id"]],
        "status": "failed",
        "error": "old all-or-nothing failure",
        "reviews": [],
        "summary": {},
    }
    atomic_write_json(service.automatic_reviews_path, [legacy_batch])

    resumed = service.resume_automatic_review_batch("auto_batch_legacy")

    assert resumed["status"] == "awaiting_user_review"
    assert "error" not in resumed
    assert resumed["summary"]["review_count"] == 1
    assert resumed["reviews"][0]["question_id"] == question["question_id"]
