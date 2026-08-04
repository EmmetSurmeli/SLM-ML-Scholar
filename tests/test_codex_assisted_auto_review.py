from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from localml_scholar.review_app.service import ReviewService
from localml_scholar.training_data import (
    AutoReviewPolicy,
    CalibrationPolicy,
    ConversationTurn,
    GroundedFact,
    GroundedInstructionExample,
    InstructionProfile,
    QuestionCandidate,
    ReviewProvenance,
    StructuredGroundedTarget,
    build_dataset,
    calibration_report,
    cluster_duplicates,
    content_sha256,
    recommend_threshold,
    review_interaction_second_pass,
    select_audit_sample,
    select_trusted_examples,
)


def _candidate(**changes):
    values = {
        "paper_ids": ("paper-a",),
        "question": "Which optimizer is used?",
        "question_type": "experiment",
        "required_concepts": ("Adam optimizer",),
    }
    values.update(changes)
    return QuestionCandidate.create(**values)


def _interaction(answer_text: str = "The paper uses the Adam optimizer. [C1]"):
    return {
        "paper_ids": ["paper-a"],
        "question": "Which optimizer is used?",
        "answer": {
            "answer_text": answer_text,
            "abstained": False,
            "evidence": [
                {
                    "label": "C1",
                    "evidence_id": "chunk-a",
                    "selected_text": "We use the Adam optimizer.",
                }
            ],
            "validation": {
                "accepted": True,
                "citations_valid": True,
                "citation_coverage": 1.0,
                "unsupported_claim_count": 0,
            },
            "sufficiency": {"query_term_coverage": 1.0},
        },
        "diagnostics": {"accepted": True, "failure_categories": []},
    }


def _example(
    example_id: str,
    *,
    status: str = "human_approved",
    question: str = "Which optimizer is used?",
    answer: str = "The paper uses Adam. [C1]",
    metadata: dict | None = None,
):
    return GroundedInstructionExample(
        example_id=example_id,
        paper_ids=("paper-a",),
        turns=(ConversationTurn("user", question),),
        instruction_profile=InstructionProfile(),
        target=StructuredGroundedTarget(
            facts=(GroundedFact("The paper uses Adam.", "paper_explicit", ("C1",)),)
        ),
        final_answer=answer,
        evidence=(
            {
                "label": "C1",
                "evidence_id": "chunk-a",
                "selected_text": "We use Adam.",
            },
        ),
        task_type="paper_question_answering",
        review_status=status,
        review_label="correct",
        metadata={} if metadata is None else metadata,
    )


def test_default_calibration_lock_blocks_automatic_approval():
    result = review_interaction_second_pass(_interaction(), _candidate())
    assert result.review_status == "needs_human_review"
    assert result.confidence >= 0.95
    assert all(item.passed for item in result.reviewer_results)
    assert result.calibration_state == "calibration_required"


def test_enabled_calibration_can_approve_only_when_all_gates_pass():
    result = review_interaction_second_pass(
        _interaction(),
        _candidate(),
        policy=AutoReviewPolicy(calibration_state="auto_approval_enabled"),
    )
    assert result.review_status == "codex_approved"
    assert len(result.reviewer_results) == 3
    assert result.to_dict()["reviewers_are_independent"] is False


@pytest.mark.parametrize(
    ("threshold", "expected"),
    (
        (0.959, "codex_approved"),
        (0.960, "codex_approved"),
        (0.961, "needs_human_review"),
    ),
)
def test_confidence_threshold_below_equal_and_above(threshold, expected):
    result = review_interaction_second_pass(
        _interaction(),
        _candidate(),
        policy=AutoReviewPolicy(
            approval_threshold=threshold,
            calibration_state="auto_approval_enabled",
        ),
    )
    assert result.review_status == expected


def test_high_risk_and_failed_citation_routes_to_human():
    candidate = _candidate(question="Was this the first influential optimizer paper?")
    result = review_interaction_second_pass(
        _interaction("The paper uses Adam without a citation."),
        candidate,
        policy=AutoReviewPolicy(calibration_state="auto_approval_enabled"),
    )
    assert result.review_status == "needs_human_review"
    assert {"first_to_claim", "impact_claim"} <= set(result.mandatory_human_categories)
    assert not result.reviewer_results[0].gates["citations_present"]


@pytest.mark.parametrize(
    "failure",
    (
        "irrelevant_evidence",
        "unsupported_claim",
        "invalid_citation",
        "wrong_answer",
        "incomplete_answer",
        "insufficient_evidence",
        "prohibited_claim",
        "external_context",
        "derivation",
        "conflicting_evidence",
    ),
)
def test_required_failure_cases_never_auto_approve(failure):
    interaction = deepcopy(_interaction())
    candidate = _candidate()
    if failure in {"irrelevant_evidence", "insufficient_evidence"}:
        interaction["answer"]["sufficiency"]["query_term_coverage"] = 0.2
    elif failure == "unsupported_claim":
        interaction["answer"]["validation"].update(
            accepted=False, unsupported_claim_count=1
        )
    elif failure == "invalid_citation":
        interaction["answer"]["answer_text"] = "The paper uses Adam. [C2]"
    elif failure in {"wrong_answer", "incomplete_answer"}:
        interaction["answer"]["answer_text"] = "The paper uses SGD. [C1]"
    elif failure == "prohibited_claim":
        candidate = _candidate(prohibited_claims=("uses Adam",))
    elif failure == "external_context":
        candidate = _candidate(question_type="external_context")
    elif failure == "derivation":
        candidate = _candidate(question_type="derivation")
    elif failure == "conflicting_evidence":
        interaction["diagnostics"]["failure_categories"] = ["source_conflict"]
    result = review_interaction_second_pass(
        interaction,
        candidate,
        policy=AutoReviewPolicy(calibration_state="auto_approval_enabled"),
    )
    assert result.review_status != "codex_approved"


def test_corrected_answer_is_revalidated_from_scratch():
    result = review_interaction_second_pass(
        _interaction(),
        _candidate(),
        policy=AutoReviewPolicy(calibration_state="auto_approval_enabled"),
        corrected_answer="Adam is used, but this edit has no citation.",
    )
    assert result.correction_revalidated
    assert result.review_status == "needs_human_review"


def test_provenance_is_stable_and_rejects_duplicate_ancestry():
    assert content_sha256({"b": 2, "a": 1}) == content_sha256({"a": 1, "b": 2})
    with pytest.raises(ValueError, match="duplicates"):
        ReviewProvenance(
            producer_system="producer",
            producer_version="1",
            reviewer_system="reviewer",
            reviewer_version="1",
            correction_system=None,
            source_hashes=("source",),
            answer_hash="answer",
            parent_example_ids=("parent", "parent"),
        )


def test_same_producer_review_warns_against_circular_approval():
    provenance = ReviewProvenance(
        producer_system="same",
        producer_version="1",
        reviewer_system="same",
        reviewer_version="1",
        correction_system=None,
        source_hashes=("source",),
        answer_hash="answer",
    )
    assert provenance.circular_warnings == (
        "same_producer_and_reviewer_without_independent_validation",
    )
    with pytest.raises(ValueError, match="source hashes"):
        provenance.validate_source_hashes(("different",))


def test_provenance_round_trip_preserves_correction_ancestry():
    provenance = ReviewProvenance(
        producer_system="producer",
        producer_version="1",
        reviewer_system="reviewer",
        reviewer_version="2",
        correction_system="corrector",
        source_hashes=("source-a", "source-b"),
        answer_hash="answer",
        parent_example_ids=("parent-a", "parent-b"),
        independent_validators=("human-a",),
    )
    loaded = ReviewProvenance.from_dict(provenance.to_dict())
    assert loaded == provenance


def test_calibration_requires_50_examples_and_explicit_enable():
    records = [
        {
            "confidence": 0.99,
            "automated_approved": True,
            "human_approved": True,
        }
        for _ in range(50)
    ]
    assert calibration_report(records[:49])["state"] == "calibration_required"
    assert calibration_report(records)["state"] == "calibration_active"
    enabled = calibration_report(records, explicit_enable=True)
    assert enabled["state"] == "auto_approval_enabled"
    assert enabled["agreement"] == 1.0


def test_calibration_suspends_on_human_overrides_and_recommends_threshold():
    records = [
        {
            "confidence": 0.96,
            "automated_approved": True,
            "human_approved": position >= 10,
        }
        for position in range(50)
    ]
    report = calibration_report(
        records,
        policy=CalibrationPolicy(maximum_brier_score=1.0),
        explicit_enable=True,
    )
    assert report["state"] == "auto_approval_suspended"
    assert report["override_rate"] == 0.2
    assert recommend_threshold(records) is None


def test_service_requires_explicit_enable_after_qualifying_calibration(tmp_path):
    service = ReviewService(tmp_path)
    records = [
        {
            "review_id": f"r-{index}",
            "confidence": 0.99,
            "automated_approved": True,
            "human_approved": True,
        }
        for index in range(50)
    ]
    service.review_policy_path.parent.mkdir(parents=True)
    service.review_policy_path.write_text(
        json.dumps(
            {
                "approval_threshold": 0.95,
                "explicit_enable": False,
                "calibration_records": records,
            }
        ),
        encoding="utf-8",
    )
    assert service.state()["calibration"]["state"] == "calibration_active"
    enabled = service.set_auto_approval_enabled(enabled=True)
    assert enabled["state"] == "auto_approval_enabled"


def test_audit_sampling_is_deterministic_and_includes_risk_cases():
    reviews = [{"review_id": f"r-{index}", "confidence": 0.7} for index in range(20)]
    reviews[3]["confidence"] = 0.95
    reviews[7]["novel_failure"] = True
    first = select_audit_sample(reviews)
    second = select_audit_sample(reviews)
    assert first == second
    selected = {item["example_id"] for item in first["items"]}
    assert {"r-3", "r-7"} <= selected


def test_audit_sampling_supports_zero_and_full_rates():
    reviews = [{"review_id": f"safe-{index}", "confidence": 0.5} for index in range(5)]
    assert select_audit_sample(reviews, sample_fraction=0.0)["selected_count"] == 0
    assert select_audit_sample(reviews, sample_fraction=1.0)["selected_count"] == 5


def test_duplicate_clustering_is_stable_and_punctuation_insensitive():
    first = _example("e1")
    second = _example(
        "e2",
        question="Which optimizer is used!!!",
        answer="The paper uses Adam [C1]",
    )
    result = cluster_duplicates((second, first))
    assert result["duplicate_cluster_count"] == 1
    assert (
        result["cluster_by_example_id"]["e1"] == result["cluster_by_example_id"]["e2"]
    )


def test_trust_tiers_apply_weights_and_deduplicate():
    human = _example("human")
    codex = _example(
        "codex",
        status="codex_approved",
        question="What optimizer is used?",
        metadata={"audit_status": "human_confirmed"},
    )
    human_only = select_trusted_examples((codex, human))
    assert [item.example_id for item in human_only] == ["human"]
    selected = select_trusted_examples(
        (codex, human), trust_tier="human-and-audited", deduplicate=False
    )
    weights = {item.example_id: item.metadata["trust_weight"] for item in selected}
    assert weights == {"codex": 0.9, "human": 1.0}


def test_circular_codex_approval_is_excluded_from_trust_exports():
    codex = _example(
        "codex",
        status="codex_approved",
        metadata={
            "review_provenance": {
                "circular_warnings": [
                    "same_producer_and_reviewer_without_independent_validation"
                ]
            }
        },
    )
    assert select_trusted_examples((codex,), trust_tier="include-codex") == ()


def test_pending_rejected_and_benchmark_records_are_excluded():
    statuses = ("proposed", "human_rejected", "codex_rejected", "benchmark_problem")
    examples = tuple(
        replace(_example(f"e-{status}"), review_status=status) for status in statuses
    )
    assert select_trusted_examples(examples, trust_tier="include-codex-approved") == ()


def test_dataset_default_remains_human_only_and_trust_export_is_explicit():
    human = _example("human")
    codex = replace(
        _example("codex"),
        review_status="codex_approved",
        metadata={"audit_status": "human_confirmed"},
    )
    default = build_dataset((human, codex))
    assert [item.example_id for item in default.examples] == ["human"]
    audited = build_dataset(
        (human, codex), trust_tier="human-and-audited", deduplicate=False
    )
    assert {item.example_id for item in audited.examples} == {"human", "codex"}
    assert audited.metadata["trust_tier"] == "human-and-audited"


def test_designated_test_only_paper_is_rejected_from_training_export():
    with pytest.raises(ValueError, match="test-only"):
        build_dataset(
            (replace(_example("human"), metadata={"test_only": True}),),
        )
