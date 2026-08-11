"""Focused tests for Milestone 12A.4 reliability hardening."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from localml_scholar.training_data import (
    CitationDecision,
    DisagreementSeverity,
    EvidenceDecision,
    autonomous_training_exclusion,
    claim_level_disagreements,
    classify_reviewer_disagreements,
    deterministic_adjudication,
    full_run_readiness,
    migrate_legacy_record,
    normalize_citations,
    reliability_report,
    repair_diagnostics,
    segment_answer_claims,
    select_diagnostic_candidates,
    should_stop_for_reliability,
    stable_evidence_identity,
    stamp_evidence_identities,
    validate_claim_citations,
)


def _evidence(
    label: str = "C1",
    *,
    document_id: str = "paper-a",
    chunk_id: str = "chunk-a",
    text: str = "The model uses Adam with learning rate 0.001.",
    heading: str = "Method",
) -> dict[str, Any]:
    return {
        "label": label,
        "evidence_id": chunk_id,
        "chunk_id": chunk_id,
        "document_id": document_id,
        "text": text,
        "heading_path": [heading],
        "page_start": 2,
        "page_end": 2,
        "start_line": 10,
        "end_line": 12,
    }


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Adam is used [C1].", "Adam is used [C1]."),
        ("Adam is used C1.", "Adam is used [C1]."),
        ("Adam [C1] [C1].", "Adam [C1]."),
        ("Adam [chunk-a].", "Adam [C1]."),
    ],
)
def test_citation_normalization_supported_forms(source: str, expected: str) -> None:
    normalized = normalize_citations(source, [_evidence()])
    assert normalized.text == expected
    assert normalized.labels == ("C1",)
    assert normalized.valid


def test_citation_normalization_multiple_reordered_labels() -> None:
    evidence = [_evidence("C1"), _evidence("C2", chunk_id="chunk-b")]
    result = normalize_citations("Result [C2] [C1] [C2].", evidence)
    assert result.text == "Result [C2] [C1]."
    assert result.labels == ("C2", "C1")


def test_citation_normalization_rejects_unknown_and_malformed() -> None:
    unknown = normalize_citations("Unsupported [C9].", [_evidence()])
    malformed = normalize_citations("Unsupported [Cfoo].", [_evidence()])
    assert unknown.unknown_labels == ("C9",)
    assert not unknown.valid
    assert malformed.unknown_labels == ("Cfoo",)
    assert malformed.malformed_markers == ("Cfoo",)


def test_claim_segmentation_classifies_multiple_claims() -> None:
    claims = segment_answer_claims(
        "Adam is used [C1]. The learning rate is 0.001 [C1]. "
        "However, this is only the initial rate [C1].",
        [_evidence()],
    )
    assert [item.claim_type for item in claims] == [
        "factual",
        "numeric",
        "qualification",
    ]
    assert all(item.citation_labels == ("C1",) for item in claims)
    assert len({item.claim_id for item in claims}) == 3


def test_claim_segmentation_detects_equation() -> None:
    claims = segment_answer_claims("The loss is L = -log p [C1].", [_evidence()])
    assert claims[0].claim_type == "equation"


def test_stable_evidence_identity_ignores_display_label() -> None:
    left = _evidence("C1")
    right = _evidence("C7")
    assert stable_evidence_identity(left) == stable_evidence_identity(right)
    stamped = stamp_evidence_identities([left])
    assert stamped[0]["stable_evidence_id"].startswith("evidence_")
    assert "stable_evidence_id" not in left


def test_stable_evidence_identity_requires_source_coordinates() -> None:
    with pytest.raises(ValueError, match="document_id"):
        stable_evidence_identity({"chunk_id": "chunk-a"})
    with pytest.raises(ValueError, match="chunk_id"):
        stable_evidence_identity({"document_id": "paper-a"})


def test_claim_citation_validator_accepts_exact_support() -> None:
    result = validate_claim_citations(
        "The model uses Adam with learning rate 0.001 [C1].",
        [_evidence()],
        selected_paper_ids=("paper-a",),
        expected_sections=("Method",),
        required_concepts=("Adam",),
    )
    assert result["structural_valid"]
    assert result["support_valid"]
    assert result["relevance_valid"]
    assert result["claim_critiques"][0]["support_status"] == "supports"


@pytest.mark.parametrize(
    ("answer", "evidence", "expected"),
    [
        ("The rate is 0.01 [C1].", _evidence(), CitationDecision.PARTIAL.value),
        ("Adam is used.", _evidence(), CitationDecision.MISSING.value),
        (
            "Adam is used [C1].",
            _evidence(text="This passage discusses unrelated data."),
            CitationDecision.SUPPORTS.value,
        ),
    ],
)
def test_claim_citation_validator_failure_modes(
    answer: str, evidence: dict[str, Any], expected: str
) -> None:
    result = validate_claim_citations(answer, [evidence])
    assert result["claim_critiques"][0]["support_status"] == expected


def test_claim_citation_validator_rejects_wrong_source_and_section() -> None:
    result = validate_claim_citations(
        "Adam is used [C1].",
        [_evidence(document_id="paper-b", heading="Results")],
        selected_paper_ids=("paper-a",),
        expected_sections=("Method",),
    )
    assert not result["structural_valid"]
    critique = result["claim_critiques"][0]
    assert critique["support_status"] == "wrong_source"
    assert "section_mismatch" in critique["missing_information"]


def test_claim_citation_validator_uses_canonical_small_caps_sections() -> None:
    result = validate_claim_citations(
        "The minibatch size was 128 [C1].",
        [
            _evidence(
                text="The minibatch size was 128.",
                heading="E XPERIMENTS",
            )
        ],
        selected_paper_ids=("paper-a",),
        expected_sections=("experiments",),
    )
    assert result["structural_valid"]
    assert "section_mismatch" not in result["claim_critiques"][0]["missing_information"]


def _pass(pass_name: str, decision: str, corrections: tuple[str, ...] = ()) -> dict:
    return {
        "pass_name": pass_name,
        "result": {
            "decision": decision,
            "required_corrections": list(corrections),
        },
    }


def test_disagreement_taxonomy_distinguishes_hard_and_soft() -> None:
    hard = classify_reviewer_disagreements(
        [
            _pass("evidence_critic", "reject"),
            _pass("answer_critic", "accept"),
            _pass("citation_critic", "accept"),
            _pass("final_adjudicator", "accept"),
        ]
    )
    assert any(item.severity == DisagreementSeverity.HARD for item in hard)
    assert (
        any(item.category == "final_adjudicator_overrides_majority" for item in hard)
        is False
    )

    soft = classify_reviewer_disagreements(
        [
            _pass("evidence_critic", "accept", ("minor wording style",)),
            _pass("answer_critic", "accept", ("more concise style",)),
            _pass("citation_critic", "accept"),
        ]
    )
    assert any(item.category == "answer_style_only" for item in soft)
    assert not any(item.severity == DisagreementSeverity.HARD for item in soft)


def test_disagreement_after_repair_and_validator_conflict() -> None:
    disagreements = classify_reviewer_disagreements(
        [_pass("citation_critic", "accept")],
        after_repair=True,
        validation={"structural_valid": False},
    )
    categories = {item.category for item in disagreements}
    assert "deterministic_validator_fail_citation_critic_accept" in categories
    assert "disagreement_after_repair" in categories


def test_claim_level_disagreement_and_semantic_wording_alignment() -> None:
    passes = [
        {
            "pass_name": "answer_critic",
            "result": {
                "decision": "accept",
                "policy_outcome": "supported",
                "claim_critiques": [
                    {"claim_id": "claim-1", "support_status": "supports"}
                ],
            },
        },
        {
            "pass_name": "citation_critic",
            "result": {
                "decision": "repair",
                "policy_outcome": "unsupported",
                "claim_critiques": [
                    {"claim_id": "claim-1", "support_status": "wrong_span"}
                ],
            },
        },
    ]
    assert claim_level_disagreements(passes)[0]["claim_id"] == "claim-1"
    assert any(
        item.severity == DisagreementSeverity.HARD
        for item in classify_reviewer_disagreements(passes)
    )

    aligned = [
        {
            "pass_name": "answer_critic",
            "result": {"decision": "accept", "policy_outcome": "supported"},
        },
        {
            "pass_name": "citation_critic",
            "result": {"decision": "repair", "policy_outcome": "supports"},
        },
    ]
    assert not any(
        item.severity == DisagreementSeverity.HARD
        for item in classify_reviewer_disagreements(aligned)
    )


def test_deterministic_adjudication_conflict_policy() -> None:
    failure = deterministic_adjudication(
        citation_validation={"structural_valid": False, "support_valid": False},
        evidence_decision=EvidenceDecision.INSUFFICIENT,
    )
    assert failure["decision"] == "abstain_or_reject"
    assert "citation_structural_failure" in failure["triggers"]
    ambiguous = deterministic_adjudication(
        citation_validation={"structural_valid": True, "support_valid": True},
        ambiguous_question=True,
    )
    assert ambiguous["decision"] == "exclude"


def test_repair_diagnostics_fixed_introduced_and_unchanged() -> None:
    failed = {
        "structural_valid": False,
        "support_valid": False,
        "relevance_valid": False,
    }
    passed = {"structural_valid": True, "support_valid": True, "relevance_valid": True}
    assert repair_diagnostics(failed, passed)["outcome"] == "fixed"
    assert repair_diagnostics(passed, failed)["outcome"] == "introduced_new_issue"
    assert repair_diagnostics(failed, failed)["outcome"] == "unchanged"


def test_unstable_categories_are_retained_but_excluded_from_training() -> None:
    assert autonomous_training_exclusion("derivation")
    assert autonomous_training_exclusion("equation")
    assert autonomous_training_exclusion("method") is None


def _safety_record(**overrides: Any) -> dict[str, Any]:
    value = {
        "codex_review_passes": [_pass("citation_critic", "accept")],
        "hard_reviewer_disagreement": False,
        "soft_reviewer_disagreement": False,
        "citation_structural_valid": True,
        "citation_support_valid": True,
        "reviewer_output_malformed": False,
        "source_hash_mismatches": [],
    }
    return {**value, **overrides}


def test_safety_stop_uses_hard_not_soft_disagreement() -> None:
    soft = [_safety_record(soft_reviewer_disagreement=True) for _ in range(10)]
    assert should_stop_for_reliability(soft) is None
    hard = soft[:-2] + [
        _safety_record(hard_reviewer_disagreement=True),
        _safety_record(hard_reviewer_disagreement=True),
    ]
    assert "Hard reviewer disagreement" in should_stop_for_reliability(hard)


def test_safety_stop_citation_and_leakage_gates() -> None:
    citation = [_safety_record() for _ in range(9)] + [
        _safety_record(citation_structural_valid=False)
    ]
    assert "Citation structural" in should_stop_for_reliability(citation)
    assert (
        "leakage"
        in should_stop_for_reliability([_safety_record(leakage_detected=True)]).lower()
    )


def test_diagnostic_sampling_is_deterministic_and_stratified() -> None:
    candidates = [
        {
            "question_id": f"q-{index}",
            "paper_ids": [f"paper-{index % 3}"],
            "question_type": ("method", "experiment", "limitation")[index % 3],
            "original_failure": ("none", "citation")[index % 2],
        }
        for index in range(30)
    ]
    first = select_diagnostic_candidates(candidates, count=12, seed=42)
    second = select_diagnostic_candidates(candidates, count=12, seed=42)
    assert first == second
    assert len({item["paper_ids"][0] for item in first}) == 3
    assert len({item["question_type"] for item in first}) == 3


def test_diagnostic_sampling_balances_papers_before_repeating() -> None:
    candidates = [
        {
            "question_id": f"paper-{paper}-question-{question}",
            "paper_ids": [f"paper-{paper}"],
            "question_type": f"type-{question % 5}",
            "original_failure": "none",
        }
        for paper in range(14)
        for question in range(20)
    ]
    selected = select_diagnostic_candidates(candidates, count=50, seed=42)
    counts = Counter(item["paper_ids"][0] for item in selected)
    assert len(counts) == 14
    assert max(counts.values()) - min(counts.values()) <= 1


def test_diagnostic_sampling_honors_optional_per_paper_cap() -> None:
    candidates = [
        {
            "question_id": f"paper-{paper}-question-{question}",
            "paper_ids": [f"paper-{paper}"],
            "question_type": f"type-{question % 3}",
        }
        for paper in range(4)
        for question in range(5)
    ]
    selected = select_diagnostic_candidates(
        candidates, count=8, seed=42, maximum_per_paper=1
    )
    assert len(selected) == 4
    assert len({item["paper_ids"][0] for item in selected}) == 4


def test_reliability_report_and_readiness() -> None:
    records = [
        _safety_record(
            curation_record_id=f"r-{index}",
            question=f"Question {index}",
            question_type="method",
            citation_relevance_valid=True,
            source_hashes=["hash"],
            stale_evidence_ids=[],
            reviewer_disagreements=[],
            repair_attempts=1,
            repair_history=[{"outcome": "fixed"}],
            claim_alignment_metrics={
                "claim_count": 1,
                "supported_claim_count": 1,
                "uncited_claim_count": 0,
                "sentence_to_claim_traceability": 1.0,
            },
        )
        for index in range(50)
    ]
    report = reliability_report(records)
    assert report["citation_structural_validity"] == 1.0
    assert report["repair_success_rate"] == 1.0
    assert full_run_readiness(report)["ready"]


def test_legacy_record_migrates_without_mutation() -> None:
    legacy = {
        "curation_record_id": "legacy-1",
        "paper_ids": ["paper-a"],
        "question": "What optimizer?",
        "question_type": "method",
        "answer": {
            "answer_text": "Adam is used [C1].",
            "evidence": [_evidence()],
        },
        "codex_review_passes": [
            _pass("evidence_critic", "accept"),
            _pass("answer_critic", "accept"),
            _pass("citation_critic", "accept"),
            _pass("final_adjudicator", "accept"),
        ],
        "repair_attempts": 0,
        "source_hashes": ["hash"],
    }
    migrated = migrate_legacy_record(legacy)
    assert "review_policy_version" not in legacy
    assert migrated["legacy_artifact_migrated_in_memory"]
    assert migrated["citation_structural_valid"]
