"""Focused safety and integration tests for Milestone 12A.3."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from localml_scholar.review_app.autonomous_curation import AutonomousCorpusCurator
from localml_scholar.review_app.service import ReviewService
from localml_scholar.training_data import (
    AutonomousCurationConfig,
    CodexReview,
    QuestionCandidate,
    autonomous_quality_report,
    balanced_paper_splits,
    blind_payload,
    build_dataset,
    curate_interaction,
    propose_correction,
)


def _review(
    decision: str = "accept",
    *,
    confidence: float = 0.99,
    evidence_relevance: float = 0.99,
    unsupported_claims: tuple[str, ...] = (),
    required_corrections: tuple[str, ...] = (),
    corrected_answer: str | None = None,
) -> CodexReview:
    return CodexReview(
        decision=decision,
        confidence=confidence,
        answer_correctness=0.99,
        evidence_relevance=evidence_relevance,
        factual_support=0.99,
        completeness=0.99,
        citation_support=0.99,
        citation_relevance=0.99,
        instruction_following=0.99,
        unsupported_claims=unsupported_claims,
        required_corrections=required_corrections,
        corrected_answer=corrected_answer,
        rationale="Focused test judgment.",
    )


class AcceptingProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def identity(self) -> tuple[str, str]:
        return "test_codex", "1"

    def available(self) -> bool:
        return True

    def review(self, pass_name: str, payload: dict[str, Any]) -> CodexReview:
        self.calls.append((pass_name, payload))
        return _review(
            corrected_answer=(
                "The supplied evidence supports this answer. [C1]"
                if pass_name == "answerer"
                else None
            )
        )


class RepairingProvider(AcceptingProvider):
    def __init__(self, *, never_succeeds: bool = False) -> None:
        super().__init__()
        self.cycle = 0
        self.never_succeeds = never_succeeds

    def review(self, pass_name: str, payload: dict[str, Any]) -> CodexReview:
        self.calls.append((pass_name, payload))
        repairing = self.never_succeeds or self.cycle == 0
        if pass_name == "answerer":
            return _review(corrected_answer="Repaired supported answer. [C1]")
        if pass_name == "evidence_critic" and repairing:
            return _review(
                "repair",
                evidence_relevance=0.4,
                required_corrections=("retrieve stronger evidence",),
            )
        if pass_name in {"answer_critic", "citation_critic"} and repairing:
            return _review("repair", required_corrections=("repair answer",))
        if pass_name == "final_adjudicator" and repairing:
            self.cycle += 1
            return _review("repair", required_corrections=("run full repair",))
        return _review()


def _fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidate = QuestionCandidate.create(
        paper_ids=("paper-a",),
        question="What method is used?",
        question_type="method",
        metadata={"source": "test"},
    ).to_dict()
    evidence = {
        "label": "C1",
        "evidence_id": "chunk-1",
        "chunk_id": "chunk-1",
        "document_id": "paper-a",
        "selected_text": "The method uses an optimizer.",
        "text": "The method uses an optimizer.",
        "score": 8.0,
        "retrieval_method": "bm25",
    }
    interaction = {
        "interaction_id": "interaction-1",
        "paper_ids": ["paper-a"],
        "question": candidate["question"],
        "instruction_profile": {"desired_depth": "standard"},
        "conversation_turns": [],
        "answer": {
            "answer_text": "The method uses an optimizer. [C1]",
            "evidence": [evidence],
            "abstained": False,
        },
    }
    deterministic = {
        "reviewer_results": [{"gates": {"all_required_checks": True}}],
        "mandatory_human_categories": [],
    }
    return interaction, candidate, deterministic


def test_reviewer_blindness_removes_forbidden_context() -> None:
    interaction, candidate, deterministic = _fixture()
    payload = {
        "question": candidate["question"],
        "answer": interaction["answer"],
        "source_passages": interaction["answer"]["evidence"],
        "structured_target": {},
        "deterministic_review": deterministic,
        "critic_results": [{"decision": "accept"}],
    }
    evidence_view = blind_payload("evidence_critic", payload)
    answer_view = blind_payload("answer_critic", payload)
    citation_view = blind_payload("citation_critic", payload)
    assert "answer" not in evidence_view
    assert "critic_results" not in evidence_view
    assert "score" not in answer_view["source_passages"][0]
    assert "retrieval_method" not in answer_view["source_passages"][0]
    assert "critic_results" not in citation_view


def test_successful_flow_has_complete_nonhuman_provenance() -> None:
    interaction, candidate, deterministic = _fixture()
    provider = AcceptingProvider()
    record = curate_interaction(
        interaction, candidate, deterministic, provider=provider
    )
    assert record["status"] == "codex_curated"
    assert record["trust_class"] == "codex_curated"
    assert len(record["codex_review_passes"]) == 5
    assert [item[0] for item in provider.calls] == [
        "answerer",
        "evidence_critic",
        "answer_critic",
        "citation_critic",
        "final_adjudicator",
    ]
    assert record["source_hashes"]
    assert record["record_hash"]
    assert record["final_adjudicator"] == "test_codex"


def test_missing_evidence_is_excluded_without_calling_codex() -> None:
    interaction, candidate, deterministic = _fixture()
    interaction["answer"]["evidence"] = []
    provider = AcceptingProvider()
    record = curate_interaction(
        interaction, candidate, deterministic, provider=provider
    )
    assert record["status"] == "insufficient_evidence"
    assert provider.calls == []


def test_repair_retrieves_revalidates_and_runs_all_passes_again() -> None:
    interaction, candidate, deterministic = _fixture()
    provider = RepairingProvider()
    retrievals: list[int] = []
    validations: list[str] = []

    def retrieve(_query: str, _paper_ids: tuple[str, ...], attempt: int):
        retrievals.append(attempt)
        return interaction["answer"]["evidence"]

    def revalidate(answer: dict[str, Any], _candidate: dict[str, Any]):
        validations.append(answer["answer_text"])
        return deterministic

    record = curate_interaction(
        interaction,
        candidate,
        deterministic,
        provider=provider,
        retrieve_evidence=retrieve,
        revalidate=revalidate,
    )
    assert record["status"] == "codex_curated"
    assert record["repair_attempts"] == 1
    assert retrievals == [1]
    assert len(validations) == 2
    assert len(record["codex_review_passes"]) == 10


def test_repair_limit_exhaustion_rejects() -> None:
    interaction, candidate, deterministic = _fixture()
    record = curate_interaction(
        interaction,
        candidate,
        deterministic,
        provider=RepairingProvider(never_succeeds=True),
        config=AutonomousCurationConfig(maximum_repair_attempts=1),
    )
    assert record["status"] == "rejected"
    assert record["repair_attempts"] == 1
    assert "repair_limit_exhausted" in record["terminal_reasons"]


def test_unsupported_claim_and_reviewer_disagreement_are_not_accepted() -> None:
    interaction, candidate, deterministic = _fixture()

    class UnsafeProvider(AcceptingProvider):
        def review(self, pass_name: str, payload: dict[str, Any]) -> CodexReview:
            if pass_name == "answer_critic":
                return _review("reject", unsupported_claims=("invented fact",))
            return _review()

    record = curate_interaction(
        interaction, candidate, deterministic, provider=UnsafeProvider()
    )
    assert record["status"] == "rejected"
    assert record["reviewer_disagreement"] is True


@pytest.mark.parametrize(
    "failing_pass", ["evidence_critic", "answer_critic", "citation_critic"]
)
def test_each_focused_critic_can_block_a_fluent_answer(failing_pass: str) -> None:
    interaction, candidate, deterministic = _fixture()

    class FocusedFailureProvider(AcceptingProvider):
        def review(self, pass_name: str, payload: dict[str, Any]) -> CodexReview:
            if pass_name == failing_pass:
                return _review(
                    "reject", unsupported_claims=(f"{failing_pass} failure",)
                )
            return _review()

    record = curate_interaction(
        interaction,
        candidate,
        deterministic,
        provider=FocusedFailureProvider(),
    )
    assert record["status"] == "rejected"


def test_derivation_requires_provenance_labelled_steps() -> None:
    interaction, candidate, deterministic = _fixture()
    candidate["question_type"] = "derivation"
    record = curate_interaction(
        interaction, candidate, deterministic, provider=AcceptingProvider()
    )
    assert record["status"] == "uncertain"
    assert "derivation_provenance_incomplete" in record["terminal_reasons"]


def test_cross_paper_examples_receive_extra_evidence_and_citation_passes() -> None:
    interaction, candidate, deterministic = _fixture()
    interaction["paper_ids"] = ["paper-a", "paper-b"]
    candidate["paper_ids"] = ["paper-a", "paper-b"]
    provider = AcceptingProvider()
    record = curate_interaction(
        interaction, candidate, deterministic, provider=provider
    )
    assert record["status"] == "codex_curated"
    assert [name for name, _payload in provider.calls].count("evidence_critic") == 2
    assert [name for name, _payload in provider.calls].count("citation_critic") == 2
    assert len(record["codex_review_passes"]) == 7


def test_malformed_provider_output_fails_closed() -> None:
    interaction, candidate, deterministic = _fixture()

    class MalformedProvider(AcceptingProvider):
        def review(self, pass_name: str, payload: dict[str, Any]):
            return {"decision": "accept"}

    with pytest.raises(TypeError, match="invalid result type"):
        curate_interaction(
            interaction, candidate, deterministic, provider=MalformedProvider()
        )


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, {"train": 1}),
        (2, {"train": 1, "test": 1}),
        (3, {"train": 1, "validation": 1, "test": 1}),
        (6, {"train": 4, "validation": 1, "test": 1}),
    ],
)
def test_small_corpus_splits_are_deterministic_and_disjoint(
    count: int, expected: dict[str, int]
) -> None:
    paper_ids = tuple(f"paper-{index}" for index in range(count))
    first = balanced_paper_splits(paper_ids, seed=42)
    second = balanced_paper_splits(tuple(reversed(paper_ids)), seed=42)
    assert first == second
    assert dict(
        sorted(
            {
                split: list(first.values()).count(split)
                for split in set(first.values())
            }.items()
        )
    ) == dict(sorted(expected.items()))


def test_codex_curated_export_never_becomes_human_gold() -> None:
    interaction, _, _ = _fixture()
    candidate = propose_correction(interaction, review_label="correct")
    curated = replace(
        candidate,
        review_status="codex_curated",
        metadata={**candidate.metadata, "trust_class": "codex_curated"},
    )
    dataset = build_dataset(
        (curated,), trust_tier="codex-curated-only", dataset_version="1.2.3"
    )
    assert dataset.examples[0].review_status == "codex_curated"
    assert dataset.metadata["trust_tier"] == "codex-curated-only"
    with pytest.raises(ValueError, match="No human-approved"):
        build_dataset((curated,), trust_tier="human-only")


def test_quality_report_uses_actual_terminal_counts() -> None:
    interaction, candidate, deterministic = _fixture()
    accepted = curate_interaction(
        interaction, candidate, deterministic, provider=AcceptingProvider()
    )
    rejected = {**accepted, "status": "rejected", "trust_class": None}
    report = autonomous_quality_report([accepted, rejected])
    assert report["examples_accepted"] == 1
    assert report["examples_rejected"] == 1
    assert report["acceptance_rate"] == 0.5


def test_full_service_flow_requires_no_human_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = AcceptingProvider()
    service = ReviewService(tmp_path, codex_provider=provider)
    paper = service.add_paper(
        filename="paper.md",
        payload=(
            b"# Optimization Study\n\n## Method\n"
            b"The method uses Adam with a learning rate of 0.001.\n"
        ),
    )
    monkeypatch.setattr(
        AutonomousCorpusCurator,
        "_deterministic_review",
        lambda self, interaction, candidate: {
            "reviewer_results": [{"gates": {"all_required_checks": True}}],
            "mandatory_human_categories": [],
        },
    )
    run = service.start_autonomous_curation(
        paper_ids=(paper["document_id"],),
        config=AutonomousCurationConfig(
            questions_per_paper=40,
            maximum_examples_per_paper=1,
            include_multi_turn=False,
        ),
    )
    assert run["status"] == "completed"
    assert run["report"]["examples_accepted"] == 1
    assert Path(run["dataset_path"]).is_file()
    assert Path(run["manifest_path"]).is_file()
    corrections = service._load_corrections()
    assert len(corrections) == 1
    assert corrections[0].review_status == "codex_curated"
    assert corrections[0].metadata["human_approved"] is False
    for field in (
        "question_origin",
        "answer_producer",
        "codex_review_passes",
        "repair_history",
        "final_adjudicator",
        "source_hashes",
        "lineage_ids",
        "acceptance_confidence",
        "package_version",
        "curation_created_at",
        "duplicate_cluster",
        "approval_provenance",
    ):
        assert field in corrections[0].metadata


def test_suspended_run_resumes_after_reviewer_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ToggleProvider(AcceptingProvider):
        ready = False

        def available(self) -> bool:
            return self.ready

    provider = ToggleProvider()
    service = ReviewService(tmp_path, codex_provider=provider)
    paper = service.add_paper(
        filename="resume.md",
        payload=b"# Resume\n\n## Method\nThe paper describes a local method.\n",
    )
    monkeypatch.setattr(
        AutonomousCorpusCurator,
        "_deterministic_review",
        lambda self, interaction, candidate: {
            "reviewer_results": [{"gates": {"all_required_checks": True}}],
            "mandatory_human_categories": [],
        },
    )
    first = service.start_autonomous_curation(
        paper_ids=(paper["document_id"],),
        config=AutonomousCurationConfig(
            questions_per_paper=40,
            maximum_examples_per_paper=1,
            include_multi_turn=False,
        ),
    )
    assert first["status"] == "suspended"
    assert first["cursor"] == {"paper_index": 0, "question_index": 0}
    provider.ready = True
    resumed = service.resume_autonomous_curation(first["run_id"])
    assert resumed["status"] == "completed"
    assert resumed["report"]["examples_accepted"] == 1


def test_malformed_codex_output_suspends_and_preserves_cursor(tmp_path: Path) -> None:
    class MalformedProvider(AcceptingProvider):
        def review(self, pass_name: str, payload: dict[str, Any]):
            return {"decision": "accept"}

    service = ReviewService(tmp_path, codex_provider=MalformedProvider())
    paper = service.add_paper(
        filename="failure.md",
        payload=b"# Failure\n\n## Method\nThe paper describes a local method.\n",
    )
    run = service.start_autonomous_curation(
        paper_ids=(paper["document_id"],),
        config=AutonomousCurationConfig(
            questions_per_paper=40,
            maximum_examples_per_paper=1,
            include_multi_turn=False,
        ),
    )
    assert run["status"] == "suspended"
    assert run["stage"] == "reviewer_or_validation_error"
    assert run["cursor"] == {"paper_index": 0, "question_index": 0}
    assert "invalid result type" in run["errors"][-1]


def test_test_paper_questions_never_enter_corrections_or_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ReviewService(tmp_path, codex_provider=AcceptingProvider())
    paper_ids = []
    for index in range(3):
        paper = service.add_paper(
            filename=f"paper-{index}.md",
            payload=(
                f"# Paper {index}\n\n## Method\n"
                f"Method {index} uses locally supplied evidence.\n"
            ).encode(),
        )
        paper_ids.append(paper["document_id"])
    monkeypatch.setattr(
        AutonomousCorpusCurator,
        "_deterministic_review",
        lambda self, interaction, candidate: {
            "reviewer_results": [{"gates": {"all_required_checks": True}}],
            "mandatory_human_categories": [],
        },
    )
    run = service.start_autonomous_curation(
        paper_ids=tuple(paper_ids),
        config=AutonomousCurationConfig(
            questions_per_paper=40,
            maximum_examples_per_paper=1,
            include_multi_turn=False,
        ),
    )
    test_papers = {
        paper_id for paper_id, split in run["paper_splits"].items() if split == "test"
    }
    corrections = service._load_corrections()
    assert test_papers
    assert all(not (set(item.paper_ids) & test_papers) for item in corrections)
    assert run["report"]["test_evaluation_question_count"] == 40
    assert all(
        not (set(item.get("paper_ids", [])) & test_papers)
        for item in run["records"]
        if item.get("status") == "codex_curated"
    )
