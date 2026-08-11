"""Focused safety and integration tests for Milestone 12A.3."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from localml_scholar.review_app.autonomous_curation import (
    AutonomousCorpusCurator,
    _unique_candidate_states,
)
from localml_scholar.review_app.service import ReviewService
from localml_scholar.training_data import (
    AutonomousCurationConfig,
    CodexCLIReviewProvider,
    CodexReview,
    CurationSuspended,
    QuestionCandidate,
    autonomous_quality_report,
    balanced_paper_splits,
    blind_payload,
    build_dataset,
    curate_interaction,
    propose_correction,
)


def test_resume_rejects_concurrent_process_without_mutating_state(
    tmp_path: Path,
) -> None:
    service = ReviewService(tmp_path, codex_provider=AcceptingProvider())
    curator = AutonomousCorpusCurator(service)
    run_id = "curation-concurrent"
    run_directory = service.autonomous_output_directory / run_id
    run_directory.mkdir(parents=True)
    (run_directory / ".resume.lock").write_text(str(os.getpid()), encoding="utf-8")

    with pytest.raises(CurationSuspended, match="already active"):
        curator.resume(run_id)


def test_invalidated_diagnostic_is_suspended_and_cannot_resume(
    tmp_path: Path,
) -> None:
    service = ReviewService(tmp_path, codex_provider=AcceptingProvider())
    paper = service.add_paper(
        filename="paper.md",
        payload=b"# Paper\n\n## Method\nA local method is described here.\n",
    )
    curator = AutonomousCorpusCurator(service)
    run = curator.create(
        paper_ids=(paper["document_id"],),
        config=AutonomousCurationConfig(
            questions_per_paper=40,
            include_multi_turn=False,
        ),
    )
    run["diagnostic"] = {"controlled": True, "valid_for_readiness": True}
    curator._persist(run)

    frozen = curator.invalidate_for_readiness(run["run_id"], "fixture defect")
    resumed = curator.resume(run["run_id"])

    assert frozen["status"] == "suspended"
    assert frozen["stage"] == "readiness_invalidated"
    assert resumed["cursor"] == frozen["cursor"]
    assert resumed["records"] == frozen["records"]


def test_retrieval_preflight_keeps_only_matching_answerability(
    tmp_path: Path,
) -> None:
    service = ReviewService(tmp_path, codex_provider=AcceptingProvider())
    paper = service.add_paper(
        filename="paper.md",
        payload=(
            b"# Study\n\n## Method\nThe method uses Adam optimization. "
            b"Memory complexity is discussed. Batch size guides training.\n\n"
            b"## Results\nAccuracy improves.\n"
        ),
    )
    answerable = QuestionCandidate.create(
        paper_ids=(paper["document_id"],),
        question="What method is used?",
        question_type="method",
        expected_sections=("method",),
    ).to_dict()
    unsupported = QuestionCandidate.create(
        paper_ids=(paper["document_id"],),
        question="What ablation was performed?",
        question_type="ablation",
        expected_sections=("ablation",),
    ).to_dict()
    vague_complexity = QuestionCandidate.create(
        paper_ids=(paper["document_id"],),
        question="What is the memory complexity?",
        question_type="complexity",
        expected_sections=("method",),
    ).to_dict()
    missing_numeric_value = QuestionCandidate.create(
        paper_ids=(paper["document_id"],),
        question="What batch size was used?",
        question_type="reproduction",
        expected_sections=("method",),
    ).to_dict()

    accepted, rejected = AutonomousCorpusCurator(
        service
    )._retrieval_preflight_candidates(
        [answerable, unsupported, vague_complexity, missing_numeric_value]
    )

    assert [item["question_id"] for item in accepted] == [answerable["question_id"]]
    assert accepted[0]["metadata"]["retrieval_preflight_passed"] is True
    assert rejected == 3


def test_codex_cli_provider_uses_supported_noninteractive_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured["kwargs"] = kwargs
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(json.dumps(_review().to_dict()), encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(
        "localml_scholar.training_data.codex_review.subprocess.run", fake_run
    )
    provider = CodexCLIReviewProvider(tmp_path)
    result = provider.review("evidence_critic", {"question": "Fixture question"})

    command = captured["command"]
    assert isinstance(command, list)
    assert command[:2] == ["codex", "exec"]
    assert "--ephemeral" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--ask-for-approval" not in command
    assert "--output-schema" in command
    assert "--output-last-message" in command
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert result.decision == "accept"


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
        review = _review(
            corrected_answer=(
                "The supplied evidence supports this answer. [C1]"
                if pass_name == "answerer"
                else None
            )
        )
        if pass_name == "answerer":
            evidence = payload["source_passages"][0]
            text = evidence.get("selected_text", evidence.get("text"))
            target = {
                "facts": [
                    {
                        "text": text,
                        "provenance": "paper_explicit",
                        "citation_ids": [evidence["evidence_id"]],
                        "confidence": 0.99,
                    }
                ],
                "equations": [],
                "derivation_steps": [],
                "assumptions": [],
                "qualifications": [],
                "limitations": [],
                "unresolved_items": [],
                "prohibited_claims": [],
            }
            return replace(review, corrected_target=target)
        return review


class RepairingProvider(AcceptingProvider):
    def __init__(self, *, never_succeeds: bool = False) -> None:
        super().__init__()
        self.cycle = 0
        self.never_succeeds = never_succeeds

    def review(self, pass_name: str, payload: dict[str, Any]) -> CodexReview:
        repairing = self.never_succeeds or self.cycle == 0
        if pass_name == "answerer":
            return super().review(pass_name, payload)
        self.calls.append((pass_name, payload))
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


def test_unchanged_repair_stops_before_third_review_cycle() -> None:
    interaction, candidate, deterministic = _fixture()
    provider = RepairingProvider(never_succeeds=True)

    record = curate_interaction(
        interaction,
        candidate,
        deterministic,
        provider=provider,
        config=AutonomousCurationConfig(maximum_repair_attempts=2),
    )

    assert record["status"] == "rejected"
    assert record["repair_attempts"] == 2
    assert record["terminal_reasons"] == ["deterministic_repair_no_progress"]
    assert len(record["codex_review_passes"]) == 10


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


def test_dataset_export_uses_splits_only_for_accepted_papers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RejectSecondCandidateProvider(AcceptingProvider):
        answerer_count = 0

        def review(self, pass_name: str, payload: dict[str, Any]) -> CodexReview:
            if pass_name == "answerer":
                self.answerer_count += 1
                return super().review(pass_name, payload)
            self.calls.append((pass_name, payload))
            if self.answerer_count >= 2:
                return _review("reject", unsupported_claims=("fixture rejection",))
            return _review()

    service = ReviewService(tmp_path, codex_provider=RejectSecondCandidateProvider())
    paper_ids = []
    for index in range(2):
        paper = service.add_paper(
            filename=f"subset-{index}.md",
            payload=(
                f"# Study {index}\n\n## Method\n"
                f"Method {index} uses a local optimizer.\n"
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
            validation_fraction=0.0,
            test_fraction=0.0,
            include_multi_turn=False,
        ),
    )

    assert run["status"] == "completed"
    assert run["report"]["examples_accepted"] == 1
    dataset = json.loads(Path(run["dataset_path"]).read_text(encoding="utf-8"))
    accepted_papers = {
        paper_id
        for record in run["records"]
        if record["status"] == "codex_curated"
        for paper_id in record["paper_ids"]
    }
    assert set(dataset["paper_splits"]) == accepted_papers


def test_autonomous_repair_uses_lexical_retrieval_on_lexical_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ReviewService(tmp_path, codex_provider=RepairingProvider())
    paper = service.add_paper(
        filename="repair.md",
        payload=(
            b"# Repair Study\n\n## Method\n"
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
    assert run["report"]["examples_repaired"] == 1
    accepted = next(
        item for item in run["records"] if item["status"] == "codex_curated"
    )
    assert accepted["repair_attempts"] == 1


def test_replacement_evidence_clears_stale_structured_target() -> None:
    interaction, candidate, deterministic = _fixture()

    class EvidenceReplacementProvider(AcceptingProvider):
        cycle = 0

        def review(self, pass_name: str, payload: dict[str, Any]) -> CodexReview:
            self.calls.append((pass_name, payload))
            if pass_name == "answerer" and self.cycle == 0:
                stale_target = {
                    "facts": [
                        {
                            "text": "Old evidence fact.",
                            "provenance": "paper_explicit",
                            "citation_ids": ["chunk-1"],
                            "confidence": 0.99,
                        }
                    ],
                    "equations": [],
                    "derivation_steps": [],
                    "assumptions": [],
                    "qualifications": [],
                    "limitations": [],
                    "unresolved_items": [],
                    "prohibited_claims": [],
                }
                return replace(
                    _review(corrected_answer="Supported answer. [C1]"),
                    corrected_target=stale_target,
                )
            if pass_name == "evidence_critic" and self.cycle == 0:
                return _review(
                    "repair",
                    evidence_relevance=0.4,
                    required_corrections=("retrieve stronger evidence",),
                )
            if pass_name == "final_adjudicator" and self.cycle == 0:
                self.cycle = 1
                return _review("repair", required_corrections=("repair",))
            if pass_name == "answerer":
                assert "facts" not in payload["structured_target"]
                target = {
                    "facts": [
                        {
                            "text": "The method uses stronger replacement evidence.",
                            "provenance": "paper_explicit",
                            "citation_ids": ["chunk-2"],
                            "confidence": 0.99,
                        }
                    ],
                    "equations": [],
                    "derivation_steps": [],
                    "assumptions": [],
                    "qualifications": [],
                    "limitations": [],
                    "unresolved_items": [],
                    "prohibited_claims": [],
                }
                return replace(
                    _review(corrected_answer="Supported answer. [C1]"),
                    corrected_evidence_ids=("chunk-2",),
                    corrected_target=target,
                )
            return _review()

    replacement = {
        "label": "C1",
        "evidence_id": "chunk-2",
        "chunk_id": "chunk-2",
        "document_id": "paper-a",
        "selected_text": "Stronger replacement evidence.",
        "text": "Stronger replacement evidence.",
    }
    record = curate_interaction(
        interaction,
        candidate,
        deterministic,
        provider=EvidenceReplacementProvider(),
        retrieve_evidence=lambda query, paper_ids, attempt: [replacement],
        revalidate=lambda answer, state: deterministic,
    )

    assert record["status"] == "codex_curated"
    assert record["repair_attempts"] == 1
    assert record["answer"]["evidence"][0]["evidence_id"] == "chunk-2"


def test_answerer_evidence_id_citations_are_normalized_to_display_labels() -> None:
    interaction, candidate, deterministic = _fixture()

    class EvidenceIdCitationProvider(AcceptingProvider):
        def review(self, pass_name: str, payload: dict[str, Any]) -> CodexReview:
            self.calls.append((pass_name, payload))
            return _review(
                corrected_answer=(
                    "The supplied evidence supports this answer. [chunk-1]"
                    if pass_name == "answerer"
                    else None
                )
            )

    record = curate_interaction(
        interaction,
        candidate,
        deterministic,
        provider=EvidenceIdCitationProvider(),
    )

    assert record["status"] == "codex_curated"
    assert record["answer"]["answer_text"].endswith("[C1]")


def test_candidate_deduplication_preserves_first_stable_question() -> None:
    candidates = [
        {"question_id": "question-a", "question": "First"},
        {"question_id": "question-a", "question": "Duplicate"},
        {"question_id": "question-b", "question": "Second"},
    ]

    assert _unique_candidate_states(candidates) == [candidates[0], candidates[2]]


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
    expected_test_questions = sum(
        len(run["candidates"][paper_id]) for paper_id in test_papers
    )
    assert run["report"]["test_evaluation_question_count"] == expected_test_questions
    assert all(
        not (set(item.get("paper_ids", [])) & test_papers)
        for item in run["records"]
        if item.get("status") == "codex_curated"
    )
