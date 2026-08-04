"""Tests for provenance, correction approval, splitting, and dataset export."""

from __future__ import annotations

from dataclasses import replace

import pytest

from localml_scholar.training_data import (
    GroundedFact,
    StructuredGroundedTarget,
    approve_correction,
    assign_paper_splits,
    build_dataset,
    dataset_report,
    load_dataset,
    propose_correction,
    save_dataset,
)
from localml_scholar.training_data.diversity import progress_status


def _interaction(paper_ids=("paper-a",)):
    return {
        "interaction_id": "interaction-1",
        "paper_ids": list(paper_ids),
        "question": "What optimizer is used?",
        "instruction_profile": {"verbosity": "concise"},
        "answer": {
            "answer_text": "The paper uses Adam. [C1]",
            "evidence": [
                {
                    "label": "C1",
                    "document_id": paper_ids[0],
                    "selected_text": "Training uses Adam.",
                }
            ],
        },
    }


def test_provenance_requires_citation_for_paper_explicit_fact():
    with pytest.raises(ValueError, match="citation_id"):
        GroundedFact("The method uses Adam.", "paper_explicit")
    inferred = GroundedFact("The gradient is zero.", "mathematical_inference")
    assert inferred.citation_ids == ()


def test_structured_target_keeps_explicit_and_inferred_steps_distinct():
    target = StructuredGroundedTarget(
        facts=(GroundedFact("Equation A is stated.", "paper_explicit", ("C1",)),),
        derivation_steps=(
            GroundedFact("Expand the product.", "mathematical_inference"),
        ),
    )
    assert target.facts[0].provenance == "paper_explicit"
    assert target.derivation_steps[0].provenance == "mathematical_inference"


def test_correction_is_only_proposed_until_explicit_approval():
    candidate = propose_correction(
        _interaction(),
        review_label="partial",
        corrected_answer="The optimizer is Adam. [C1]",
        required_facts=("The paper uses Adam.",),
    )
    assert candidate.review_status == "proposed"
    assert candidate.metadata["human_approval_required"] is True
    approved = approve_correction(candidate, reviewer="local-reviewer")
    assert approved.review_status == "human_approved"
    assert approved.metadata["reviewer"] == "local-reviewer"


def test_should_abstain_correction_can_supply_standard_target():
    interaction = _interaction()
    interaction["answer"] = {"answer_text": "", "evidence": []}
    candidate = propose_correction(interaction, review_label="should_abstain")
    assert "insufficient" in candidate.final_answer
    assert candidate.target.unresolved_items == (interaction["question"],)


def test_approval_rejects_ungrounded_and_benchmark_problem_examples():
    no_evidence = _interaction()
    no_evidence["answer"] = {"answer_text": "Unsupported answer.", "evidence": []}
    candidate = propose_correction(no_evidence, review_label="incorrect")
    with pytest.raises(ValueError, match="requires reviewed evidence"):
        approve_correction(candidate, reviewer="reviewer")
    benchmark_problem = propose_correction(
        _interaction(), review_label="benchmark_problem"
    )
    with pytest.raises(ValueError, match="cannot become a training example"):
        approve_correction(benchmark_problem, reviewer="reviewer")


def test_paper_splits_are_deterministic_and_manual_assignments_win():
    first = assign_paper_splits(
        ("a", "b", "c", "d"), seed=42, manual_assignments={"a": "test"}
    )
    second = assign_paper_splits(
        ("d", "c", "b", "a"), seed=42, manual_assignments={"a": "test"}
    )
    assert first == second
    assert first["a"] == "test"


def test_dataset_export_filters_unapproved_and_round_trips(tmp_path):
    candidate = propose_correction(
        _interaction(), review_label="correct", required_facts=("Adam is used.",)
    )
    approved = approve_correction(candidate, reviewer="reviewer")
    dataset = build_dataset(
        (candidate, approved),
        manual_paper_splits={"paper-a": "test"},
    )
    assert len(dataset.examples) == 1
    assert dataset.examples[0].split == "test"
    path = save_dataset(dataset, tmp_path / "dataset.json")
    loaded = load_dataset(path)
    assert loaded.to_dict() == dataset.to_dict()
    report = dataset_report(loaded)
    assert report["paper_level_leakage"] is False
    assert report["split_example_counts"]["test"] == 1


def test_dataset_rejects_explicit_unapproved_export():
    candidate = propose_correction(_interaction(), review_label="correct")
    with pytest.raises(ValueError, match="cannot include"):
        build_dataset((candidate,), approved_only=False)


def test_cross_paper_example_cannot_leak_across_splits():
    candidate = propose_correction(
        _interaction(("paper-a", "paper-b")), review_label="correct"
    )
    approved = approve_correction(candidate, reviewer="reviewer")
    dataset = build_dataset((approved,), seed=8)
    assert dataset.paper_splits["paper-a"] == dataset.paper_splits["paper-b"]
    assert dataset.examples[0].split == dataset.paper_splits["paper-a"]
    with pytest.raises(ValueError, match="connected cross-paper"):
        build_dataset(
            (approved,),
            manual_paper_splits={"paper-a": "train", "paper-b": "test"},
        )


def test_schema_rejects_example_split_mismatch():
    approved = approve_correction(
        propose_correction(_interaction(), review_label="correct"),
        reviewer="reviewer",
    )
    mismatched = replace(approved, split="train")
    from localml_scholar.training_data.schemas import GroundedInstructionDataset

    with pytest.raises(ValueError, match="does not match"):
        GroundedInstructionDataset("1.0", (mismatched,), {"paper-a": "test"})


def test_progress_targets_are_100_300_600():
    status = progress_status(120)
    assert [item["count"] for item in status["targets"]] == [100, 300, 600]
    assert status["next_target"] == 300
