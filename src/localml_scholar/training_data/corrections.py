"""Candidate correction construction and explicit human approval."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from localml_scholar.training_data.instructions import infer_instruction_profile
from localml_scholar.training_data.provenance import ReviewProvenance, content_sha256
from localml_scholar.training_data.schemas import (
    ConversationTurn,
    GroundedFact,
    GroundedInstructionExample,
    StructuredGroundedTarget,
)


def _evidence_citation_id(item: dict[str, Any]) -> str:
    label = item.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    evidence_id = item.get("evidence_id")
    if isinstance(evidence_id, str) and evidence_id.strip():
        return evidence_id.strip()
    raise ValueError("Every evidence item must have a label or evidence_id.")


def propose_correction(
    interaction: dict[str, Any],
    *,
    review_label: str,
    corrected_answer: str | None = None,
    required_facts: tuple[str, ...] = (),
    prohibited_claims: tuple[str, ...] = (),
    derivation_steps: tuple[GroundedFact, ...] = (),
    notes: str = "",
) -> GroundedInstructionExample:
    """Build an unapproved correction candidate from an immutable interaction.

    Extracted answer text is only a suggestion. This function cannot create a
    human-approved record; :func:`approve_correction` is a separate explicit act.
    """
    if not isinstance(interaction, dict):
        raise TypeError("interaction must be a dictionary.")
    question = interaction.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("interaction must contain a non-empty question.")
    answer = interaction.get("answer")
    if not isinstance(answer, dict):
        raise ValueError("interaction must contain an answer object.")
    evidence = answer.get("evidence", [])
    if not isinstance(evidence, list) or not all(
        isinstance(item, dict) for item in evidence
    ):
        raise ValueError("interaction answer evidence must be a list of objects.")
    paper_ids = interaction.get("paper_ids")
    if paper_ids is None:
        document_id = interaction.get("document_id")
        paper_ids = [document_id] if isinstance(document_id, str) else []
    if not isinstance(paper_ids, (list, tuple)) or not all(
        isinstance(item, str) and item.strip() for item in paper_ids
    ):
        raise ValueError("interaction must identify at least one paper.")
    suggested = answer.get("answer_text", "")
    final_answer = corrected_answer if corrected_answer is not None else suggested
    if not isinstance(final_answer, str) or not final_answer.strip():
        if review_label == "should_abstain":
            final_answer = (
                "The selected paper evidence is insufficient to answer this question."
            )
        else:
            raise ValueError("A non-empty corrected answer is required.")

    citation_ids = tuple(_evidence_citation_id(item) for item in evidence)
    fact_candidates = required_facts
    if not fact_candidates and review_label != "should_abstain" and citation_ids:
        inferred_fact = re.sub(r"\[C[1-9][0-9]*\]", "", final_answer).strip()
        if inferred_fact:
            fact_candidates = (inferred_fact,)
    facts = tuple(
        GroundedFact(
            text=fact,
            provenance="paper_explicit" if citation_ids else "uncertain",
            citation_ids=citation_ids,
            confidence=1.0 if citation_ids else 0.0,
        )
        for fact in fact_candidates
    )
    unresolved = ()
    if review_label == "should_abstain":
        unresolved = (question.strip(),)
    target = StructuredGroundedTarget(
        facts=facts,
        derivation_steps=derivation_steps,
        unresolved_items=unresolved,
        prohibited_claims=prohibited_claims,
    )
    recent = tuple(
        ConversationTurn(**item)
        for item in interaction.get("conversation_turns", [])
        if isinstance(item, dict)
    )
    profile_state = interaction.get("instruction_profile")
    profile = (
        infer_instruction_profile(question, recent_turns=recent)
        if not isinstance(profile_state, dict)
        else infer_instruction_profile(
            question,
            recent_turns=recent,
            explicit_overrides={
                key: value
                for key, value in profile_state.items()
                if key
                in {
                    "desired_depth",
                    "mathematical_depth",
                    "assumed_background",
                    "explanation_style",
                    "output_format",
                    "verbosity",
                    "use_analogy",
                    "include_derivation",
                    "include_critique",
                    "include_comparison",
                    "simplify_previous",
                    "constraints",
                    "canonical_audience",
                }
            },
        )
    )
    turns = recent + (ConversationTurn("user", question.strip()),)
    provenance = ReviewProvenance(
        producer_system="localml_scholar_grounded_answer_pipeline",
        producer_version="1.2.1",
        reviewer_system="pending_human_review",
        reviewer_version="1",
        correction_system="localml_scholar_correction_editor",
        source_hashes=tuple(content_sha256(item) for item in evidence)
        or (content_sha256({"paper_ids": paper_ids}),),
        answer_hash=content_sha256(final_answer.strip()),
        parent_example_ids=(),
    )
    return GroundedInstructionExample.create(
        paper_ids=tuple(paper_ids),
        turns=turns,
        instruction_profile=profile,
        target=target,
        final_answer=final_answer.strip(),
        evidence=tuple(evidence),
        task_type=interaction.get("task_type", "paper_question_answering"),
        review_status="proposed",
        review_label=review_label,
        source_interaction_id=interaction.get("interaction_id"),
        metadata={
            "automatic_correction_is_suggestion": True,
            "human_approval_required": True,
            "review_notes": notes.strip(),
            "original_answer": suggested,
            "source_question_id": interaction.get("question_id"),
            "parent_question_id": interaction.get("parent_question_id"),
            "review_provenance": provenance.to_dict(),
        },
    )


def approve_correction(
    candidate: GroundedInstructionExample,
    *,
    reviewer: str,
) -> GroundedInstructionExample:
    """Explicitly mark one inspected candidate as human-approved."""
    if not isinstance(candidate, GroundedInstructionExample):
        raise TypeError("candidate must be GroundedInstructionExample.")
    if candidate.review_status != "proposed":
        raise ValueError("Only proposed corrections can be approved.")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("reviewer must contain non-whitespace text.")
    if candidate.review_label == "benchmark_problem":
        raise ValueError("A benchmark-problem record cannot become a training example.")
    if candidate.review_label != "should_abstain":
        if not candidate.evidence:
            raise ValueError(
                "A non-abstention training example requires reviewed evidence."
            )
        labels = {
            item.get("label")
            for item in candidate.evidence
            if isinstance(item.get("label"), str)
        }
        cited = set(re.findall(r"\[C[1-9][0-9]*\]", candidate.final_answer))
        cited = {item[1:-1] for item in cited}
        if not cited:
            raise ValueError(
                "A non-abstention final answer requires at least one inline citation."
            )
        if not cited <= labels:
            raise ValueError(
                "Final answer citations must identify retained evidence labels."
            )
    metadata = dict(candidate.metadata)
    source_provenance = metadata.get("review_provenance", {})
    parent_ids = tuple(source_provenance.get("parent_example_ids", []))
    approval_provenance = ReviewProvenance(
        producer_system=source_provenance.get(
            "producer_system", "localml_scholar_grounded_answer_pipeline"
        ),
        producer_version=source_provenance.get("producer_version", "1.2.1"),
        reviewer_system=f"human:{reviewer.strip()}",
        reviewer_version="1",
        correction_system=source_provenance.get("correction_system"),
        source_hashes=tuple(
            source_provenance.get(
                "source_hashes", [content_sha256(item) for item in candidate.evidence]
            )
        ),
        answer_hash=content_sha256(candidate.final_answer),
        parent_example_ids=parent_ids,
        independent_validators=(f"human:{reviewer.strip()}",),
    )
    metadata.update(
        human_approved=True,
        reviewer=reviewer.strip(),
        automatic_correction_is_suggestion=False,
        approval_provenance=approval_provenance.to_dict(),
    )
    return replace(candidate, review_status="human_approved", metadata=metadata)
