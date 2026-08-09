"""Transparent first-pass review proposals for batch paper evaluation."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from localml_scholar.training_data.auto_review import (
    AutoReviewPolicy,
    review_interaction_second_pass,
)
from localml_scholar.training_data.schemas import QuestionCandidate

_EXPLANATORY_TYPES = {
    "architecture",
    "comparison",
    "complexity",
    "counterfactual",
    "critical_reasoning",
    "derivation",
    "equation",
    "extension",
    "intuition",
    "interpretation",
    "limitation",
    "method",
    "prerequisites",
    "summary",
    "teaching",
    "user_style",
    "user_authored",
}
_EXPECTED_ABSTENTION_TYPES = {
    "external_context",
    "false_premise",
    "insufficient_evidence",
}
_CITATION = re.compile(r"\s*\[C[1-9][0-9]*\]\s*")


def _review_id(batch_id: str, interaction_id: str) -> str:
    digest = hashlib.sha256(f"{batch_id}:{interaction_id}".encode()).hexdigest()
    return f"auto_review_{digest[:20]}"


def _facts_from_answer(answer_text: str) -> list[str]:
    facts = []
    for line in answer_text.splitlines():
        cleaned = line.strip().removeprefix("- ").strip()
        cleaned = _CITATION.sub(" ", cleaned)
        cleaned = " ".join(cleaned.split())
        if cleaned and cleaned.casefold() != "the indexed sources state:":
            facts.append(cleaned)
    return list(dict.fromkeys(facts))[:6]


def propose_automatic_review(
    interaction: dict[str, Any],
    candidate: QuestionCandidate,
    *,
    batch_id: str,
    policy: AutoReviewPolicy | None = None,
) -> dict[str, Any]:
    """Propose a cautious review from transparent deterministic diagnostics.

    This function is not a semantic judge. Its output is always a draft and
    cannot enter a correction dataset without an explicit user decision.
    """
    if not isinstance(interaction, dict):
        raise TypeError("interaction must be a dictionary.")
    if not isinstance(candidate, QuestionCandidate):
        raise TypeError("candidate must be QuestionCandidate.")
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ValueError("batch_id must contain non-whitespace text.")
    answer = interaction.get("answer")
    if not isinstance(answer, dict):
        raise ValueError("interaction must contain an answer object.")
    diagnostics = interaction.get("diagnostics", {})
    validation = answer.get("validation", {})
    sufficiency = answer.get("sufficiency", {})
    evidence = answer.get("evidence", [])
    if not isinstance(evidence, list):
        raise ValueError("answer evidence must be a list.")
    abstained = bool(answer.get("abstained"))
    accepted = bool(validation.get("accepted", diagnostics.get("accepted", False)))
    citation_coverage = float(validation.get("citation_coverage", 0.0))
    query_coverage = float(sufficiency.get("query_term_coverage", 0.0))
    unsupported = int(validation.get("unsupported_claim_count", 0))
    comparison_incomplete = bool(
        interaction.get("comparison", {}).get("requested")
        and not interaction.get("comparison", {}).get("complete")
    )
    expected_abstention = candidate.question_type in _EXPECTED_ABSTENTION_TYPES

    rationale: list[str] = []
    if expected_abstention and abstained:
        label = "correct"
        confidence = 0.86
        rationale.append(
            "The question category expects caution or another source, and the "
            "system abstained."
        )
    elif expected_abstention and not abstained:
        label = "should_abstain"
        confidence = 0.82
        rationale.append(
            "This candidate is designed to test a false premise, missing source, "
            "or insufficient evidence, but the system answered."
        )
    elif abstained:
        label = "incorrect"
        confidence = 0.76
        rationale.append(
            "The question is intended to be paper-answerable, but the system "
            "returned no answer."
        )
    elif unsupported or not accepted:
        label = "incorrect"
        confidence = 0.84
        rationale.append(
            "The answer failed citation/claim acceptance or contains an "
            "unsupported claim diagnostic."
        )
    elif comparison_incomplete:
        label = "partial"
        confidence = 0.88
        rationale.append(
            "The comparison did not retrieve evidence from every selected paper."
        )
    elif candidate.question_type in _EXPLANATORY_TYPES:
        label = "partial"
        confidence = 0.68
        rationale.append(
            "The cited extractive answer is grounded, but explanatory questions "
            "usually need synthesis or teaching beyond quoted facts."
        )
    elif query_coverage < 0.6:
        label = "partial"
        confidence = 0.72
        rationale.append(
            "The selected evidence covers fewer than 60% of meaningful query terms."
        )
    elif citation_coverage == 1.0 and evidence:
        label = "correct"
        confidence = 0.74
        rationale.append(
            "The answer passed deterministic validation and every detected claim "
            "has a retained citation."
        )
    else:
        label = "partial"
        confidence = 0.6
        rationale.append(
            "The checks found some support, but they cannot establish semantic "
            "completeness."
        )

    if citation_coverage < 1.0 and not abstained:
        rationale.append(f"Citation coverage is {citation_coverage:.0%}.")
    if query_coverage < 1.0:
        rationale.append(f"Query-term coverage is {query_coverage:.0%}.")
    if not candidate.required_concepts:
        rationale.append(
            "No human-authored required-concept target exists yet, so completeness "
            "cannot be automatically verified."
        )

    answer_text = answer.get("answer_text", "")
    corrected_answer = answer_text
    if label == "should_abstain":
        corrected_answer = (
            "The selected paper evidence is insufficient to answer this question. "
            "Another source or a narrower question is required."
        )
    evidence_ids = [
        item.get("evidence_id", item.get("chunk_id", item.get("label")))
        for item in evidence
    ]
    evidence_ids = [item for item in evidence_ids if isinstance(item, str)]
    needs_answer_edit = label in {"partial", "incorrect"} and not abstained
    second_pass = review_interaction_second_pass(
        interaction, candidate, policy=policy
    ).to_dict()
    return {
        "review_id": _review_id(batch_id, interaction["interaction_id"]),
        "batch_id": batch_id,
        "question_id": candidate.question_id,
        "interaction_id": interaction["interaction_id"],
        "paper_ids": list(candidate.paper_ids),
        "question": candidate.question,
        "question_type": candidate.question_type,
        "answer": answer,
        "diagnostics": diagnostics,
        "proposed_label": label,
        "proposed_confidence": confidence,
        "rationale": rationale,
        "proposed_required_facts": _facts_from_answer(answer_text),
        "proposed_prohibited_claims": list(candidate.prohibited_claims),
        "proposed_corrected_answer": corrected_answer,
        "proposed_evidence_ids": evidence_ids,
        "needs_answer_edit": needs_answer_edit,
        "decision": "pending_user_review",
        "reviewer_type": "deterministic_local_first_pass",
        "semantic_judge_used": False,
        "second_pass": second_pass,
        "review_status": second_pass["review_status"],
    }


def propose_automatic_failure_review(
    candidate: QuestionCandidate,
    *,
    batch_id: str,
    error: Exception,
    policy: AutoReviewPolicy | None = None,
) -> dict[str, Any]:
    """Represent one failed answer attempt without aborting the whole batch."""
    if not isinstance(candidate, QuestionCandidate):
        raise TypeError("candidate must be QuestionCandidate.")
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ValueError("batch_id must contain non-whitespace text.")
    if not isinstance(error, Exception):
        raise TypeError("error must be an Exception.")
    message = str(error).strip() or type(error).__name__
    answer_text = "No reviewable grounded answer was produced for this question."
    interaction = {
        "paper_ids": list(candidate.paper_ids),
        "question": candidate.question,
        "answer": {
            "answer_text": answer_text,
            "abstained": True,
            "evidence": [],
            "validation": {
                "accepted": False,
                "citations_valid": False,
                "citation_coverage": 0.0,
                "unsupported_claim_count": 0,
                "rejection_reasons": ["answer_execution_error"],
            },
            "sufficiency": {"query_term_coverage": 0.0},
        },
        "diagnostics": {
            "accepted": False,
            "citation_coverage": 0.0,
            "query_term_coverage": 0.0,
            "failure_categories": ["answer_execution_error"],
            "execution_error": message,
            "human_review_required": True,
        },
    }
    second_pass = review_interaction_second_pass(
        interaction, candidate, policy=policy
    ).to_dict()
    return {
        "review_id": _review_id(batch_id, f"failure:{candidate.question_id}"),
        "batch_id": batch_id,
        "question_id": candidate.question_id,
        "interaction_id": None,
        "paper_ids": list(candidate.paper_ids),
        "question": candidate.question,
        "question_type": candidate.question_type,
        "answer": interaction["answer"],
        "diagnostics": interaction["diagnostics"],
        "proposed_label": "incorrect",
        "proposed_confidence": 1.0,
        "rationale": [
            "The answer attempt raised an error instead of returning a reviewable "
            "grounded response.",
            f"Recorded error: {message}",
        ],
        "proposed_required_facts": list(candidate.required_concepts),
        "proposed_prohibited_claims": list(candidate.prohibited_claims),
        "proposed_corrected_answer": "",
        "proposed_evidence_ids": [],
        "needs_answer_edit": True,
        "saveable": False,
        "default_selected": False,
        "decision": "pending_user_review",
        "reviewer_type": "deterministic_local_first_pass",
        "semantic_judge_used": False,
        "second_pass": second_pass,
        "review_status": second_pass["review_status"],
    }


def summarize_automatic_reviews(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one draft batch without treating proposals as human labels."""
    labels = {
        name: 0
        for name in (
            "correct",
            "partial",
            "incorrect",
            "should_abstain",
            "benchmark_problem",
        )
    }
    for review in reviews:
        label = review.get("proposed_label")
        if label in labels:
            labels[label] += 1
    second_pass_counts: dict[str, int] = {}
    for review in reviews:
        status = review.get("second_pass", {}).get("review_status", "not_run")
        second_pass_counts[status] = second_pass_counts.get(status, 0) + 1
    calibration_states = {
        review.get("second_pass", {}).get("calibration_state")
        for review in reviews
        if isinstance(review.get("second_pass", {}).get("calibration_state"), str)
    }
    calibration_state = (
        next(iter(calibration_states))
        if len(calibration_states) == 1
        else "mixed"
        if calibration_states
        else "calibration_required"
    )
    return {
        "review_count": len(reviews),
        "proposed_label_counts": labels,
        "needs_answer_edit_count": sum(
            bool(item.get("needs_answer_edit")) for item in reviews
        ),
        "pending_user_review_count": sum(
            item.get("decision") == "pending_user_review" for item in reviews
        ),
        "saved_review_count": sum(
            item.get("decision") == "saved_as_user_review" for item in reviews
        ),
        "excluded_count": sum(
            item.get("decision") == "excluded_by_user" for item in reviews
        ),
        "execution_error_count": sum(
            not bool(item.get("saveable", True)) for item in reviews
        ),
        "second_pass_status_counts": dict(sorted(second_pass_counts.items())),
        "automatic_approval_enabled": calibration_state == "auto_approval_enabled",
        "calibration_state": calibration_state,
    }
