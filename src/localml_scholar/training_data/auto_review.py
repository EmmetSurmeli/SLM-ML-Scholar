"""Confidence-gated second-pass review for local grounded-answer drafts.

The reviewers in this module are deliberately transparent deterministic
configurations. They are correlated checks over the same answer and evidence;
they are not represented as independent agents or independent model judgments.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from localml_scholar.training_data.provenance import ReviewProvenance, content_sha256
from localml_scholar.training_data.schemas import QuestionCandidate

AUTO_REVIEW_FORMAT_VERSION = "1.0"
DEFAULT_APPROVAL_THRESHOLD = 0.95
REVIEWER_PROFILES = ("evidence_strict", "answer_strict", "policy_strict")

MANDATORY_GATES = (
    "direct_evidence",
    "answer_relevance",
    "claim_support",
    "citations_present",
    "citations_resolve",
    "citations_support_claims",
    "citations_relevant",
    "prohibited_claims_absent",
    "contradictions_absent",
    "question_answerable",
    "evidence_sufficient",
    "required_concepts_covered",
    "no_external_claim_as_paper_claim",
    "no_inferred_derivation_as_explicit_claim",
    "instruction_following",
    "confidence_threshold",
)

MANDATORY_HUMAN_CATEGORIES = {
    "impact_claim",
    "novelty_claim",
    "first_to_claim",
    "cross_paper_comparison",
    "source_conflict",
    "inferred_derivation",
    "inferred_limitation",
    "research_gap",
    "ambiguous_benchmark",
    "external_literature",
    "extraction_corruption",
    "multiple_interpretations",
    "numeric_contradiction",
    "unusual_citation",
    "metadata_disagreement",
    "figure_table_uncertainty",
    "reviewer_disagreement",
}

_CITATION = re.compile(r"\[C([1-9][0-9]*)\]")
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_HIGH_RISK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("impact_claim", re.compile(r"\b(impact|influential|changed the field)\b", re.I)),
    ("novelty_claim", re.compile(r"\b(novel|novelty|new contribution)\b", re.I)),
    ("first_to_claim", re.compile(r"\b(first|earliest|pioneered)\b", re.I)),
    (
        "research_gap",
        re.compile(r"\b(research gap|open problem|future research)\b", re.I),
    ),
    (
        "external_literature",
        re.compile(r"\b(literature|later work|subsequent work)\b", re.I),
    ),
    ("multiple_interpretations", re.compile(r"\b(ambiguous|interpretations?)\b", re.I)),
    ("figure_table_uncertainty", re.compile(r"\b(figure|table|diagram|plot)\b", re.I)),
)


def _score(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number in [0, 1].")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    return number


@dataclass(frozen=True)
class ReviewerPassResult:
    """One correlated reviewer configuration's scores and gate decisions."""

    reviewer_profile: str
    confidence: float
    evidence_score: float
    answer_score: float
    citation_score: float
    concept_score: float
    completeness_score: float
    instruction_score: float
    style_score: float
    abstention_score: float
    gates: dict[str, bool]
    rationale: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.reviewer_profile not in REVIEWER_PROFILES:
            raise ValueError(f"Unknown reviewer_profile: {self.reviewer_profile}")
        for name in (
            "confidence",
            "evidence_score",
            "answer_score",
            "citation_score",
            "concept_score",
            "completeness_score",
            "instruction_score",
            "style_score",
            "abstention_score",
        ):
            object.__setattr__(self, name, _score(getattr(self, name), name))
        if set(self.gates) != set(MANDATORY_GATES):
            missing = sorted(set(MANDATORY_GATES) - set(self.gates))
            extra = sorted(set(self.gates) - set(MANDATORY_GATES))
            raise ValueError(f"gates mismatch; missing={missing}, extra={extra}.")
        if any(not isinstance(value, bool) for value in self.gates.values()):
            raise TypeError("Every gate decision must be boolean.")
        if not isinstance(self.rationale, (tuple, list)) or not all(
            isinstance(item, str) and item.strip() for item in self.rationale
        ):
            raise TypeError("rationale must contain non-empty strings.")
        object.__setattr__(self, "rationale", tuple(self.rationale))

    @property
    def passed(self) -> bool:
        return all(self.gates.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewer_profile": self.reviewer_profile,
            "reviewers_are_independent": False,
            "confidence": self.confidence,
            "evidence_score": self.evidence_score,
            "evidence_quality": self.evidence_score,
            "answer_score": self.answer_score,
            "answer_relevance": self.answer_score,
            "factual_support": float(self.gates["claim_support"]),
            "citation_score": self.citation_score,
            "citation_support": self.citation_score,
            "citation_relevance": float(self.gates["citations_relevant"]),
            "concept_score": self.concept_score,
            "completeness_score": self.completeness_score,
            "instruction_score": self.instruction_score,
            "style_score": self.style_score,
            "abstention_score": self.abstention_score,
            "gates": dict(self.gates),
            "passed": self.passed,
            "rationale": list(self.rationale),
        }


@dataclass(frozen=True)
class AutoReviewDecision:
    """Final second-pass recommendation and immutable supporting diagnostics."""

    review_status: str
    confidence: float
    reviewer_results: tuple[ReviewerPassResult, ...]
    mandatory_human_categories: tuple[str, ...]
    rationale: tuple[str, ...]
    corrected_answer: str | None
    correction_revalidated: bool
    human_review_route: str | None
    provenance: ReviewProvenance
    calibration_state: str
    example_id: str | None = None
    format_version: str = AUTO_REVIEW_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.review_status not in {
            "codex_approved",
            "codex_rejected",
            "needs_human_review",
            "ambiguous",
            "benchmark_problem",
        }:
            raise ValueError("Invalid automated review_status.")
        if self.example_id is not None and (
            not isinstance(self.example_id, str) or not self.example_id.strip()
        ):
            raise ValueError("example_id must be None or non-whitespace text.")
        object.__setattr__(self, "confidence", _score(self.confidence, "confidence"))
        if not self.reviewer_results:
            raise ValueError("reviewer_results must not be empty.")
        unknown = set(self.mandatory_human_categories) - MANDATORY_HUMAN_CATEGORIES
        if unknown:
            raise ValueError(f"Unknown mandatory-human categories: {sorted(unknown)}")
        if self.review_status == "codex_approved":
            if self.calibration_state != "auto_approval_enabled":
                raise ValueError("codex_approved requires enabled calibration.")
            if self.mandatory_human_categories:
                raise ValueError("Mandatory-human cases cannot be codex_approved.")
            if not all(result.passed for result in self.reviewer_results):
                raise ValueError("codex_approved requires every reviewer gate to pass.")
        if self.corrected_answer is not None and not self.correction_revalidated:
            raise ValueError(
                "Every proposed correction must be revalidated from scratch."
            )

    def to_dict(self) -> dict[str, Any]:
        would_approve = not self.mandatory_human_categories and all(
            item.passed for item in self.reviewer_results
        )
        failed_gates = sorted(
            {
                gate
                for result in self.reviewer_results
                for gate, passed in result.gates.items()
                if not passed
            }
        )
        return {
            "format_version": self.format_version,
            "example_id": self.example_id,
            "decision": self.review_status,
            "review_status": self.review_status,
            "confidence": self.confidence,
            "evidence_quality": min(
                item.evidence_score for item in self.reviewer_results
            ),
            "answer_relevance": min(
                item.answer_score for item in self.reviewer_results
            ),
            "factual_support": min(
                float(item.gates["claim_support"]) for item in self.reviewer_results
            ),
            "citation_support": min(
                item.citation_score for item in self.reviewer_results
            ),
            "citation_relevance": min(
                float(item.gates["citations_relevant"])
                for item in self.reviewer_results
            ),
            "concept_coverage": min(
                item.concept_score for item in self.reviewer_results
            ),
            "completeness": min(
                item.completeness_score for item in self.reviewer_results
            ),
            "instruction_following": min(
                item.instruction_score for item in self.reviewer_results
            ),
            "style_match": min(item.style_score for item in self.reviewer_results),
            "abstention_correctness": min(
                item.abstention_score for item in self.reviewer_results
            ),
            "detected_failures": failed_gates,
            "reviewer_results": [item.to_dict() for item in self.reviewer_results],
            "reviewers_are_independent": False,
            "reviewer_correlation_note": (
                "Three deterministic configurations inspect the same local artifact; "
                "their agreement is not independent replication."
            ),
            "mandatory_human_categories": list(self.mandatory_human_categories),
            "rationale": list(self.rationale),
            "corrected_answer": self.corrected_answer,
            "corrected_evidence": None,
            "corrected_structured_target": None,
            "correction_revalidated": self.correction_revalidated,
            "human_review_route": self.human_review_route,
            "requires_human_review": self.review_status == "needs_human_review",
            "provenance": self.provenance.to_dict(),
            "calibration_state": self.calibration_state,
            "would_approve_if_enabled": would_approve,
        }


@dataclass(frozen=True)
class AutoReviewPolicy:
    """Safety policy for automatic second-pass decisions."""

    approval_threshold: float = DEFAULT_APPROVAL_THRESHOLD
    calibration_state: str = "calibration_required"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "approval_threshold",
            _score(self.approval_threshold, "approval_threshold"),
        )
        if self.approval_threshold < 0.5:
            raise ValueError("approval_threshold must be at least 0.5.")
        if self.calibration_state not in {
            "calibration_required",
            "calibration_active",
            "auto_approval_enabled",
            "auto_approval_suspended",
        }:
            raise ValueError("Unknown calibration_state.")


def _terms(text: str) -> set[str]:
    return {item.casefold() for item in _TOKEN.findall(text) if len(item) > 2}


def detect_mandatory_human_categories(
    candidate: QuestionCandidate,
    interaction: dict[str, Any],
) -> tuple[str, ...]:
    """Identify semantic/risk categories that automation may never approve."""
    combined = (
        f"{candidate.question} {interaction.get('answer', {}).get('answer_text', '')}"
    )
    categories = {
        label for label, pattern in _HIGH_RISK_PATTERNS if pattern.search(combined)
    }
    if len(candidate.paper_ids) > 1 or candidate.question_type == "comparison":
        categories.add("cross_paper_comparison")
    if candidate.question_type in {"derivation", "equation"}:
        categories.add("inferred_derivation")
    if candidate.question_type in {"limitation", "critical_reasoning"}:
        categories.add("inferred_limitation")
    diagnostics = interaction.get("diagnostics", {})
    failures = set(diagnostics.get("failure_categories", []))
    mapping = {
        "extraction_error": "extraction_corruption",
        "source_conflict": "source_conflict",
        "numeric_contradiction": "numeric_contradiction",
        "metadata_disagreement": "metadata_disagreement",
        "unusual_citation": "unusual_citation",
    }
    categories.update(mapping[item] for item in failures if item in mapping)
    return tuple(sorted(categories))


def _reviewer_result(
    interaction: dict[str, Any],
    candidate: QuestionCandidate,
    profile: str,
    threshold: float,
) -> ReviewerPassResult:
    answer = interaction.get("answer")
    if not isinstance(answer, dict):
        raise ValueError("interaction must contain an answer object.")
    answer_text = answer.get("answer_text")
    if not isinstance(answer_text, str):
        raise ValueError("interaction answer_text must be a string.")
    evidence = answer.get("evidence", [])
    if not isinstance(evidence, list) or not all(
        isinstance(item, dict) for item in evidence
    ):
        raise ValueError("interaction evidence must be a list of objects.")
    validation = answer.get("validation", {})
    sufficiency = answer.get("sufficiency", {})
    diagnostics = interaction.get("diagnostics", {})
    abstained = bool(answer.get("abstained"))
    expected_abstention = candidate.question_type in {
        "external_context",
        "false_premise",
        "insufficient_evidence",
    }
    accepted = bool(validation.get("accepted", diagnostics.get("accepted", False)))
    unsupported = int(validation.get("unsupported_claim_count", 0))
    citation_coverage = _score(
        validation.get("citation_coverage", 0.0), "citation_coverage"
    )
    query_coverage = _score(
        sufficiency.get("query_term_coverage", 0.0), "query_term_coverage"
    )
    citations = _CITATION.findall(answer_text)
    labels = {
        str(item.get("label", ""))[1:]
        for item in evidence
        if re.fullmatch(r"C[1-9][0-9]*", str(item.get("label", "")))
    }
    citations_resolve = bool(citations) and set(citations) <= labels
    required = {
        _term for concept in candidate.required_concepts for _term in _terms(concept)
    }
    answer_terms = _terms(answer_text)
    concept_score = (
        1.0 if not required else len(required & answer_terms) / len(required)
    )
    prohibited_absent = all(
        not _terms(claim).issubset(answer_terms)
        for claim in candidate.prohibited_claims
    )
    contradiction = "contradiction" in set(diagnostics.get("failure_categories", []))

    evidence_score = (
        1.0
        if expected_abstention and abstained
        else min(1.0, 0.55 * float(bool(evidence)) + 0.45 * query_coverage)
    )
    citation_score = (
        1.0
        if expected_abstention and abstained
        else min(citation_coverage, 1.0 if citations_resolve else 0.0)
    )
    answer_score = (
        1.0
        if expected_abstention and abstained
        else min(
            1.0, 0.55 * query_coverage + 0.45 * float(accepted and unsupported == 0)
        )
    )
    completeness = min(query_coverage, concept_score)
    instruction_score = 1.0 if answer_text.strip() else 0.0
    style_score = 1.0 if len(answer_text) <= 50_000 else 0.0
    abstention_score = 1.0 if abstained == expected_abstention else 0.0
    adjustments = {
        "evidence_strict": (0.04, 0.0, 0.02),
        "answer_strict": (0.0, 0.04, 0.0),
        "policy_strict": (0.0, 0.0, 0.04),
    }
    evidence_penalty, answer_penalty, policy_penalty = adjustments[profile]
    confidence = max(
        0.0,
        min(
            1.0,
            min(
                evidence_score - evidence_penalty,
                answer_score - answer_penalty,
                citation_score,
                concept_score,
                instruction_score,
                1.0 - policy_penalty if prohibited_absent else 0.0,
            ),
        ),
    )
    citations_expected = not (expected_abstention and abstained)
    gates = {
        "direct_evidence": expected_abstention and abstained or bool(evidence),
        "answer_relevance": expected_abstention and abstained or query_coverage >= 0.6,
        "claim_support": expected_abstention
        and abstained
        or (accepted and unsupported == 0),
        "citations_present": not citations_expected or bool(citations),
        "citations_resolve": not citations_expected or citations_resolve,
        "citations_support_claims": not citations_expected or citation_coverage == 1.0,
        "citations_relevant": not citations_expected or query_coverage >= 0.6,
        "prohibited_claims_absent": prohibited_absent,
        "contradictions_absent": not contradiction,
        "question_answerable": abstained == expected_abstention,
        "evidence_sufficient": expected_abstention
        and abstained
        or query_coverage >= 0.6,
        "required_concepts_covered": concept_score == 1.0,
        "no_external_claim_as_paper_claim": candidate.question_type
        != "external_context",
        "no_inferred_derivation_as_explicit_claim": candidate.question_type
        not in {"derivation", "equation"},
        "instruction_following": instruction_score == 1.0,
        "confidence_threshold": confidence >= threshold,
    }
    failed = [name for name, passed in gates.items() if not passed]
    rationale = (
        f"{profile} confidence={confidence:.3f}.",
        "All mandatory gates passed."
        if not failed
        else f"Failed gates: {', '.join(failed)}.",
    )
    return ReviewerPassResult(
        reviewer_profile=profile,
        confidence=confidence,
        evidence_score=evidence_score,
        answer_score=answer_score,
        citation_score=citation_score,
        concept_score=concept_score,
        completeness_score=completeness,
        instruction_score=instruction_score,
        style_score=style_score,
        abstention_score=abstention_score,
        gates=gates,
        rationale=rationale,
    )


def review_interaction_second_pass(
    interaction: dict[str, Any],
    candidate: QuestionCandidate,
    *,
    policy: AutoReviewPolicy | None = None,
    corrected_answer: str | None = None,
    parent_example_ids: tuple[str, ...] = (),
) -> AutoReviewDecision:
    """Review an immutable local interaction under all mandatory confidence gates."""
    if not isinstance(interaction, dict):
        raise TypeError("interaction must be a dictionary.")
    if not isinstance(candidate, QuestionCandidate):
        raise TypeError("candidate must be QuestionCandidate.")
    policy = AutoReviewPolicy() if policy is None else policy
    if not isinstance(policy, AutoReviewPolicy):
        raise TypeError("policy must be AutoReviewPolicy or None.")
    working = interaction
    correction_revalidated = False
    if corrected_answer is not None:
        if not isinstance(corrected_answer, str) or not corrected_answer.strip():
            raise ValueError("corrected_answer must contain non-whitespace text.")
        answer = dict(interaction.get("answer", {}))
        answer["answer_text"] = corrected_answer.strip()
        working = {**interaction, "answer": answer}
        correction_revalidated = True
    results = tuple(
        _reviewer_result(working, candidate, profile, policy.approval_threshold)
        for profile in REVIEWER_PROFILES
    )
    categories = list(detect_mandatory_human_categories(candidate, working))
    passed_values = {item.passed for item in results}
    if len(passed_values) > 1:
        categories.append("reviewer_disagreement")
    categories = sorted(set(categories))
    confidence = min(item.confidence for item in results)
    all_pass = all(item.passed for item in results)
    expected_abstention = candidate.question_type in {
        "external_context",
        "false_premise",
        "insufficient_evidence",
    }
    abstained = bool(working.get("answer", {}).get("abstained"))
    if candidate.review_status == "benchmark_problem":
        status = "benchmark_problem"
    elif categories:
        status = "needs_human_review"
    elif all_pass and policy.calibration_state == "auto_approval_enabled":
        status = "codex_approved"
    elif confidence <= 0.25 and abstained != expected_abstention:
        status = "codex_rejected"
    else:
        status = "needs_human_review"
    answer = working.get("answer", {})
    evidence = answer.get("evidence", [])
    source_hashes = tuple(content_sha256(item) for item in evidence) or (
        content_sha256({"paper_ids": candidate.paper_ids}),
    )
    provenance = ReviewProvenance(
        producer_system="localml_scholar_grounded_answer_pipeline",
        producer_version="1.2.1",
        reviewer_system="localml_scholar_correlated_review_profiles",
        reviewer_version="1.2.1",
        correction_system=(
            "localml_scholar_correlated_review_profiles"
            if corrected_answer is not None
            else None
        ),
        source_hashes=source_hashes,
        answer_hash=content_sha256(answer.get("answer_text", "")),
        parent_example_ids=parent_example_ids,
        independent_validators=(),
    )
    rationale = [
        f"Minimum correlated-review confidence is {confidence:.3f}.",
        f"Calibration state is {policy.calibration_state}.",
    ]
    if categories:
        rationale.append(f"Mandatory human categories: {', '.join(categories)}.")
    if all_pass and policy.calibration_state != "auto_approval_enabled":
        rationale.append(
            "All gates passed, but automatic approval is calibration-locked."
        )
    return AutoReviewDecision(
        review_status=status,
        confidence=confidence,
        reviewer_results=results,
        mandatory_human_categories=tuple(categories),
        rationale=tuple(rationale),
        corrected_answer=corrected_answer.strip()
        if corrected_answer is not None
        else None,
        correction_revalidated=correction_revalidated,
        human_review_route=(
            "high_risk" if categories else "calibration_or_gate_failure"
        )
        if status == "needs_human_review"
        else None,
        provenance=provenance,
        calibration_state=policy.calibration_state,
        example_id=candidate.question_id,
    )
