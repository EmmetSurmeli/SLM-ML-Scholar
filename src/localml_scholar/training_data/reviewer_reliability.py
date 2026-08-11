"""Deterministic reviewer-agreement and claim-level citation reliability tools.

The helpers in this module deliberately do not attempt semantic proof.  They
provide one shared vocabulary and structural/lexical gate around the Codex
review passes so obvious identity, citation, and reviewer-contract failures are
handled consistently before an example can be accepted for training.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from localml_scholar.retrieval import section_topics_compatible
from localml_scholar.training_data.claim_alignment import (
    build_claim_graph,
    claim_graph_metrics,
)
from localml_scholar.training_data.provenance import content_sha256

REVIEW_POLICY_VERSION = "1.0"
DEFAULT_AUTONOMOUS_TRAINING_EXCLUSIONS = frozenset(
    {
        "complexity",
        "critical_reasoning",
        "derivation",
        "equation",
        "figure_interpretation",
        "historical_impact",
        "cross_paper_synthesis",
    }
)


class EvidenceDecision(str, Enum):
    """Whether supplied passages can support a correct answer."""

    DIRECT = "direct"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    IRRELEVANT = "irrelevant"
    EXTERNAL_REQUIRED = "external_required"


class ClaimSupport(str, Enum):
    """How a substantive answer claim relates to local evidence."""

    EXPLICIT = "explicit"
    INFERRED_VALID = "inferred_valid"
    EXTERNAL = "external"
    UNSUPPORTED = "unsupported"


class CitationDecision(str, Enum):
    """Claim-level citation support outcome."""

    SUPPORTS = "supports"
    PARTIAL = "partial"
    WRONG_SOURCE = "wrong_source"
    WRONG_SPAN = "wrong_span"
    IRRELEVANT = "irrelevant"
    MISSING = "missing"
    MALFORMED = "malformed"


class DisagreementSeverity(str, Enum):
    """Whether a reviewer conflict is safety-relevant or stylistic."""

    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True)
class AnswerClaim:
    """One atomic factual answer claim and its normalized citation labels."""

    claim_id: str
    text: str
    claim_type: str
    citation_labels: tuple[str, ...]
    support_type: ClaimSupport = ClaimSupport.UNSUPPORTED
    required_support: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "claim_type": self.claim_type,
            "citation_labels": list(self.citation_labels),
            "support_type": self.support_type.value,
            "required_support": self.required_support,
        }


@dataclass(frozen=True)
class CitationCritique:
    """Deterministic or model-produced judgment for one atomic claim."""

    claim_id: str
    support_status: CitationDecision
    relevance_status: CitationDecision
    source_match: bool
    missing_information: tuple[str, ...]
    confidence: float

    def __post_init__(self) -> None:
        if not self.claim_id.strip():
            raise ValueError("claim_id must contain text.")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1].")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "support_status": self.support_status.value,
            "relevance_status": self.relevance_status.value,
            "source_match": self.source_match,
            "missing_information": list(self.missing_information),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CitationNormalization:
    """Canonicalized prose and the labels that could or could not be resolved."""

    text: str
    labels: tuple[str, ...]
    unknown_labels: tuple[str, ...]
    malformed_markers: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.unknown_labels and not self.malformed_markers

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "labels": list(self.labels),
            "unknown_labels": list(self.unknown_labels),
            "malformed_markers": list(self.malformed_markers),
            "valid": self.valid,
        }


@dataclass(frozen=True)
class Disagreement:
    """One explicit reviewer disagreement label."""

    category: str
    severity: DisagreementSeverity
    reviewers: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity.value,
            "reviewers": list(self.reviewers),
            "reason": self.reason,
        }


CANONICAL_REVIEW_POLICY: dict[str, str] = {
    "relevance": (
        "A passage is relevant only when it addresses the question or a required "
        "answer concept."
    ),
    "sufficient_evidence": (
        "Evidence is sufficient only when it can support every required "
        "substantive claim without outside facts."
    ),
    "factual_support": (
        "A claim is supported when the cited local passage explicitly states it "
        "or licenses a clearly identified valid inference."
    ),
    "completeness": (
        "An answer is complete when it covers the question's required concepts; "
        "optional detail is not required."
    ),
    "acceptable_inference": (
        "An inference must follow from supplied passages, be labelled inferred, "
        "and introduce no external premise."
    ),
    "citation_support": (
        "Each substantive claim needs at least one citation whose passage "
        "supports that claim."
    ),
    "citation_relevance": (
        "A citation must point to the passage used for the claim, not merely the "
        "correct paper."
    ),
    "abstention": (
        "Abstain when local evidence cannot support a correct answer; do not fill "
        "gaps with external knowledge."
    ),
    "derivation_support": (
        "Separate paper-explicit steps from mathematical inference and cite "
        "every paper-explicit premise."
    ),
    "external_knowledge": (
        "A fact absent from supplied passages is external and cannot enter an "
        "accepted grounded answer."
    ),
    "partial_correctness": (
        "Partial means some required content is correct and supported but at "
        "least one required part is absent or unsupported."
    ),
}

_CITATION_TOKEN = re.compile(r"(?<![A-Za-z0-9])\[?\s*[Cc](\d+)\s*\]?(?![A-Za-z0-9])")
_BRACKET_CANDIDATE = re.compile(r"\[([^\[\]]+)\]")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]*|\d+(?:\.\d+)?")
_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?%?", re.I)
_STYLE_TERMS = {
    "style",
    "wording",
    "verbose",
    "verbosity",
    "tone",
    "technical",
    "concise",
}


def canonical_policy_payload() -> dict[str, Any]:
    """Return the immutable policy vocabulary injected into every review pass."""
    return {
        "version": REVIEW_POLICY_VERSION,
        "definitions": dict(CANONICAL_REVIEW_POLICY),
    }


def autonomous_training_exclusion(question_type: str) -> str | None:
    """Return the temporary reliability exclusion reason for a question type."""
    if question_type in DEFAULT_AUTONOMOUS_TRAINING_EXCLUSIONS:
        return "question_type_pending_reliability_validation"
    return None


def normalize_citations(
    text: str, evidence: Sequence[Mapping[str, Any]]
) -> CitationNormalization:
    """Normalize all supported citation spellings through one strict parser.

    ``True`` evidence labels are the display labels (for example ``C1``). Raw
    evidence/chunk IDs are accepted as migration input only when bracketed and
    are rewritten to their current display labels.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        raise TypeError("evidence must be a sequence of mappings.")
    aliases: dict[str, str] = {}
    valid_labels: set[str] = set()
    for item in evidence:
        if not isinstance(item, Mapping):
            raise TypeError("Every evidence item must be a mapping.")
        label = item.get("label")
        if not isinstance(label, str) or not re.fullmatch(r"C[1-9]\d*", label):
            raise ValueError(
                "Every evidence item requires a canonical label such as C1."
            )
        valid_labels.add(label)
        for key in ("label", "evidence_id", "chunk_id"):
            alias = item.get(key)
            if isinstance(alias, str) and alias.strip():
                aliases[alias.strip()] = label

    malformed: list[str] = []
    unknown: list[str] = []

    def replace_bracket(match: re.Match[str]) -> str:
        raw = match.group(1).strip()
        token = re.fullmatch(r"[Cc](\d+)", raw)
        canonical = f"C{int(token.group(1))}" if token else aliases.get(raw)
        if canonical is None:
            if raw.lower().startswith("c") or raw.startswith(("ev_", "chk_")):
                unknown.append(raw)
            return match.group(0)
        if canonical not in valid_labels:
            unknown.append(canonical)
        return f"[{canonical}]"

    normalized = _BRACKET_CANDIDATE.sub(replace_bracket, text)

    def replace_token(match: re.Match[str]) -> str:
        label = f"C{int(match.group(1))}"
        if label not in valid_labels:
            unknown.append(label)
        return f"[{label}]"

    normalized = _CITATION_TOKEN.sub(replace_token, normalized)
    for bracketed in _BRACKET_CANDIDATE.findall(normalized):
        if bracketed.startswith(("C", "c", "ev_", "chk_")) and not re.fullmatch(
            r"C[1-9]\d*", bracketed
        ):
            malformed.append(bracketed)

    # Remove duplicate labels inside a contiguous citation cluster while
    # preserving first-use order and punctuation outside the cluster.
    cluster = re.compile(r"(?:\s*\[C[1-9]\d*\]){2,}")

    def deduplicate(match: re.Match[str]) -> str:
        labels = re.findall(r"C[1-9]\d*", match.group(0))
        return " " + " ".join(f"[{label}]" for label in dict.fromkeys(labels))

    normalized = cluster.sub(deduplicate, normalized).strip()
    labels = tuple(dict.fromkeys(re.findall(r"\[(C[1-9]\d*)\]", normalized)))
    return CitationNormalization(
        text=normalized,
        labels=labels,
        unknown_labels=tuple(dict.fromkeys(unknown)),
        malformed_markers=tuple(dict.fromkeys(malformed)),
    )


def segment_answer_claims(
    text: str, evidence: Sequence[Mapping[str, Any]]
) -> tuple[AnswerClaim, ...]:
    """Split answer prose into stable, citation-aware atomic claims."""
    normalized = normalize_citations(text, evidence)
    parts = [part.strip(" \t-*•") for part in _SENTENCE_BOUNDARY.split(normalized.text)]
    claims: list[AnswerClaim] = []
    for part in parts:
        if not part or not _WORD.search(re.sub(r"\[C[1-9]\d*\]", "", part)):
            continue
        labels = tuple(dict.fromkeys(re.findall(r"\[(C[1-9]\d*)\]", part)))
        plain = re.sub(r"\s*\[C[1-9]\d*\]", "", part).strip()
        lower = plain.casefold()
        if any(symbol in plain for symbol in ("=", "∑", "∂", "√")):
            claim_type = "equation"
        elif _NUMBER.search(plain):
            claim_type = "numeric"
        elif lower.startswith(("however", "although", "unless", "provided that")):
            claim_type = "qualification"
        else:
            claim_type = "factual"
        identity = content_sha256({"text": plain, "position": len(claims)})[:20]
        claims.append(
            AnswerClaim(
                claim_id=f"claim_{identity}",
                text=plain,
                claim_type=claim_type,
                citation_labels=labels,
            )
        )
    return tuple(claims)


def stable_evidence_identity(evidence: Mapping[str, Any]) -> str:
    """Derive an identity from immutable source coordinates, not display rank."""
    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a mapping.")
    citation = evidence.get("citation")
    citation = citation if isinstance(citation, Mapping) else {}
    document_id = evidence.get("document_id", citation.get("document_id"))
    chunk_id = evidence.get("chunk_id", citation.get("chunk_id"))
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("Evidence identity requires document_id.")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError("Evidence identity requires chunk_id.")
    coordinates = {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "page_start": evidence.get("page_start", citation.get("page_start")),
        "page_end": evidence.get("page_end", citation.get("page_end")),
        "start_line": evidence.get("start_line", citation.get("start_line")),
        "end_line": evidence.get("end_line", citation.get("end_line")),
        "heading_path": evidence.get("heading_path", citation.get("heading_path", [])),
        "source_hash": evidence.get("source_hash"),
    }
    return f"evidence_{content_sha256(coordinates)[:24]}"


def stamp_evidence_identities(
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy evidence and attach stable identity fields without mutating input."""
    stamped = []
    for item in evidence:
        copied = dict(item)
        copied["stable_evidence_id"] = stable_evidence_identity(copied)
        stamped.append(copied)
    return stamped


def _evidence_text(item: Mapping[str, Any]) -> str:
    for key in ("selected_text", "text", "content"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _terms(text: str) -> set[str]:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "is",
        "are",
        "this",
        "that",
        "with",
        "for",
    }
    return {token.casefold() for token in _WORD.findall(text)} - stop


def validate_claim_citations(
    answer_text: str,
    evidence: Sequence[Mapping[str, Any]],
    *,
    selected_paper_ids: Sequence[str] = (),
    expected_sections: Sequence[str] = (),
    required_concepts: Sequence[str | Sequence[str]] = (),
    expected_source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Apply strict structural and lexical claim-to-citation gates.

    The result intentionally distinguishes structural validity from support and
    relevance. Passing this function does not prove a claim semantically true.
    """
    normalized = normalize_citations(answer_text, evidence)
    claims = segment_answer_claims(normalized.text, evidence)
    by_label = {str(item.get("label")): item for item in evidence}
    selected = set(selected_paper_ids)
    expected_hashes = expected_source_hashes or {}
    critiques: list[CitationCritique] = []
    stale_ids: set[str] = set()
    source_hash_mismatches: set[str] = set()
    structural_failures: list[str] = []
    if normalized.unknown_labels:
        structural_failures.append("unknown_citation_label")
    if normalized.malformed_markers:
        structural_failures.append("malformed_citation_marker")
    for claim in claims:
        missing: list[str] = []
        passages: list[Mapping[str, Any]] = []
        if claim.required_support and not claim.citation_labels:
            missing.append("missing_citation")
        for label in claim.citation_labels:
            item = by_label.get(label)
            if item is None:
                missing.append(f"unresolved:{label}")
                continue
            passages.append(item)
            try:
                stable_evidence_identity(item)
            except (TypeError, ValueError):
                stale_ids.add(str(item.get("evidence_id", label)))
            document_id = item.get("document_id")
            if selected and document_id not in selected:
                missing.append(f"wrong_source:{document_id}")
            expected_hash = expected_hashes.get(str(document_id))
            if expected_hash is not None and item.get("source_hash") not in {
                None,
                expected_hash,
            }:
                source_hash_mismatches.add(str(document_id))
        passage_text = "\n".join(_evidence_text(item) for item in passages)
        if passages and not passage_text:
            missing.append("empty_cited_text")
        claim_numbers = set(_NUMBER.findall(claim.text))
        passage_numbers = set(_NUMBER.findall(passage_text))
        missing_numbers = sorted(claim_numbers - passage_numbers)
        if missing_numbers:
            missing.append("unsupported_numeric:" + ",".join(missing_numbers))
        if expected_sections and passages:
            actual = tuple(
                dict.fromkeys(
                    str(part)
                    for item in passages
                    for part in item.get("heading_path", [])
                    if isinstance(part, str) and part.strip()
                )
            )
            if actual and not section_topics_compatible(
                tuple(str(value) for value in expected_sections), actual
            ):
                missing.append("section_mismatch")
        if required_concepts and passages:
            passage_terms = _terms(passage_text)
            for concept in required_concepts:
                aliases = (concept,) if isinstance(concept, str) else tuple(concept)
                if aliases and not any(
                    _terms(alias) <= passage_terms for alias in aliases
                ):
                    missing.append("missing_concept:" + str(aliases[0]))
        overlap = _terms(claim.text) & _terms(passage_text)
        wrong_source = any(item.startswith("wrong_source:") for item in missing)
        if not claim.citation_labels:
            support = CitationDecision.MISSING
        elif wrong_source:
            support = CitationDecision.WRONG_SOURCE
        elif any(item.startswith("unresolved:") for item in missing):
            support = CitationDecision.MALFORMED
        elif any(item.startswith("unsupported_numeric:") for item in missing):
            support = CitationDecision.PARTIAL
        elif not passage_text:
            support = CitationDecision.IRRELEVANT
        elif missing:
            support = CitationDecision.PARTIAL
        else:
            support = CitationDecision.SUPPORTS
        relevance = (
            CitationDecision.SUPPORTS
            if passage_text and overlap and not wrong_source
            else CitationDecision.IRRELEVANT
        )
        critiques.append(
            CitationCritique(
                claim_id=claim.claim_id,
                support_status=support,
                relevance_status=relevance,
                source_match=not wrong_source,
                missing_information=tuple(missing),
                confidence=1.0,
            )
        )
    if any(
        critique.support_status
        in {
            CitationDecision.MISSING,
            CitationDecision.MALFORMED,
            CitationDecision.WRONG_SOURCE,
        }
        for critique in critiques
    ):
        structural_failures.append("claim_citation_structure_failed")
    structural_valid = (
        not structural_failures and not stale_ids and not source_hash_mismatches
    )
    support_valid = bool(claims) and all(
        item.support_status == CitationDecision.SUPPORTS for item in critiques
    )
    relevance_valid = bool(claims) and all(
        item.relevance_status == CitationDecision.SUPPORTS for item in critiques
    )
    return {
        "policy_version": REVIEW_POLICY_VERSION,
        "normalized_answer": normalized.text,
        "normalization": normalized.to_dict(),
        "claims": [item.to_dict() for item in claims],
        "claim_critiques": [item.to_dict() for item in critiques],
        "structural_valid": structural_valid,
        "support_valid": support_valid,
        "relevance_valid": relevance_valid,
        "structural_failures": list(dict.fromkeys(structural_failures)),
        "stale_evidence_ids": sorted(stale_ids),
        "source_hash_mismatches": sorted(source_hash_mismatches),
        "semantic_proof": False,
    }


def _pass_results(
    passes: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in passes:
        if not isinstance(item, Mapping):
            continue
        result = item.get("result")
        if isinstance(result, Mapping):
            grouped[str(item.get("pass_name", "unknown"))].append(result)
    return grouped


def classify_reviewer_disagreements(
    passes: Sequence[Mapping[str, Any]],
    *,
    after_repair: bool = False,
    validation: Mapping[str, Any] | None = None,
) -> tuple[Disagreement, ...]:
    """Classify reviewer conflicts with multiple explicit hard/soft labels."""
    grouped = _pass_results(passes)
    latest = {name: values[-1] for name, values in grouped.items() if values}
    disagreements: list[Disagreement] = []

    def decision(name: str) -> str | None:
        value = latest.get(name, {}).get("decision")
        return value if isinstance(value, str) else None

    def semantic_family(name: str) -> str | None:
        result = latest.get(name, {})
        outcome = result.get("policy_outcome")
        if outcome in {
            "direct",
            "supported",
            "complete",
            "supports",
            "eligible",
            "correct_abstention",
        }:
            return "pass"
        if outcome in {"partial"}:
            return "partial"
        if outcome in {
            "insufficient",
            "irrelevant",
            "external_required",
            "unsupported",
            "incomplete",
            "wrong_source",
            "wrong_span",
            "missing",
            "malformed",
            "excluded",
            "incorrect_abstention",
        }:
            return "fail"
        value = decision(name)
        if value == "accept":
            return "pass"
        if value == "reject":
            return "fail"
        if value in {"repair", "uncertain"}:
            return "partial"
        return None

    pairs = (
        ("evidence_critic", "answer_critic"),
        ("answer_critic", "citation_critic"),
        ("evidence_critic", "citation_critic"),
    )
    for left, right in pairs:
        left_value, right_value = semantic_family(left), semantic_family(right)
        if {left_value, right_value} == {"pass", "fail"}:
            category = f"{left}_vs_{right}"
            disagreements.append(
                Disagreement(
                    category=category,
                    severity=DisagreementSeverity.HARD,
                    reviewers=(left, right),
                    reason=f"{left}={left_value}; {right}={right_value}",
                )
            )
    critic_values = [
        semantic_family(name)
        for name in ("evidence_critic", "answer_critic", "citation_critic")
    ]
    critic_values = [value for value in critic_values if value]
    final = semantic_family("final_adjudicator")
    if final and critic_values:
        majority = Counter(critic_values).most_common(1)[0][0]
        if final != majority:
            disagreements.append(
                Disagreement(
                    "final_adjudicator_overrides_majority",
                    DisagreementSeverity.HARD,
                    ("final_adjudicator", "critic_majority"),
                    f"majority={majority}; adjudicator={final}",
                )
            )
    if validation is not None:
        citation = semantic_family("citation_critic")
        if validation.get("structural_valid") is False and citation == "pass":
            disagreements.append(
                Disagreement(
                    "deterministic_validator_fail_citation_critic_accept",
                    DisagreementSeverity.HARD,
                    ("deterministic_validator", "citation_critic"),
                    "Structural citation failure was accepted by the citation critic.",
                )
            )
        if validation.get("structural_valid") is True and citation == "fail":
            disagreements.append(
                Disagreement(
                    "deterministic_validator_pass_citation_critic_fail",
                    DisagreementSeverity.HARD,
                    ("deterministic_validator", "citation_critic"),
                    "Citation critic rejected a structurally valid mapping; "
                    "semantic review required.",
                )
            )
    corrections = [
        str(value).casefold()
        for result in latest.values()
        for value in result.get("required_corrections", [])
    ]
    if corrections and all(
        any(term in item for term in _STYLE_TERMS) for item in corrections
    ):
        disagreements.append(
            Disagreement(
                "answer_style_only",
                DisagreementSeverity.SOFT,
                tuple(sorted(latest)),
                "Reviewer differences concern wording/style rather than factual "
                "support.",
            )
        )
    if after_repair and disagreements:
        disagreements.append(
            Disagreement(
                "disagreement_after_repair",
                DisagreementSeverity.HARD,
                tuple(sorted(latest)),
                "A safety-relevant conflict remained after repair.",
            )
        )
    return tuple(disagreements)


def claim_level_disagreements(
    passes: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Report hard support conflicts for each explicit atomic claim."""
    grouped = _pass_results(passes)
    judgments: dict[str, dict[str, str]] = defaultdict(dict)
    for reviewer, values in grouped.items():
        if not values:
            continue
        for item in values[-1].get("claim_critiques", []):
            if not isinstance(item, Mapping):
                continue
            claim_id = item.get("claim_id")
            support = item.get("support_status")
            if isinstance(claim_id, str) and isinstance(support, str):
                family = "supported" if support == "supports" else "unsupported"
                judgments[claim_id][reviewer] = family
    results = []
    for claim_id, reviewers in sorted(judgments.items()):
        values = set(reviewers.values())
        if len(values) > 1:
            results.append(
                {
                    "claim_id": claim_id,
                    "severity": "hard",
                    "category": "claim_support_conflict",
                    "reviewers": dict(sorted(reviewers.items())),
                }
            )
    return tuple(results)


def deterministic_adjudication(
    *,
    citation_validation: Mapping[str, Any],
    evidence_decision: EvidenceDecision | None = None,
    unsupported_claims: Sequence[str] = (),
    ambiguous_question: bool = False,
    disagreements: Sequence[Disagreement] = (),
) -> dict[str, Any]:
    """Apply deterministic conflict policy before model adjudication."""
    triggers: list[str] = []
    if not citation_validation.get("structural_valid", False):
        triggers.append("citation_structural_failure")
    if not citation_validation.get("support_valid", False):
        triggers.append("unsupported_claim")
    if evidence_decision in {
        EvidenceDecision.INSUFFICIENT,
        EvidenceDecision.IRRELEVANT,
    }:
        triggers.append("insufficient_evidence")
    if unsupported_claims:
        triggers.append("unsupported_claim")
    if ambiguous_question:
        triggers.append("ambiguous_benchmark")
    if any(item.severity == DisagreementSeverity.HARD for item in disagreements):
        triggers.append("hard_reviewer_disagreement")
    triggers = list(dict.fromkeys(triggers))
    if "ambiguous_benchmark" in triggers:
        decision = "exclude"
    elif "insufficient_evidence" in triggers:
        decision = "abstain_or_reject"
    elif triggers:
        decision = "repair_or_reject"
    else:
        decision = "eligible_for_model_adjudication"
    return {
        "decision": decision,
        "triggers": triggers,
        "policy_version": REVIEW_POLICY_VERSION,
    }


def repair_diagnostics(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare pre/post-repair gates and identify whether repair helped."""
    fields = ("structural_valid", "support_valid", "relevance_valid")
    fixed = [name for name in fields if not before.get(name) and after.get(name)]
    introduced = [name for name in fields if before.get(name) and not after.get(name)]
    if fixed and not introduced:
        outcome = "fixed"
    elif introduced:
        outcome = "introduced_new_issue"
    else:
        outcome = "unchanged"
    return {"outcome": outcome, "fixed": fixed, "introduced": introduced}


def should_stop_for_reliability(
    records: Sequence[Mapping[str, Any]],
    *,
    minimum_records: int = 10,
    maximum_hard_disagreement_rate: float = 0.15,
    maximum_citation_structural_failure_rate: float = 0.05,
    maximum_unresolved_support_failure_rate: float = 0.05,
    maximum_malformed_output_rate: float = 0.02,
) -> str | None:
    """Return a conservative safety-stop reason, or ``None``."""
    if any(item.get("leakage_detected") for item in records):
        return "Paper-level split leakage was detected."
    if any(item.get("source_hash_mismatches") for item in records):
        return "A source hash mismatch was detected."
    reviewed = [item for item in records if item.get("codex_review_passes")]
    if len(reviewed) < minimum_records:
        return None
    total = len(reviewed)
    hard = sum(bool(item.get("hard_reviewer_disagreement")) for item in reviewed)
    structural = sum(
        not item.get("citation_structural_valid", False) for item in reviewed
    )
    support = sum(not item.get("citation_support_valid", False) for item in reviewed)
    malformed = sum(bool(item.get("reviewer_output_malformed")) for item in reviewed)
    if hard / total > maximum_hard_disagreement_rate:
        return "Hard reviewer disagreement exceeded the configured safety threshold."
    if structural / total > maximum_citation_structural_failure_rate:
        return "Citation structural failures exceeded the configured safety threshold."
    if support / total > maximum_unresolved_support_failure_rate:
        return (
            "Unresolved citation support failures exceeded the configured safety "
            "threshold."
        )
    if malformed / total > maximum_malformed_output_rate:
        return "Malformed reviewer outputs exceeded the configured safety threshold."
    return None


def reliability_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate reliability metrics, taxonomy, matrices, and failure examples."""
    total = len(records)
    reviewed = [item for item in records if item.get("codex_review_passes")]
    reviewed_total = len(reviewed)
    taxonomy = Counter(
        item.get("category", "unknown")
        for record in records
        for item in record.get("reviewer_disagreements", [])
    )
    hard = sum(bool(item.get("hard_reviewer_disagreement")) for item in reviewed)
    soft = sum(bool(item.get("soft_reviewer_disagreement")) for item in reviewed)
    structural = sum(bool(item.get("citation_structural_valid")) for item in reviewed)
    support = sum(bool(item.get("citation_support_valid")) for item in reviewed)
    relevance = sum(bool(item.get("citation_relevance_valid")) for item in reviewed)
    evidence = sum(bool(item.get("source_hashes")) for item in records)
    stale = sum(len(item.get("stale_evidence_ids", [])) for item in records)
    malformed = sum(bool(item.get("reviewer_output_malformed")) for item in records)
    repaired = [item for item in records if item.get("repair_attempts", 0)]
    repaired_fixed = sum(
        any(
            repair.get("outcome") == "fixed"
            for repair in item.get("repair_history", [])
        )
        for item in repaired
    )
    graph_metrics = [
        item.get("claim_alignment_metrics", {})
        for item in reviewed
        if item.get("claim_alignment_metrics")
    ]
    supported_claims = sum(
        int(item.get("supported_claim_count", 0)) for item in graph_metrics
    )
    cited_claims = supported_claims - sum(
        int(item.get("uncited_claim_count", 0)) for item in graph_metrics
    )
    claim_count = sum(int(item.get("claim_count", 0)) for item in graph_metrics)
    trace_denominator = sum(
        int(item.get("supported_claim_count", 0)) for item in graph_metrics
    )
    trace_numerator = sum(
        float(item.get("sentence_to_claim_traceability", 0.0))
        * int(item.get("supported_claim_count", 0))
        for item in graph_metrics
    )
    repair_by_type: dict[str, dict[str, int | float]] = {}
    for record in repaired:
        for repair in record.get("repair_history", []):
            outcome = str(
                repair.get("claim_repair_outcome", repair.get("outcome", "unchanged"))
            )
            for repair_type in repair.get("repair_types", ["unspecified"]):
                bucket = repair_by_type.setdefault(
                    str(repair_type),
                    {
                        "count": 0,
                        "fixed": 0,
                        "unchanged": 0,
                        "worsened": 0,
                        "introduced_new_failure": 0,
                    },
                )
                bucket["count"] = int(bucket["count"]) + 1
                normalized = (
                    "introduced_new_failure"
                    if outcome in {"introduced_new_issue", "introduced_new_failure"}
                    else outcome
                )
                if normalized in bucket:
                    bucket[normalized] = int(bucket[normalized]) + 1
    for bucket in repair_by_type.values():
        count = int(bucket["count"])
        bucket["success_rate"] = int(bucket["fixed"]) / count if count else 0.0
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in records:
        by_type[str(item.get("question_type", "unknown"))].append(item)
    type_metrics = {
        name: {
            "count": len(items),
            "hard_disagreement_rate": sum(
                bool(item.get("hard_reviewer_disagreement")) for item in items
            )
            / len(items),
            "citation_structural_validity": sum(
                bool(item.get("citation_structural_valid")) for item in items
            )
            / len(items),
            "citation_support_rate": sum(
                bool(item.get("citation_support_valid")) for item in items
            )
            / len(items),
        }
        for name, items in sorted(by_type.items())
    }
    category_aliases = {
        "architecture": "architecture",
        "metadata": "metadata",
        "motivation": "motivation",
        "method": "method",
        "intuition": "natural_tutoring",
        "prerequisites": "natural_tutoring",
        "multi_turn_follow_up": "natural_tutoring",
        "equation": "equation",
        "derivation": "derivation",
        "experiment": "experiment",
        "extraction": "reproduction",
        "result": "result",
        "reproduction": "reproduction",
        "limitation": "limitation",
        "critical_reasoning": "critique",
        "false_premise": "false_premise",
        "cross_paper_comparison": "cross_paper",
        "comparison": "cross_paper",
        "historical_impact": "historical_impact",
    }
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for name, items in by_type.items():
        by_category[category_aliases.get(name, name)].extend(items)
    category_metrics = {
        name: {
            "count": len(items),
            "citation_support_rate": sum(
                bool(item.get("citation_support_valid")) for item in items
            )
            / len(items),
            "claim_citation_completeness": sum(
                float(
                    item.get("claim_alignment_metrics", {}).get(
                        "claim_citation_completeness", 0.0
                    )
                )
                for item in items
            )
            / len(items),
            "hard_disagreement_rate": sum(
                bool(item.get("hard_reviewer_disagreement")) for item in items
            )
            / len(items),
            "autonomous_training_excluded": any(
                autonomous_training_exclusion(str(item.get("question_type", "")))
                is not None
                for item in items
            ),
        }
        for name, items in sorted(by_category.items())
    }
    matrix = Counter()
    representatives: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped = _pass_results(record.get("codex_review_passes", []))
        deterministic = "PASS" if record.get("citation_structural_valid") else "FAIL"
        citation = grouped.get("citation_critic", [{}])[-1].get("decision", "missing")
        key = (
            f"Citation validator {deterministic} / Citation critic "
            f"{str(citation).upper()}"
        )
        matrix[key] += 1
        for item in record.get("reviewer_disagreements", []):
            category = str(item.get("category", "unknown"))
            if len(representatives[category]) < 3:
                representatives[category].append(
                    {
                        "curation_record_id": record.get("curation_record_id"),
                        "question": record.get("question"),
                        "question_type": record.get("question_type"),
                        "reason": item.get("reason"),
                    }
                )
    return {
        "record_count": total,
        "reviewed_record_count": reviewed_total,
        "hard_disagreement_rate": hard / reviewed_total if reviewed_total else 0.0,
        "soft_disagreement_rate": soft / reviewed_total if reviewed_total else 0.0,
        "overall_disagreement_rate": sum(
            bool(item.get("reviewer_disagreements")) for item in reviewed
        )
        / reviewed_total
        if reviewed_total
        else 0.0,
        "citation_structural_validity": structural / reviewed_total
        if reviewed_total
        else 0.0,
        "citation_support_rate": support / reviewed_total if reviewed_total else 0.0,
        "citation_relevance_rate": relevance / reviewed_total
        if reviewed_total
        else 0.0,
        "evidence_validation_rate": evidence / total if total else 0.0,
        "repair_success_rate": repaired_fixed / len(repaired) if repaired else 0.0,
        "repair_case_count": len(repaired),
        "repair_success_by_type": dict(sorted(repair_by_type.items())),
        "claim_count": claim_count,
        "claim_citation_completeness": cited_claims / supported_claims
        if supported_claims
        else 0.0,
        "evidence_to_claim_alignment": supported_claims / claim_count
        if claim_count
        else 0.0,
        "sentence_to_claim_traceability": trace_numerator / trace_denominator
        if trace_denominator
        else 0.0,
        "claim_hard_disagreement_count": sum(
            len(item.get("claim_level_disagreements", [])) for item in reviewed
        ),
        "stale_evidence_id_count": stale,
        "malformed_reviewer_output_rate": malformed / reviewed_total
        if reviewed_total
        else 0.0,
        "leakage_count": sum(bool(item.get("leakage_detected")) for item in records),
        "source_hash_mismatch_count": sum(
            len(item.get("source_hash_mismatches", [])) for item in records
        ),
        "disagreement_taxonomy": dict(sorted(taxonomy.items())),
        "reviewer_pair_matrix": dict(sorted(matrix.items())),
        "metrics_by_question_type": type_metrics,
        "metrics_by_category": category_metrics,
        "representative_failures": dict(sorted(representatives.items())),
    }


READINESS_THRESHOLDS = {
    "citation_structural_validity": 0.98,
    "citation_support_rate": 0.95,
    "citation_relevance_rate": 0.95,
    "evidence_validation_rate": 0.95,
    "claim_citation_completeness": 0.95,
    "repair_success_rate": 0.70,
    "maximum_hard_disagreement_rate": 0.10,
    "maximum_overall_disagreement_rate": 0.15,
    "maximum_malformed_reviewer_output_rate": 0.02,
}


def full_run_readiness(report: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the documented full-run readiness policy without gaming metrics."""
    failures = []
    for name in (
        "citation_structural_validity",
        "citation_support_rate",
        "citation_relevance_rate",
        "evidence_validation_rate",
        "claim_citation_completeness",
    ):
        if float(report.get(name, 0.0)) < READINESS_THRESHOLDS[name]:
            failures.append(name)
    for metric, threshold in (
        (
            "hard_disagreement_rate",
            READINESS_THRESHOLDS["maximum_hard_disagreement_rate"],
        ),
        (
            "overall_disagreement_rate",
            READINESS_THRESHOLDS["maximum_overall_disagreement_rate"],
        ),
        (
            "malformed_reviewer_output_rate",
            READINESS_THRESHOLDS["maximum_malformed_reviewer_output_rate"],
        ),
    ):
        if float(report.get(metric, 1.0)) > threshold:
            failures.append(metric)
    for name in (
        "leakage_count",
        "source_hash_mismatch_count",
        "stale_evidence_id_count",
    ):
        if int(report.get(name, 0)) != 0:
            failures.append(name)
    if (
        report.get("record_count")
        and float(report.get("repair_success_rate", 0.0))
        < (READINESS_THRESHOLDS["repair_success_rate"])
    ):
        failures.append("repair_success_rate")
    return {
        "ready": not failures,
        "failures": failures,
        "thresholds": dict(READINESS_THRESHOLDS),
    }


def select_diagnostic_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int = 42,
    maximum_per_paper: int | None = None,
) -> list[dict[str, Any]]:
    """Select a deterministic subset balanced across types and papers."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if maximum_per_paper is not None and (
        isinstance(maximum_per_paper, bool)
        or not isinstance(maximum_per_paper, int)
        or maximum_per_paper <= 0
    ):
        raise ValueError("maximum_per_paper must be None or a positive integer.")
    if count > len(candidates):
        raise ValueError("count cannot exceed available candidates.")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for item in candidates:
        if not isinstance(item, Mapping):
            raise TypeError("Every candidate must be a mapping.")
        failure = str(item.get("original_failure", "none"))
        groups[(str(item.get("question_type", "unknown")), failure)].append(item)
    ranked_groups = sorted(
        groups,
        key=lambda key: (
            hashlib.sha256(f"{seed}:{key[0]}:{key[1]}".encode()).digest(),
            key,
        ),
    )
    for key in ranked_groups:
        groups[key].sort(
            key=lambda item: (
                hashlib.sha256(
                    (
                        f"{seed}:{item.get('question_id', item.get('question', ''))}"
                    ).encode()
                ).digest(),
                str(item.get("question_id", "")),
            )
        )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    paper_counts: Counter[str] = Counter()
    while len(selected) < count:
        progressed = False
        for key in ranked_groups:
            available = [
                item
                for item in groups[key]
                if str(item.get("question_id", item.get("question", "")))
                not in selected_ids
                and (
                    maximum_per_paper is None
                    or paper_counts[
                        str(item.get("paper_ids", ["unknown"])[0])
                        if item.get("paper_ids")
                        else "unknown"
                    ]
                    < maximum_per_paper
                )
            ]
            if not available or len(selected) >= count:
                continue
            minimum_paper_count = min(
                paper_counts[
                    str(item.get("paper_ids", ["unknown"])[0])
                    if item.get("paper_ids")
                    else "unknown"
                ]
                for item in available
            )
            chosen = next(
                item
                for item in available
                if paper_counts[
                    str(item.get("paper_ids", ["unknown"])[0])
                    if item.get("paper_ids")
                    else "unknown"
                ]
                == minimum_paper_count
            )
            identity = str(chosen.get("question_id", chosen.get("question", "")))
            paper_ids = chosen.get("paper_ids", [])
            paper = str(paper_ids[0]) if paper_ids else "unknown"
            selected.append(dict(chosen))
            selected_ids.add(identity)
            paper_counts[paper] += 1
            progressed = True
        if not progressed:
            break
    return selected


def migrate_legacy_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Enrich a 1.2.3 record in memory without mutating the source artifact."""
    migrated = dict(record)
    answer = migrated.get("answer", {})
    evidence = answer.get("evidence", []) if isinstance(answer, Mapping) else []
    selected = migrated.get("paper_ids", [])
    try:
        validation = validate_claim_citations(
            str(answer.get("answer_text", "")), evidence, selected_paper_ids=selected
        )
    except (TypeError, ValueError):
        validation = {
            "structural_valid": False,
            "support_valid": False,
            "relevance_valid": False,
            "stale_evidence_ids": [],
            "source_hash_mismatches": [],
            "structural_failures": ["legacy_record_malformed"],
            "claims": [],
            "claim_critiques": [],
        }
    disagreements = classify_reviewer_disagreements(
        migrated.get("codex_review_passes", []),
        after_repair=bool(migrated.get("repair_attempts")),
        validation=validation,
    )
    existing_graph = migrated.get("supported_claim_graph")
    if isinstance(existing_graph, Mapping):
        claim_graph = dict(existing_graph)
    else:
        try:
            graph = build_claim_graph(
                str(migrated.get("question", "Legacy question")),
                str(migrated.get("question_type", "unknown")),
                evidence,
                structured_target=(
                    migrated.get("structured_target")
                    if isinstance(migrated.get("structured_target"), Mapping)
                    else None
                ),
            )
            claim_graph = graph.to_dict()
            claim_graph["construction_mode"] = "legacy_in_memory_migration"
        except (TypeError, ValueError):
            claim_graph = {
                "policy_version": "legacy",
                "claims": [],
                "answer_plan": {"answerability": "insufficient"},
                "answer_sentences": [],
                "answer_text": str(answer.get("answer_text", "")),
                "unsupported_language": [
                    {"category": "legacy_claim_graph_migration_failed"}
                ],
                "construction_mode": "legacy_in_memory_migration_failed",
            }
    migrated.update(
        {
            "review_policy_version": REVIEW_POLICY_VERSION,
            "claim_citation_validation": validation,
            "supported_claim_graph": claim_graph,
            "claim_alignment_metrics": claim_graph_metrics(claim_graph),
            "citation_structural_valid": validation["structural_valid"],
            "citation_support_valid": validation["support_valid"],
            "citation_relevance_valid": validation["relevance_valid"],
            "stale_evidence_ids": validation.get("stale_evidence_ids", []),
            "source_hash_mismatches": validation.get("source_hash_mismatches", []),
            "reviewer_disagreements": [item.to_dict() for item in disagreements],
            "claim_level_disagreements": list(
                claim_level_disagreements(migrated.get("codex_review_passes", []))
            ),
            "hard_reviewer_disagreement": any(
                item.severity == DisagreementSeverity.HARD for item in disagreements
            ),
            "soft_reviewer_disagreement": any(
                item.severity == DisagreementSeverity.SOFT for item in disagreements
            ),
            "legacy_artifact_migrated_in_memory": True,
        }
    )
    return migrated
