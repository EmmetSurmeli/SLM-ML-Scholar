"""Resumable, evidence-first autonomous corpus-curation primitives."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from localml_scholar._version import __version__
from localml_scholar.training_data.codex_review import (
    CodexReviewPass,
    CodexReviewProvider,
    execute_review_pass,
)
from localml_scholar.training_data.provenance import content_sha256

AUTONOMOUS_TERMINAL_STATES = {
    "codex_curated",
    "rejected",
    "uncertain",
    "external_source_required",
    "insufficient_evidence",
    "duplicate",
    "split_excluded",
}
_CITATION = re.compile(r"\[(C[1-9][0-9]*)\]")


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class AutonomousCurationConfig:
    """Conservative configuration for one autonomous corpus run."""

    questions_per_paper: int = 60
    maximum_examples_per_paper: int = 40
    acceptance_threshold: float = 0.97
    evidence_threshold: float = 0.97
    maximum_repair_attempts: int = 2
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    include_multi_turn: bool = True
    include_derivations: bool = True
    include_cross_paper: bool = False
    include_abstentions: bool = True
    per_question_type_cap: int = 12
    maximum_disagreement_rate: float = 0.10

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("questions_per_paper", 40, 80),
            ("maximum_examples_per_paper", 1, 80),
            ("maximum_repair_attempts", 0, 5),
            ("per_question_type_cap", 1, 80),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if not lower <= value <= upper:
                raise ValueError(f"{name} must be in [{lower}, {upper}].")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")
        for name in (
            "acceptance_threshold",
            "evidence_threshold",
            "validation_fraction",
            "test_fraction",
            "maximum_disagreement_rate",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1].")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation_fraction + test_fraction must be below 1.")
        for name in (
            "include_multi_turn",
            "include_derivations",
            "include_cross_paper",
            "include_abstentions",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__  # type: ignore[attr-defined]
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AutonomousCurationConfig:
        if not isinstance(value, dict):
            raise TypeError("Autonomous curation config must be an object.")
        return cls(**value)


class CurationSuspended(RuntimeError):
    """A safety stop that preserves resumable state instead of lowering quality."""


EvidenceRetriever = Callable[[str, tuple[str, ...], int], list[dict[str, Any]]]
DeterministicValidator = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _citation_mappings(answer: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = answer.get("evidence", [])
    by_label = {
        item.get("label"): item
        for item in evidence
        if isinstance(item, dict) and isinstance(item.get("label"), str)
    }
    return [
        {
            "citation": label,
            "resolves": label in by_label,
            "evidence": by_label.get(label),
        }
        for label in sorted(set(_CITATION.findall(answer.get("answer_text", ""))))
    ]


def _source_hashes(answer: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        content_sha256(item)
        for item in answer.get("evidence", [])
        if isinstance(item, dict)
    )


def _all_citations_resolve(answer: dict[str, Any]) -> bool:
    mappings = _citation_mappings(answer)
    if answer.get("abstained"):
        return True
    return bool(mappings) and all(item["resolves"] for item in mappings)


def _derivation_target_is_explicit(candidate: dict[str, Any]) -> bool:
    """Require provenance-labelled steps before accepting a derivation."""
    if candidate.get("question_type") not in {"derivation", "equation"}:
        return True
    target = candidate.get("structured_target", {})
    steps = target.get("derivation_steps", []) if isinstance(target, dict) else []
    if not isinstance(steps, list) or not steps:
        return False
    allowed = {
        "paper_explicit",
        "mathematical_inference",
        "external_background",
        "external_knowledge",
        "uncertain",
    }
    for step in steps:
        if not isinstance(step, dict) or step.get("provenance") not in allowed:
            return False
        if step.get("provenance") == "paper_explicit" and not step.get("citation_ids"):
            return False
    return True


def _review_payload(
    *,
    interaction: dict[str, Any],
    candidate: dict[str, Any],
    answer: dict[str, Any],
    deterministic_review: dict[str, Any],
    critic_results: list[dict[str, Any]],
    repair_history: list[dict[str, Any]],
    required_corrections: Sequence[str] = (),
) -> dict[str, Any]:
    evidence = answer.get("evidence", [])
    return {
        "paper_metadata": interaction.get(
            "paper_metadata",
            {"paper_ids": interaction.get("paper_ids", candidate.get("paper_ids", []))},
        ),
        "question": candidate.get("question", interaction.get("question")),
        "conversation_context": interaction.get("conversation_turns", []),
        "instruction_profile": interaction.get("instruction_profile", {}),
        "answer": answer,
        "source_passages": evidence,
        "citation_mappings": _citation_mappings(answer),
        "structured_target": candidate.get(
            "structured_target",
            {
                "required_concepts": candidate.get("required_concepts", []),
                "prohibited_claims": candidate.get("prohibited_claims", []),
            },
        ),
        "deterministic_review": deterministic_review,
        "known_failure_labels": interaction.get("diagnostics", {}).get(
            "failure_categories", []
        ),
        "critic_results": critic_results,
        "repair_history": repair_history,
        "required_corrections": list(required_corrections),
    }


def _apply_answerer_result(
    answer: dict[str, Any], pass_result: CodexReviewPass
) -> dict[str, Any]:
    updated = {**answer}
    result = pass_result.result
    if result.corrected_answer:
        updated["answer_text"] = result.corrected_answer.strip()
    if result.corrected_evidence_ids:
        by_id = {
            str(item.get("evidence_id", item.get("chunk_id", item.get("label")))): item
            for item in answer.get("evidence", [])
            if isinstance(item, dict)
        }
        missing = set(result.corrected_evidence_ids) - set(by_id)
        if missing:
            raise ValueError(
                "Codex selected evidence IDs outside the supplied local evidence: "
                f"{sorted(missing)}."
            )
        selected = []
        for position, identity in enumerate(result.corrected_evidence_ids, start=1):
            item = dict(by_id[identity])
            item["label"] = f"C{position}"
            selected.append(item)
        updated["evidence"] = selected
    updated["abstained"] = result.abstention_required
    return updated


def curate_interaction(
    interaction: dict[str, Any],
    candidate: dict[str, Any],
    deterministic_review: dict[str, Any],
    *,
    provider: CodexReviewProvider,
    config: AutonomousCurationConfig | None = None,
    retrieve_evidence: EvidenceRetriever | None = None,
    revalidate: DeterministicValidator | None = None,
) -> dict[str, Any]:
    """Run separated Codex passes and a bounded evidence-first repair loop."""
    config = AutonomousCurationConfig() if config is None else config
    if not isinstance(interaction, dict) or not isinstance(candidate, dict):
        raise TypeError("interaction and candidate must be dictionaries.")
    if not isinstance(deterministic_review, dict):
        raise TypeError("deterministic_review must be a dictionary.")
    if not provider.available():
        raise CurationSuspended("Codex reviewer service is unavailable.")
    answer = dict(interaction.get("answer", {}))
    answer["evidence"] = [dict(item) for item in answer.get("evidence", [])]
    if not answer.get("evidence"):
        status = (
            "external_source_required"
            if candidate.get("question_type") == "external_context"
            else "insufficient_evidence"
        )
        return _terminal_record(
            interaction,
            candidate,
            answer,
            deterministic_review,
            status=status,
            reasons=["No local evidence was available."],
        )
    passes: list[CodexReviewPass] = []
    repairs: list[dict[str, Any]] = []
    final_confidence = 0.0
    terminal_status = "uncertain"
    terminal_reasons: list[str] = []
    maximum_cycles = config.maximum_repair_attempts + 1
    for cycle in range(maximum_cycles):
        payload = _review_payload(
            interaction=interaction,
            candidate=candidate,
            answer=answer,
            deterministic_review=deterministic_review,
            critic_results=[],
            repair_history=repairs,
            required_corrections=(
                repairs[-1].get("required_corrections", []) if repairs else ()
            ),
        )
        answerer = execute_review_pass(provider, "answerer", payload)
        passes.append(answerer)
        before_hash = content_sha256(answer)
        answer = _apply_answerer_result(answer, answerer)
        if answerer.result.corrected_target is not None:
            candidate = {
                **candidate,
                "structured_target": answerer.result.corrected_target,
            }
        if revalidate is not None:
            deterministic_review = revalidate(answer, candidate)
        critic_passes = []
        for pass_name in ("evidence_critic", "answer_critic", "citation_critic"):
            critic = execute_review_pass(
                provider,
                pass_name,
                _review_payload(
                    interaction=interaction,
                    candidate=candidate,
                    answer=answer,
                    deterministic_review=deterministic_review,
                    critic_results=[],
                    repair_history=repairs,
                ),
            )
            passes.append(critic)
            critic_passes.append(critic)
        if len(candidate.get("paper_ids", ())) > 1:
            for pass_name in ("evidence_critic", "citation_critic"):
                critic = execute_review_pass(
                    provider,
                    pass_name,
                    _review_payload(
                        interaction=interaction,
                        candidate=candidate,
                        answer=answer,
                        deterministic_review=deterministic_review,
                        critic_results=[item.to_dict() for item in critic_passes],
                        repair_history=repairs,
                    ),
                )
                passes.append(critic)
                critic_passes.append(critic)
        critic_states = [item.result.decision for item in critic_passes]
        critic_dicts = [item.to_dict() for item in critic_passes]
        adjudicator = execute_review_pass(
            provider,
            "final_adjudicator",
            _review_payload(
                interaction=interaction,
                candidate=candidate,
                answer=answer,
                deterministic_review=deterministic_review,
                critic_results=critic_dicts,
                repair_history=repairs,
            ),
        )
        passes.append(adjudicator)
        final_confidence = adjudicator.result.confidence
        all_critics_accept = all(state == "accept" for state in critic_states)
        score_floor = min(
            adjudicator.result.evidence_relevance,
            adjudicator.result.factual_support,
            adjudicator.result.citation_support,
            adjudicator.result.citation_relevance,
            adjudicator.result.answer_correctness,
        )
        deterministic_gates = [
            bool(passed)
            for reviewer in deterministic_review.get("reviewer_results", [])
            for gate, passed in reviewer.get("gates", {}).items()
            if gate != "confidence_threshold"
        ]
        deterministic_pass = all(deterministic_gates) if deterministic_gates else True
        required_score = (
            max(config.evidence_threshold, 0.99)
            if candidate.get("question_type") in {"derivation", "equation"}
            else config.evidence_threshold
        )
        accepted = (
            adjudicator.result.decision == "accept"
            and all_critics_accept
            and final_confidence >= config.acceptance_threshold
            and score_floor >= required_score
            and not adjudicator.result.unsupported_claims
            and not adjudicator.result.uncertainty_reasons
            and _all_citations_resolve(answer)
            and deterministic_pass
            and _derivation_target_is_explicit(candidate)
        )
        if accepted:
            terminal_status = "codex_curated"
            terminal_reasons = []
            break
        disagreements = len(set(critic_states + [adjudicator.result.decision])) > 1
        repair_requested = adjudicator.result.decision == "repair" or any(
            state == "repair" for state in critic_states
        )
        if disagreements and adjudicator.result.decision == "accept":
            terminal_status = "rejected"
            terminal_reasons = ["reviewer_disagreement"]
            break
        if repair_requested and cycle < config.maximum_repair_attempts:
            if retrieve_evidence is not None and any(
                item.result.evidence_relevance < config.evidence_threshold
                for item in critic_passes
            ):
                replacement = retrieve_evidence(
                    str(candidate.get("question", "")),
                    tuple(candidate.get("paper_ids", interaction.get("paper_ids", []))),
                    cycle + 1,
                )
                if replacement:
                    answer["evidence"] = [dict(item) for item in replacement]
            repairs.append(
                {
                    "attempt": cycle + 1,
                    "before_answer_hash": before_hash,
                    "after_answer_hash": content_sha256(answer),
                    "required_corrections": list(
                        adjudicator.result.required_corrections
                    ),
                    "critic_decisions": critic_states,
                    "adjudicator_decision": adjudicator.result.decision,
                    "revalidated": revalidate is not None,
                }
            )
            continue
        terminal_status = (
            "rejected"
            if adjudicator.result.decision == "reject" or repair_requested
            else "uncertain"
        )
        terminal_reasons = list(
            dict.fromkeys(
                [
                    *adjudicator.result.uncertainty_reasons,
                    *adjudicator.result.unsupported_claims,
                    *(("reviewer_disagreement",) if disagreements else ()),
                    *(("repair_limit_exhausted",) if repair_requested else ()),
                    *(
                        ("citation_validation_failed",)
                        if not _all_citations_resolve(answer)
                        else ()
                    ),
                    *(
                        ("deterministic_validation_failed",)
                        if not deterministic_pass
                        else ()
                    ),
                    *(
                        ("derivation_provenance_incomplete",)
                        if not _derivation_target_is_explicit(candidate)
                        else ()
                    ),
                ]
            )
        )
        break
    record = _terminal_record(
        interaction,
        candidate,
        answer,
        deterministic_review,
        status=terminal_status,
        reasons=terminal_reasons,
    )
    record.update(
        {
            "codex_review_passes": [item.to_dict() for item in passes],
            "repair_history": repairs,
            "repair_attempts": len(repairs),
            "final_adjudicator_confidence": final_confidence,
            "final_adjudicator": (
                passes[-1].reviewer_system if passes else "not_executed"
            ),
            "reviewer_disagreement": any(
                len(
                    {
                        item.result.decision
                        for item in passes
                        if item.pass_name
                        in {
                            "evidence_critic",
                            "answer_critic",
                            "citation_critic",
                            "final_adjudicator",
                        }
                    }
                )
                > 1
                for _ in (0,)
            ),
        }
    )
    record["record_hash"] = content_sha256(
        {key: value for key, value in record.items() if key != "record_hash"}
    )
    return record


def _terminal_record(
    interaction: dict[str, Any],
    candidate: dict[str, Any],
    answer: dict[str, Any],
    deterministic_review: dict[str, Any],
    *,
    status: str,
    reasons: list[str],
) -> dict[str, Any]:
    if status not in AUTONOMOUS_TERMINAL_STATES:
        raise ValueError(f"Unknown autonomous status: {status}")
    paper_ids = list(candidate.get("paper_ids", interaction.get("paper_ids", [])))
    question_id = candidate.get("question_id")
    identity = content_sha256(
        {
            "question_id": question_id,
            "interaction_id": interaction.get("interaction_id"),
        }
    )
    return {
        "curation_record_id": f"curated_{identity[:20]}",
        "question_id": question_id,
        "interaction_id": interaction.get("interaction_id"),
        "paper_ids": paper_ids,
        "question": candidate.get("question", interaction.get("question")),
        "question_type": candidate.get("question_type", "unknown"),
        "conversation_context": interaction.get("conversation_turns", []),
        "instruction_profile": interaction.get("instruction_profile", {}),
        "answer": answer,
        "structured_target": candidate.get(
            "structured_target",
            {
                "required_concepts": candidate.get("required_concepts", []),
                "prohibited_claims": candidate.get("prohibited_claims", []),
            },
        ),
        "deterministic_review": deterministic_review,
        "status": status,
        "trust_class": "codex_curated" if status == "codex_curated" else None,
        "terminal_reasons": reasons,
        "source_hashes": list(_source_hashes(answer)),
        "question_origin": candidate.get("metadata", {}).get(
            "source", "autonomous_generation"
        ),
        "answer_producer": "localml_scholar_grounded_answer_pipeline",
        "final_adjudicator": "not_executed",
        "package_version": __version__,
        "created_at": _timestamp(),
        "lineage_ids": [
            value
            for value in (question_id, interaction.get("interaction_id"))
            if isinstance(value, str)
        ],
        "split": None,
        "duplicate_cluster": None,
    }


def balanced_paper_splits(
    paper_ids: Sequence[str],
    *,
    seed: int = 42,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, str]:
    """Assign deterministic paper-level splits with sensible small-corpus counts."""
    if not isinstance(paper_ids, Sequence) or isinstance(paper_ids, (str, bytes)):
        raise TypeError("paper_ids must be a sequence.")
    unique = sorted(set(paper_ids))
    if not unique or not all(isinstance(item, str) and item.strip() for item in unique):
        raise ValueError("paper_ids must contain non-empty strings.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below 1.")
    ranked = sorted(
        unique,
        key=lambda item: (
            hashlib.sha256(f"{seed}:{item}".encode()).digest(),
            item,
        ),
    )
    count = len(ranked)
    test_count = 0 if count < 2 else max(1, round(count * test_fraction))
    validation_count = 0 if count < 3 else max(1, round(count * validation_fraction))
    while test_count + validation_count >= count:
        if validation_count:
            validation_count -= 1
        elif test_count:
            test_count -= 1
    result = {}
    for position, paper_id in enumerate(ranked):
        if position < test_count:
            split = "test"
        elif position < test_count + validation_count:
            split = "validation"
        else:
            split = "train"
        result[paper_id] = split
    return dict(sorted(result.items()))


def autonomous_quality_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize actual autonomous run outcomes without capability claims."""
    if not isinstance(records, list) or not all(
        isinstance(item, dict) for item in records
    ):
        raise TypeError("records must be a list of dictionaries.")
    statuses = Counter(item.get("status", "unknown") for item in records)
    accepted = [item for item in records if item.get("status") == "codex_curated"]
    confidence = [
        float(item["final_adjudicator_confidence"])
        for item in records
        if isinstance(item.get("final_adjudicator_confidence"), (int, float))
    ]
    types = Counter(item.get("question_type", "unknown") for item in accepted)
    reasons = Counter(
        reason for item in records for reason in item.get("terminal_reasons", [])
    )
    repair_reasons = Counter(
        reason
        for item in records
        for repair in item.get("repair_history", [])
        for reason in repair.get("required_corrections", [])
    )
    high_risk = Counter(
        category
        for item in records
        for category in item.get("deterministic_review", {}).get(
            "mandatory_human_categories", []
        )
    )
    repaired = [item for item in records if item.get("repair_attempts", 0) > 0]
    citation_passes = sum(
        _all_citations_resolve(item.get("answer", {})) for item in records
    )
    evidence_passes = sum(bool(item.get("source_hashes")) for item in records)
    split_counts = Counter(item.get("split") for item in accepted if item.get("split"))
    strongest = max(
        accepted,
        key=lambda item: item.get("final_adjudicator_confidence", 0),
        default=None,
    )
    weakest = min(
        accepted,
        key=lambda item: item.get("final_adjudicator_confidence", 0),
        default=None,
    )
    total = len(records)
    profiles = Counter(
        content_sha256(item.get("instruction_profile", {})) for item in accepted
    )
    duplicate_count = statuses["duplicate"]
    confidence_distribution = {
        "below_0_80": sum(value < 0.80 for value in confidence),
        "0_80_to_0_90": sum(0.80 <= value < 0.90 for value in confidence),
        "0_90_to_0_97": sum(0.90 <= value < 0.97 for value in confidence),
        "0_97_to_1_00": sum(value >= 0.97 for value in confidence),
    }
    return {
        "questions_generated": total,
        "answers_produced": sum(bool(item.get("answer")) for item in records),
        "reviews_completed": sum(
            bool(item.get("codex_review_passes")) for item in records
        ),
        "examples_accepted": len(accepted),
        "examples_repaired": len(repaired),
        "examples_rejected": statuses["rejected"],
        "examples_uncertain": statuses["uncertain"],
        "insufficient_evidence": statuses["insufficient_evidence"],
        "duplicates_removed": statuses["duplicate"],
        "duplicate_removal_rate": duplicate_count / total if total else 0.0,
        "acceptance_rate": len(accepted) / total if total else 0.0,
        "evidence_validation_rate": evidence_passes / total if total else 0.0,
        "citation_validation_rate": citation_passes / total if total else 0.0,
        "reviewer_agreement": 1
        - sum(bool(item.get("reviewer_disagreement")) for item in records) / total
        if total
        else 0.0,
        "average_adjudicator_confidence": sum(confidence) / len(confidence)
        if confidence
        else None,
        "confidence_distribution": confidence_distribution,
        "question_type_diversity": dict(sorted(types.items())),
        "instruction_profile_diversity": len(profiles),
        "high_risk_category_counts": dict(sorted(high_risk.items())),
        "derivation_count": sum(
            item.get("question_type") in {"derivation", "equation"} for item in accepted
        ),
        "multi_turn_count": sum(
            bool(item.get("conversation_context")) for item in accepted
        ),
        "cross_paper_count": sum(
            len(item.get("paper_ids", [])) > 1 for item in accepted
        ),
        "abstention_count": sum(
            bool(item.get("answer", {}).get("abstained")) for item in accepted
        ),
        "common_rejection_reasons": dict(reasons.most_common(10)),
        "common_repair_reasons": dict(repair_reasons.most_common(10)),
        "split_counts": dict(sorted(split_counts.items())),
        "strongest_accepted_example": strongest,
        "lowest_confidence_accepted_example": weakest,
        "status_counts": dict(sorted(statuses.items())),
    }
