"""Versioned schemas for adaptive, grounded, human-approved examples."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

INSTRUCTION_DATA_FORMAT_VERSION = "1.0"
PROVENANCE_LABELS = {
    "paper_explicit",
    "mathematical_inference",
    "external_background",
    "external_knowledge",
    "uncertain",
}
REVIEW_LABELS = {
    "correct",
    "partial",
    "incorrect",
    "should_abstain",
    "benchmark_problem",
}
REVIEW_STATUSES = {
    "proposed",
    "human_approved",
    "codex_approved",
    "human_rejected",
    "codex_rejected",
    "needs_human_review",
    "ambiguous",
    "benchmark_problem",
    "codex_curated",
    "uncertain",
    "external_source_required",
    "insufficient_evidence",
    "duplicate",
    "split_excluded",
    # Read compatibility for pre-1.2.1 local workspaces.
    "rejected",
}
DATA_SPLITS = {"train", "validation", "test"}


def _text(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    cleaned = value.strip()
    if not allow_empty and not cleaned:
        raise ValueError(f"{name} must contain non-whitespace text.")
    return cleaned


def _text_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise TypeError(f"{name} must be a sequence of non-empty strings.")
    return tuple(dict.fromkeys(item.strip() for item in value))


def _stable_id(prefix: str, payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:20]}"


@dataclass(frozen=True)
class InstructionProfile:
    """Deterministic interpretation of how an answer should be presented.

    This profile is presentation metadata. It must not influence evidence
    retrieval, paper sufficiency, or citation requirements.
    """

    desired_depth: str = "standard"
    mathematical_depth: str = "moderate"
    assumed_background: str = "undergraduate"
    explanation_style: str = "direct"
    output_format: str = "prose"
    verbosity: str = "standard"
    use_analogy: bool = False
    include_derivation: bool = False
    include_critique: bool = False
    include_comparison: bool = False
    simplify_previous: bool = False
    constraints: tuple[str, ...] = ()
    confidence: float = 0.5
    canonical_audience: str | None = None
    signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        allowed = {
            "desired_depth": {"brief", "standard", "deep"},
            "mathematical_depth": {"none", "basic", "moderate", "advanced"},
            "assumed_background": {
                "novice",
                "high_school",
                "undergraduate",
                "researcher",
            },
            "explanation_style": {"direct", "intuitive", "formal", "socratic"},
            "output_format": {"prose", "bullets", "checklist", "table", "derivation"},
            "verbosity": {"concise", "standard", "detailed"},
        }
        for name, values in allowed.items():
            if getattr(self, name) not in values:
                raise ValueError(f"{name} must be one of {sorted(values)}.")
        for name in (
            "use_analogy",
            "include_derivation",
            "include_critique",
            "include_comparison",
            "simplify_previous",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")
        if self.canonical_audience not in {
            None,
            "beginner",
            "undergraduate",
            "researcher",
        }:
            raise ValueError("canonical_audience must be optional canonical metadata.")
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise TypeError("confidence must be a finite number in [0, 1].")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1].")
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(
            self,
            "constraints",
            _text_tuple(self.constraints, "constraints") if self.constraints else (),
        )
        object.__setattr__(
            self,
            "signals",
            _text_tuple(self.signals, "signals") if self.signals else (),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "desired_depth": self.desired_depth,
            "mathematical_depth": self.mathematical_depth,
            "assumed_background": self.assumed_background,
            "explanation_style": self.explanation_style,
            "output_format": self.output_format,
            "verbosity": self.verbosity,
            "use_analogy": self.use_analogy,
            "include_derivation": self.include_derivation,
            "include_critique": self.include_critique,
            "include_comparison": self.include_comparison,
            "simplify_previous": self.simplify_previous,
            "constraints": list(self.constraints),
            "confidence": self.confidence,
            "canonical_audience": self.canonical_audience,
            "signals": list(self.signals),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> InstructionProfile:
        if not isinstance(value, dict):
            raise TypeError("InstructionProfile JSON must be an object.")
        return cls(**value)


@dataclass(frozen=True)
class ConversationTurn:
    """One immutable local conversation message."""

    role: str
    content: str
    interaction_id: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'.")
        object.__setattr__(self, "content", _text(self.content, "content"))
        if self.interaction_id is not None:
            object.__setattr__(
                self, "interaction_id", _text(self.interaction_id, "interaction_id")
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "interaction_id": self.interaction_id,
        }


@dataclass(frozen=True)
class ConversationContext:
    """Local session context; persistence is always explicit."""

    session_id: str
    selected_paper_ids: tuple[str, ...] = ()
    turns: tuple[ConversationTurn, ...] = ()
    preferences: dict[str, Any] = field(default_factory=dict)
    persist_preferences: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _text(self.session_id, "session_id"))
        object.__setattr__(
            self,
            "selected_paper_ids",
            _text_tuple(self.selected_paper_ids, "selected_paper_ids")
            if self.selected_paper_ids
            else (),
        )
        if not isinstance(self.turns, (tuple, list)) or not all(
            isinstance(turn, ConversationTurn) for turn in self.turns
        ):
            raise TypeError("turns must contain ConversationTurn instances.")
        object.__setattr__(self, "turns", tuple(self.turns))
        if not isinstance(self.preferences, dict):
            raise TypeError("preferences must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "selected_paper_ids": list(self.selected_paper_ids),
            "turns": [turn.to_dict() for turn in self.turns],
            "preferences": self.preferences if self.persist_preferences else {},
            "persist_preferences": self.persist_preferences,
        }


@dataclass(frozen=True)
class GroundedFact:
    """One target claim with an explicit provenance category."""

    text: str
    provenance: str
    citation_ids: tuple[str, ...] = ()
    confidence: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _text(self.text, "fact text"))
        if self.provenance not in PROVENANCE_LABELS:
            raise ValueError(f"provenance must be one of {sorted(PROVENANCE_LABELS)}.")
        object.__setattr__(
            self,
            "citation_ids",
            _text_tuple(self.citation_ids, "citation_ids") if self.citation_ids else (),
        )
        if self.provenance == "paper_explicit" and not self.citation_ids:
            raise ValueError("paper_explicit facts require at least one citation_id.")
        if not isinstance(self.confidence, (int, float)) or isinstance(
            self.confidence, bool
        ):
            raise TypeError("fact confidence must be numeric.")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("fact confidence must be in [0, 1].")
        object.__setattr__(self, "confidence", float(self.confidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provenance": self.provenance,
            "citation_ids": list(self.citation_ids),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class StructuredGroundedTarget:
    """Evidence-first target independent of the requested writing style."""

    facts: tuple[GroundedFact, ...] = ()
    equations: tuple[GroundedFact, ...] = ()
    derivation_steps: tuple[GroundedFact, ...] = ()
    assumptions: tuple[GroundedFact, ...] = ()
    qualifications: tuple[GroundedFact, ...] = ()
    limitations: tuple[GroundedFact, ...] = ()
    unresolved_items: tuple[str, ...] = ()
    prohibited_claims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "facts",
            "equations",
            "derivation_steps",
            "assumptions",
            "qualifications",
            "limitations",
        ):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or not all(
                isinstance(item, GroundedFact) for item in value
            ):
                raise TypeError(f"{name} must contain GroundedFact instances.")
            object.__setattr__(self, name, tuple(value))
        object.__setattr__(
            self,
            "unresolved_items",
            _text_tuple(self.unresolved_items, "unresolved_items")
            if self.unresolved_items
            else (),
        )
        object.__setattr__(
            self,
            "prohibited_claims",
            _text_tuple(self.prohibited_claims, "prohibited_claims")
            if self.prohibited_claims
            else (),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: [item.to_dict() for item in getattr(self, name)]
            for name in (
                "facts",
                "equations",
                "derivation_steps",
                "assumptions",
                "qualifications",
                "limitations",
            )
        } | {
            "unresolved_items": list(self.unresolved_items),
            "prohibited_claims": list(self.prohibited_claims),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StructuredGroundedTarget:
        if not isinstance(value, dict):
            raise TypeError("StructuredGroundedTarget JSON must be an object.")
        groups = {}
        for name in (
            "facts",
            "equations",
            "derivation_steps",
            "assumptions",
            "qualifications",
            "limitations",
        ):
            raw = value.get(name, [])
            if not isinstance(raw, list):
                raise TypeError(f"{name} must be a JSON list.")
            groups[name] = tuple(GroundedFact(**item) for item in raw)
        return cls(
            **groups,
            unresolved_items=tuple(value.get("unresolved_items", [])),
            prohibited_claims=tuple(value.get("prohibited_claims", [])),
        )


@dataclass(frozen=True)
class QuestionCandidate:
    """Untrusted question proposal awaiting human review."""

    question_id: str
    paper_ids: tuple[str, ...]
    question: str
    question_type: str
    expected_sections: tuple[str, ...] = ()
    required_concepts: tuple[str, ...] = ()
    prohibited_claims: tuple[str, ...] = ()
    review_status: str = "proposed"
    canonical_audience: str | None = None
    parent_question_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "question_id", _text(self.question_id, "question_id"))
        object.__setattr__(self, "paper_ids", _text_tuple(self.paper_ids, "paper_ids"))
        object.__setattr__(self, "question", _text(self.question, "question"))
        object.__setattr__(
            self, "question_type", _text(self.question_type, "question_type")
        )
        for name in ("expected_sections", "required_concepts", "prohibited_claims"):
            value = getattr(self, name)
            object.__setattr__(self, name, _text_tuple(value, name) if value else ())
        if self.review_status not in REVIEW_STATUSES:
            raise ValueError(f"review_status must be one of {sorted(REVIEW_STATUSES)}.")
        if self.canonical_audience not in {
            None,
            "beginner",
            "undergraduate",
            "researcher",
        }:
            raise ValueError("canonical_audience must use canonical regression labels.")
        if self.parent_question_id is not None:
            object.__setattr__(
                self,
                "parent_question_id",
                _text(self.parent_question_id, "parent_question_id"),
            )
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

    @classmethod
    def create(
        cls,
        *,
        paper_ids: tuple[str, ...],
        question: str,
        question_type: str,
        **kwargs: Any,
    ) -> QuestionCandidate:
        identity = {
            "paper_ids": sorted(paper_ids),
            "question": question.strip(),
            "question_type": question_type,
            "parent_question_id": kwargs.get("parent_question_id"),
        }
        return cls(
            question_id=_stable_id("question", identity),
            paper_ids=paper_ids,
            question=question,
            question_type=question_type,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "paper_ids": list(self.paper_ids),
            "question": self.question,
            "question_type": self.question_type,
            "expected_sections": list(self.expected_sections),
            "required_concepts": list(self.required_concepts),
            "prohibited_claims": list(self.prohibited_claims),
            "review_status": self.review_status,
            "canonical_audience": self.canonical_audience,
            "parent_question_id": self.parent_question_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> QuestionCandidate:
        if not isinstance(value, dict):
            raise TypeError("QuestionCandidate JSON must be an object.")
        return cls(**value)


@dataclass(frozen=True)
class GroundedInstructionExample:
    """One versioned instruction example; only approved records are exportable."""

    example_id: str
    paper_ids: tuple[str, ...]
    turns: tuple[ConversationTurn, ...]
    instruction_profile: InstructionProfile
    target: StructuredGroundedTarget
    final_answer: str
    evidence: tuple[dict[str, Any], ...]
    task_type: str
    review_status: str
    review_label: str
    split: str | None = None
    source_interaction_id: str | None = None
    parent_example_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_id", _text(self.example_id, "example_id"))
        object.__setattr__(self, "paper_ids", _text_tuple(self.paper_ids, "paper_ids"))
        if (
            not isinstance(self.turns, (tuple, list))
            or not self.turns
            or not all(isinstance(turn, ConversationTurn) for turn in self.turns)
        ):
            raise TypeError("turns must be a non-empty sequence of ConversationTurn.")
        object.__setattr__(self, "turns", tuple(self.turns))
        if not isinstance(self.instruction_profile, InstructionProfile):
            raise TypeError("instruction_profile must be InstructionProfile.")
        if not isinstance(self.target, StructuredGroundedTarget):
            raise TypeError("target must be StructuredGroundedTarget.")
        object.__setattr__(
            self, "final_answer", _text(self.final_answer, "final_answer")
        )
        if not isinstance(self.evidence, (tuple, list)) or not all(
            isinstance(item, dict) for item in self.evidence
        ):
            raise TypeError("evidence must be a sequence of dictionaries.")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "task_type", _text(self.task_type, "task_type"))
        if self.review_status not in REVIEW_STATUSES:
            raise ValueError(f"review_status must be one of {sorted(REVIEW_STATUSES)}.")
        if self.review_label not in REVIEW_LABELS:
            raise ValueError(f"review_label must be one of {sorted(REVIEW_LABELS)}.")
        if self.split not in DATA_SPLITS | {None}:
            raise ValueError(f"split must be one of {sorted(DATA_SPLITS)} or None.")
        if self.source_interaction_id is not None:
            object.__setattr__(
                self,
                "source_interaction_id",
                _text(self.source_interaction_id, "source_interaction_id"),
            )
        if self.parent_example_id is not None:
            object.__setattr__(
                self,
                "parent_example_id",
                _text(self.parent_example_id, "parent_example_id"),
            )
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

    @classmethod
    def create(cls, **kwargs: Any) -> GroundedInstructionExample:
        identity = {
            "paper_ids": sorted(kwargs["paper_ids"]),
            "turns": [turn.to_dict() for turn in kwargs["turns"]],
            "source_interaction_id": kwargs.get("source_interaction_id"),
            "parent_example_id": kwargs.get("parent_example_id"),
        }
        return cls(example_id=_stable_id("example", identity), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "paper_ids": list(self.paper_ids),
            "turns": [turn.to_dict() for turn in self.turns],
            "instruction_profile": self.instruction_profile.to_dict(),
            "target": self.target.to_dict(),
            "final_answer": self.final_answer,
            "evidence": list(self.evidence),
            "task_type": self.task_type,
            "review_status": self.review_status,
            "review_label": self.review_label,
            "split": self.split,
            "source_interaction_id": self.source_interaction_id,
            "parent_example_id": self.parent_example_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GroundedInstructionExample:
        if not isinstance(value, dict):
            raise TypeError("GroundedInstructionExample JSON must be an object.")
        copy = dict(value)
        copy["turns"] = tuple(ConversationTurn(**item) for item in copy["turns"])
        copy["instruction_profile"] = InstructionProfile.from_dict(
            copy["instruction_profile"]
        )
        copy["target"] = StructuredGroundedTarget.from_dict(copy["target"])
        return cls(**copy)


@dataclass(frozen=True)
class GroundedInstructionDataset:
    """Immutable dataset artifact with paper-level split assignments."""

    dataset_version: str
    examples: tuple[GroundedInstructionExample, ...]
    paper_splits: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)
    format_version: str = INSTRUCTION_DATA_FORMAT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dataset_version", _text(self.dataset_version, "dataset_version")
        )
        if not isinstance(self.examples, (tuple, list)) or not all(
            isinstance(item, GroundedInstructionExample) for item in self.examples
        ):
            raise TypeError(
                "examples must contain GroundedInstructionExample instances."
            )
        object.__setattr__(self, "examples", tuple(self.examples))
        if not isinstance(self.paper_splits, dict) or any(
            not isinstance(paper, str) or split not in DATA_SPLITS
            for paper, split in self.paper_splits.items()
        ):
            raise ValueError("paper_splits must map paper IDs to valid split names.")
        for example in self.examples:
            assigned = {
                self.paper_splits.get(paper_id) for paper_id in example.paper_ids
            }
            if None in assigned or len(assigned) != 1:
                raise ValueError(
                    f"Example {example.example_id} crosses paper splits or has "
                    "an unassigned paper."
                )
            expected = next(iter(assigned))
            if example.split != expected:
                raise ValueError(
                    f"Example {example.example_id} split does not match its papers."
                )
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "format_version": self.format_version,
            "dataset_version": self.dataset_version,
            "paper_splits": dict(sorted(self.paper_splits.items())),
            "examples": [item.to_dict() for item in self.examples],
            "metadata": self.metadata,
        }
        payload["dataset_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return payload
