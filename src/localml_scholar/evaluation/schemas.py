"""Validated immutable schemas for real-paper benchmarks and evaluation runs."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from localml_scholar._version import __version__
from localml_scholar.retrieval import RetrievalIndex
from localml_scholar.retrieval.documents import canonical_json, stable_identifier

EVALUATION_FORMAT_VERSION = 1
AUDIENCE_LEVELS = frozenset({"beginner", "undergraduate", "researcher"})
QUESTION_TYPES = frozenset(
    {
        "metadata",
        "motivation",
        "main_method",
        "architecture",
        "equation",
        "notation",
        "assumption",
        "methodology",
        "hyperparameter",
        "experiment",
        "result",
        "ablation",
        "limitation",
        "reproduction",
        "comparison",
        "historical_impact",
        "external_context_required",
        "false_premise",
        "insufficient_evidence",
        "synthesis",
        "interpretation",
    }
)
ANSWERABILITY = frozenset(
    {
        "paper_answerable",
        "external_sources_required",
        "unanswerable",
        "ambiguous",
    }
)
PAPER_SUFFICIENCY = frozenset(
    {"sufficient", "partially_sufficient", "insufficient", "external_required"}
)
REVIEW_STATUSES = frozenset({"proposed", "approved", "edited", "rejected"})
REVIEWER_LABELS = frozenset(
    {
        "correct",
        "partially_correct",
        "incorrect",
        "should_abstain",
        "benchmark_problem",
    }
)


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _optional_nonempty(value: object, name: str) -> str | None:
    return None if value is None else _nonempty(value, name)


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{name} must be a tuple of non-empty strings.")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must not contain duplicates.")
    return value


def _fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1].")
    return normalized


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


@dataclass(frozen=True)
class ConceptGroup:
    """One manually specified concept with accepted lexical aliases."""

    concept: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.concept, "concept")
        _string_tuple(self.aliases, "aliases")
        normalized = {" ".join(item.casefold().split()) for item in self.aliases}
        if " ".join(self.concept.casefold().split()) in normalized:
            raise ValueError("aliases must not repeat the canonical concept.")

    def to_dict(self) -> dict[str, Any]:
        return {"concept": self.concept, "aliases": list(self.aliases)}

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> ConceptGroup:
        if not isinstance(state, Mapping) or set(state) != {"concept", "aliases"}:
            raise ValueError("Concept group state is malformed.")
        aliases = state["aliases"]
        if not isinstance(aliases, list):
            raise ValueError("Serialized concept aliases must be a list.")
        return cls(concept=state["concept"], aliases=tuple(aliases))


@dataclass(frozen=True)
class GoldEvidence:
    """Gold or acceptable evidence bound to an exact indexed chunk/range."""

    chunk_id: str
    section_id: str | None = None
    start_character: int | None = None
    end_character: int | None = None
    relevance_grade: int = 2

    def __post_init__(self) -> None:
        _nonempty(self.chunk_id, "chunk_id")
        _optional_nonempty(self.section_id, "section_id")
        if (self.start_character is None) != (self.end_character is None):
            raise ValueError("Gold evidence range must be fully present or absent.")
        if self.start_character is not None and (
            isinstance(self.start_character, bool)
            or not isinstance(self.start_character, int)
            or isinstance(self.end_character, bool)
            or not isinstance(self.end_character, int)
            or not 0 <= self.start_character < self.end_character
        ):
            raise ValueError("Gold evidence range must be ordered and non-empty.")
        if (
            isinstance(self.relevance_grade, bool)
            or not isinstance(self.relevance_grade, int)
            or self.relevance_grade not in {1, 2, 3}
        ):
            raise ValueError("relevance_grade must be 1, 2, or 3.")

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> GoldEvidence:
        if not isinstance(state, Mapping) or set(state) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("Gold evidence state is malformed.")
        return cls(**state)


@dataclass(frozen=True)
class BenchmarkQuestion:
    """One proposed or human-reviewed paper benchmark question."""

    question_id: str
    paper_id: str
    question: str
    question_type: str
    audience_level: str
    answerability: str
    paper_sufficiency: str
    expected_sections: tuple[str, ...] = ()
    forbidden_sections: tuple[str, ...] = ()
    gold_evidence: tuple[GoldEvidence, ...] = ()
    acceptable_chunk_ids: tuple[str, ...] = ()
    required_concepts: tuple[ConceptGroup, ...] = ()
    optional_concepts: tuple[ConceptGroup, ...] = ()
    prohibited_claims: tuple[str, ...] = ()
    expected_numbers: tuple[str, ...] = ()
    expected_identifiers: tuple[str, ...] = ()
    acceptable_abstention_reasons: tuple[str, ...] = ()
    completeness_requirements: tuple[str, ...] = ()
    gold_core_answer: str | None = None
    gold_notes: str | None = None
    review_status: str = "proposed"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("question_id", "paper_id", "question"):
            _nonempty(getattr(self, name), name)
        if self.question_type not in QUESTION_TYPES:
            raise ValueError(f"question_type must be one of {sorted(QUESTION_TYPES)}.")
        if self.audience_level not in AUDIENCE_LEVELS:
            raise ValueError(
                f"audience_level must be one of {sorted(AUDIENCE_LEVELS)}."
            )
        if self.answerability not in ANSWERABILITY:
            raise ValueError(f"answerability must be one of {sorted(ANSWERABILITY)}.")
        if self.paper_sufficiency not in PAPER_SUFFICIENCY:
            raise ValueError(
                f"paper_sufficiency must be one of {sorted(PAPER_SUFFICIENCY)}."
            )
        compatible = {
            "paper_answerable": {"sufficient", "partially_sufficient"},
            "external_sources_required": {"external_required"},
            "unanswerable": {"insufficient"},
            "ambiguous": {"partially_sufficient", "insufficient"},
        }
        if self.paper_sufficiency not in compatible[self.answerability]:
            raise ValueError(
                "answerability and paper_sufficiency labels are contradictory."
            )
        for name in (
            "expected_sections",
            "forbidden_sections",
            "acceptable_chunk_ids",
            "prohibited_claims",
            "expected_numbers",
            "expected_identifiers",
            "acceptable_abstention_reasons",
            "completeness_requirements",
        ):
            _string_tuple(getattr(self, name), name)
        if set(self.expected_sections) & set(self.forbidden_sections):
            raise ValueError("Expected and forbidden sections must not overlap.")
        if not isinstance(self.gold_evidence, tuple) or not all(
            isinstance(item, GoldEvidence) for item in self.gold_evidence
        ):
            raise TypeError("gold_evidence must contain GoldEvidence objects.")
        gold_ids = [item.chunk_id for item in self.gold_evidence]
        if len(gold_ids) != len(set(gold_ids)):
            raise ValueError("gold_evidence chunk IDs must be unique.")
        if set(gold_ids) & set(self.acceptable_chunk_ids):
            raise ValueError("Gold and alternate chunk IDs must not overlap.")
        for name in ("required_concepts", "optional_concepts"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not all(
                isinstance(item, ConceptGroup) for item in values
            ):
                raise TypeError(f"{name} must contain ConceptGroup objects.")
            concepts = [item.concept.casefold() for item in values]
            if len(concepts) != len(set(concepts)):
                raise ValueError(f"{name} concepts must be unique.")
        required = {item.concept.casefold() for item in self.required_concepts}
        optional = {item.concept.casefold() for item in self.optional_concepts}
        if required & optional:
            raise ValueError("Required and optional concepts must not overlap.")
        _optional_nonempty(self.gold_core_answer, "gold_core_answer")
        _optional_nonempty(self.gold_notes, "gold_notes")
        if self.review_status not in REVIEW_STATUSES:
            raise ValueError(f"review_status must be one of {sorted(REVIEW_STATUSES)}.")
        if self.review_status in {"approved", "edited"}:
            if self.answerability == "paper_answerable" and not (
                self.gold_evidence or self.acceptable_chunk_ids
            ):
                raise ValueError(
                    "Approved paper-answerable questions need gold evidence."
                )
            if not self.gold_notes:
                raise ValueError("Approved or edited questions require gold_notes.")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")
        canonical_json(self.metadata)

    @property
    def relevant_chunk_ids(self) -> tuple[str, ...]:
        """Return primary and acceptable evidence IDs in deterministic order."""
        return tuple(item.chunk_id for item in self.gold_evidence) + (
            self.acceptable_chunk_ids
        )

    @classmethod
    def create(
        cls,
        *,
        paper_id: str,
        question: str,
        question_type: str,
        audience_level: str,
        answerability: str,
        paper_sufficiency: str,
        **kwargs: Any,
    ) -> BenchmarkQuestion:
        """Create a deterministic identity from stable question semantics."""
        question_id = stable_identifier(
            "bq",
            paper_id,
            " ".join(question.split()),
            question_type,
            audience_level,
        )
        return cls(
            question_id=question_id,
            paper_id=paper_id,
            question=question,
            question_type=question_type,
            audience_level=audience_level,
            answerability=answerability,
            paper_sufficiency=paper_sufficiency,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        state = dict(vars(self))
        for name in (
            "expected_sections",
            "forbidden_sections",
            "acceptable_chunk_ids",
            "prohibited_claims",
            "expected_numbers",
            "expected_identifiers",
            "acceptable_abstention_reasons",
            "completeness_requirements",
        ):
            state[name] = list(state[name])
        state["gold_evidence"] = [item.to_dict() for item in self.gold_evidence]
        state["required_concepts"] = [item.to_dict() for item in self.required_concepts]
        state["optional_concepts"] = [item.to_dict() for item in self.optional_concepts]
        return state

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> BenchmarkQuestion:
        if not isinstance(state, Mapping) or set(state) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("Benchmark question state is malformed.")
        values = dict(state)
        for name in (
            "expected_sections",
            "forbidden_sections",
            "acceptable_chunk_ids",
            "prohibited_claims",
            "expected_numbers",
            "expected_identifiers",
            "acceptable_abstention_reasons",
            "completeness_requirements",
        ):
            if not isinstance(values[name], list):
                raise ValueError(f"Serialized {name} must be a list.")
            values[name] = tuple(values[name])
        for name, item_type in (
            ("gold_evidence", GoldEvidence),
            ("required_concepts", ConceptGroup),
            ("optional_concepts", ConceptGroup),
        ):
            if not isinstance(values[name], list):
                raise ValueError(f"Serialized {name} must be a list.")
            values[name] = tuple(item_type.from_dict(item) for item in values[name])
        return cls(**values)


@dataclass(frozen=True)
class Benchmark:
    """Versioned benchmark tied to exact immutable source and index identities."""

    name: str
    benchmark_version: str
    index_sha256: str
    document_hashes: dict[str, str]
    questions: tuple[BenchmarkQuestion, ...]
    package_version: str = __version__
    format_version: int = EVALUATION_FORMAT_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.name, "name")
        _nonempty(self.benchmark_version, "benchmark_version")
        _sha256(self.index_sha256, "index_sha256")
        if not isinstance(self.document_hashes, dict) or not self.document_hashes:
            raise ValueError("document_hashes must be a non-empty dictionary.")
        for document_id, digest in self.document_hashes.items():
            _nonempty(document_id, "document_id")
            _sha256(digest, "document hash")
        if (
            not isinstance(self.questions, tuple)
            or not self.questions
            or not all(isinstance(item, BenchmarkQuestion) for item in self.questions)
        ):
            raise ValueError("questions must contain BenchmarkQuestion objects.")
        identifiers = [item.question_id for item in self.questions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Benchmark question IDs must be unique.")
        if any(item.paper_id not in self.document_hashes for item in self.questions):
            raise ValueError("Every benchmark question paper_id needs a source hash.")
        _nonempty(self.package_version, "package_version")
        if self.format_version != EVALUATION_FORMAT_VERSION:
            raise ValueError("Benchmark format_version is unsupported.")

    @property
    def benchmark_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json(self._identity_state()).encode("utf-8")
        ).hexdigest()

    @property
    def approved_questions(self) -> tuple[BenchmarkQuestion, ...]:
        """Return only trusted human-approved or human-edited questions."""
        return tuple(
            item
            for item in self.questions
            if item.review_status in {"approved", "edited"}
        )

    def _identity_state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "benchmark_version": self.benchmark_version,
            "index_sha256": self.index_sha256,
            "document_hashes": dict(sorted(self.document_hashes.items())),
            "questions": [item.to_dict() for item in self.questions],
            "package_version": self.package_version,
            "format_version": self.format_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_state(), "benchmark_sha256": self.benchmark_sha256}

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> Benchmark:
        expected = set(cls.__dataclass_fields__) | {"benchmark_sha256"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Benchmark state is malformed.")
        values = dict(state)
        digest = values.pop("benchmark_sha256")
        questions = values["questions"]
        if not isinstance(questions, list):
            raise ValueError("Serialized benchmark questions must be a list.")
        values["questions"] = tuple(
            BenchmarkQuestion.from_dict(item) for item in questions
        )
        benchmark = cls(**values)
        if digest != benchmark.benchmark_sha256:
            raise ValueError("Benchmark hash is inconsistent.")
        return benchmark

    def validate_against_index(self, index: RetrievalIndex) -> None:
        """Reject stale index, source, chunk, section, or evidence ranges."""
        if not isinstance(index, RetrievalIndex):
            raise TypeError("index must be a RetrievalIndex.")
        if self.index_sha256 != index.index_sha256:
            raise ValueError("Benchmark index hash is stale.")
        documents = {item.document_id: item for item in index.documents}
        chunks = {item.chunk_id: item for item in index.chunks}
        for document_id, digest in self.document_hashes.items():
            document = documents.get(document_id)
            if document is None or document.content_sha256 != digest:
                raise ValueError("Benchmark source document hash is stale.")
        for question in self.questions:
            document = documents[question.paper_id]
            sections = {item.section_id for item in document.sections}
            for evidence in question.gold_evidence:
                chunk = chunks.get(evidence.chunk_id)
                if chunk is None or chunk.document_id != question.paper_id:
                    raise ValueError("Benchmark contains a stale gold chunk ID.")
                if (
                    evidence.section_id is not None
                    and evidence.section_id not in sections
                ):
                    raise ValueError("Benchmark contains a stale gold section ID.")
                if evidence.start_character is not None and not (
                    chunk.start_character
                    <= evidence.start_character
                    < evidence.end_character
                    <= chunk.end_character
                ):
                    raise ValueError("Benchmark gold range lies outside its chunk.")
            for chunk_id in question.acceptable_chunk_ids:
                chunk = chunks.get(chunk_id)
                if chunk is None or chunk.document_id != question.paper_id:
                    raise ValueError("Benchmark contains a stale alternate chunk ID.")


@dataclass(frozen=True)
class CitedAnswerPoint:
    """One audience-neutral factual statement with exact answer-local citations."""

    text: str
    citations: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.text, "text")
        _string_tuple(self.citations, "citations")
        if not self.citations:
            raise ValueError("Every substantive answer point requires a citation.")

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "citations": list(self.citations)}

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> CitedAnswerPoint:
        if not isinstance(state, Mapping) or set(state) != {"text", "citations"}:
            raise ValueError("Cited answer point state is malformed.")
        citations = state["citations"]
        if not isinstance(citations, list):
            raise ValueError("Serialized answer-point citations must be a list.")
        return cls(text=state["text"], citations=tuple(citations))


@dataclass(frozen=True)
class StructuredAnswerTarget:
    """One cited factual basis rendered at multiple audience depths."""

    core_answer: CitedAnswerPoint
    supporting_points: tuple[CitedAnswerPoint, ...] = ()
    equations: tuple[CitedAnswerPoint, ...] = ()
    assumptions: tuple[CitedAnswerPoint, ...] = ()
    qualifications: tuple[CitedAnswerPoint, ...] = ()
    limitations: tuple[CitedAnswerPoint, ...] = ()
    unresolved_items: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.core_answer, CitedAnswerPoint):
            raise TypeError("core_answer must be CitedAnswerPoint.")
        for name in (
            "supporting_points",
            "equations",
            "assumptions",
            "qualifications",
            "limitations",
        ):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not all(
                isinstance(item, CitedAnswerPoint) for item in value
            ):
                raise TypeError(f"{name} must contain CitedAnswerPoint objects.")
        _string_tuple(self.unresolved_items, "unresolved_items")

    @property
    def all_points(self) -> tuple[CitedAnswerPoint, ...]:
        return (
            (self.core_answer,)
            + self.supporting_points
            + self.equations
            + self.assumptions
            + self.qualifications
            + self.limitations
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_answer": self.core_answer.to_dict(),
            "supporting_points": [item.to_dict() for item in self.supporting_points],
            "equations": [item.to_dict() for item in self.equations],
            "assumptions": [item.to_dict() for item in self.assumptions],
            "qualifications": [item.to_dict() for item in self.qualifications],
            "limitations": [item.to_dict() for item in self.limitations],
            "unresolved_items": list(self.unresolved_items),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> StructuredAnswerTarget:
        if not isinstance(state, Mapping) or set(state) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("Structured answer target state is malformed.")
        values = dict(state)
        values["core_answer"] = CitedAnswerPoint.from_dict(values["core_answer"])
        for name in (
            "supporting_points",
            "equations",
            "assumptions",
            "qualifications",
            "limitations",
        ):
            if not isinstance(values[name], list):
                raise ValueError(f"Serialized {name} must be a list.")
            values[name] = tuple(
                CitedAnswerPoint.from_dict(item) for item in values[name]
            )
        if not isinstance(values["unresolved_items"], list):
            raise ValueError("Serialized unresolved_items must be a list.")
        values["unresolved_items"] = tuple(values["unresolved_items"])
        return cls(**values)


@dataclass(frozen=True)
class EvaluationConfig:
    """Versioned configuration that makes evaluation runs comparable."""

    mode: str = "extractive"
    retrieval_method: str = "bm25"
    top_k: int = 5
    audience_renderer: str = "deterministic_v1"
    model_checkpoint_sha256: str | None = None
    tokenizer_sha256: str | None = None
    random_seed: int = 0
    retrieval_parameters: dict[str, Any] = field(default_factory=dict)
    evidence_selection_settings: dict[str, Any] = field(default_factory=dict)
    sufficiency_settings: dict[str, Any] = field(default_factory=dict)
    acceptance_policy: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {
            "retrieval_only",
            "top_passage",
            "extractive",
            "generative",
            "generative_with_extractive_fallback",
        }:
            raise ValueError("Unsupported evaluation mode.")
        if self.retrieval_method not in {
            "bm25",
            "tfidf",
            "semantic",
            "hybrid",
            "hybrid_reranked",
        }:
            raise ValueError("Unsupported retrieval_method.")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise TypeError("top_k must be an integer.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        _nonempty(self.audience_renderer, "audience_renderer")
        for name in ("model_checkpoint_sha256", "tokenizer_sha256"):
            value = getattr(self, name)
            if value is not None:
                _sha256(value, name)
        if self.mode in {"generative", "generative_with_extractive_fallback"} and (
            self.model_checkpoint_sha256 is None or self.tokenizer_sha256 is None
        ):
            raise ValueError(
                "Generative evaluation requires checkpoint and tokenizer identities."
            )
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an integer.")
        for name in (
            "retrieval_parameters",
            "evidence_selection_settings",
            "sufficiency_settings",
            "acceptance_policy",
        ):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise TypeError(f"{name} must be a dictionary.")
            canonical_json(value)

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> EvaluationConfig:
        if not isinstance(state, Mapping) or set(state) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("Evaluation configuration state is malformed.")
        return cls(**state)


@dataclass(frozen=True)
class RetrievalGrade:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    reciprocal_rank: float
    hit_rate_at_k: float
    ndcg_at_k: float
    expected_section_hit: float
    title_page_hit: float
    motivation_source_hit: float
    forbidden_section_rate: float
    boilerplate_rate: float
    evidence_redundancy: float
    irrelevant_positive_score_rate: float
    retrieved_chunk_ids: tuple[str, ...]
    missed_gold_chunk_ids: tuple[str, ...]
    wrong_section_chunk_ids: tuple[str, ...]
    boilerplate_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "reciprocal_rank",
            "hit_rate_at_k",
            "ndcg_at_k",
            "expected_section_hit",
            "title_page_hit",
            "motivation_source_hit",
            "forbidden_section_rate",
            "boilerplate_rate",
            "evidence_redundancy",
            "irrelevant_positive_score_rate",
        ):
            object.__setattr__(self, name, _fraction(getattr(self, name), name))
        for name in (
            "retrieved_chunk_ids",
            "missed_gold_chunk_ids",
            "wrong_section_chunk_ids",
            "boilerplate_chunk_ids",
        ):
            _string_tuple(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        state = dict(vars(self))
        for name in (
            "retrieved_chunk_ids",
            "missed_gold_chunk_ids",
            "wrong_section_chunk_ids",
            "boilerplate_chunk_ids",
        ):
            state[name] = list(state[name])
        return state

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> RetrievalGrade:
        values = _validated_dataclass_state(cls, state, "Retrieval grade")
        for name in (
            "retrieved_chunk_ids",
            "missed_gold_chunk_ids",
            "wrong_section_chunk_ids",
            "boilerplate_chunk_ids",
        ):
            if not isinstance(values[name], list):
                raise ValueError(f"Serialized {name} must be a list.")
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class SufficiencyGrade:
    gold_label: str
    predicted_label: str
    correct: float
    false_answer: float
    false_abstention: float
    external_context_recognized: float
    correctly_qualified: float

    def __post_init__(self) -> None:
        if self.gold_label not in PAPER_SUFFICIENCY:
            raise ValueError("Unknown gold sufficiency label.")
        if self.predicted_label not in {
            "sufficient",
            "partially_sufficient",
            "insufficient",
            "external_required",
        }:
            raise ValueError("Unknown predicted sufficiency label.")
        for name in (
            "correct",
            "false_answer",
            "false_abstention",
            "external_context_recognized",
            "correctly_qualified",
        ):
            object.__setattr__(self, name, _fraction(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> SufficiencyGrade:
        return cls(**_validated_dataclass_state(cls, state, "Sufficiency grade"))


@dataclass(frozen=True)
class ConceptCoverage:
    required_recall: float
    optional_recall: float
    present_required: tuple[str, ...]
    missing_required: tuple[str, ...]
    present_optional: tuple[str, ...]
    uncited_required: tuple[str, ...]
    unsupported_required: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("required_recall", "optional_recall"):
            object.__setattr__(self, name, _fraction(getattr(self, name), name))
        for name in (
            "present_required",
            "missing_required",
            "present_optional",
            "uncited_required",
            "unsupported_required",
        ):
            _string_tuple(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        state = dict(vars(self))
        for name in (
            "present_required",
            "missing_required",
            "present_optional",
            "uncited_required",
            "unsupported_required",
        ):
            state[name] = list(state[name])
        return state

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> ConceptCoverage:
        values = _validated_dataclass_state(cls, state, "Concept coverage")
        for name in (
            "present_required",
            "missing_required",
            "present_optional",
            "uncited_required",
            "unsupported_required",
        ):
            if not isinstance(values[name], list):
                raise ValueError(f"Serialized {name} must be a list.")
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class CitationGrade:
    syntax_valid: float
    existence_valid: float
    source_location_correct: float
    support_rate: float
    relevance_rate: float
    precision: float
    recall: float
    coverage: float
    wrong_section_count: int
    boilerplate_count: int

    def __post_init__(self) -> None:
        for name in (
            "syntax_valid",
            "existence_valid",
            "source_location_correct",
            "support_rate",
            "relevance_rate",
            "precision",
            "recall",
            "coverage",
        ):
            object.__setattr__(self, name, _fraction(getattr(self, name), name))
        for name in ("wrong_section_count", "boilerplate_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> CitationGrade:
        return cls(**_validated_dataclass_state(cls, state, "Citation grade"))


@dataclass(frozen=True)
class AudienceGrade:
    audience_level: str
    appropriateness: float
    jargon_density: float
    average_sentence_words: float
    equation_count: int
    definition_present: float
    mechanism_present: float
    qualification_present: float
    limitation_present: float
    reasons: tuple[str, ...]
    requires_human_review: bool

    def __post_init__(self) -> None:
        if self.audience_level not in AUDIENCE_LEVELS:
            raise ValueError("Unknown audience level.")
        for name in (
            "appropriateness",
            "jargon_density",
            "definition_present",
            "mechanism_present",
            "qualification_present",
            "limitation_present",
        ):
            object.__setattr__(self, name, _fraction(getattr(self, name), name))
        if (
            isinstance(self.average_sentence_words, bool)
            or not isinstance(self.average_sentence_words, (int, float))
            or not math.isfinite(float(self.average_sentence_words))
            or self.average_sentence_words < 0
        ):
            raise ValueError("average_sentence_words must be finite and non-negative.")
        if (
            isinstance(self.equation_count, bool)
            or not isinstance(self.equation_count, int)
            or self.equation_count < 0
        ):
            raise ValueError("equation_count must be a non-negative integer.")
        _string_tuple(self.reasons, "reasons")
        if not isinstance(self.requires_human_review, bool):
            raise TypeError("requires_human_review must be boolean.")

    def to_dict(self) -> dict[str, Any]:
        state = dict(vars(self))
        state["reasons"] = list(self.reasons)
        return state

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> AudienceGrade:
        values = _validated_dataclass_state(cls, state, "Audience grade")
        if not isinstance(values["reasons"], list):
            raise ValueError("Serialized audience reasons must be a list.")
        values["reasons"] = tuple(values["reasons"])
        return cls(**values)


@dataclass(frozen=True)
class AnswerGrade:
    relevance: float
    completeness: float
    answerability_correct: float
    numerical_accuracy: float
    prohibited_claim_count: int
    false_premise_accepted: float
    wrong_entity_type: float
    missing_requirements: tuple[str, ...]
    relevance_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "relevance",
            "completeness",
            "answerability_correct",
            "numerical_accuracy",
            "false_premise_accepted",
            "wrong_entity_type",
        ):
            object.__setattr__(self, name, _fraction(getattr(self, name), name))
        if (
            isinstance(self.prohibited_claim_count, bool)
            or not isinstance(self.prohibited_claim_count, int)
            or self.prohibited_claim_count < 0
        ):
            raise ValueError("prohibited_claim_count must be non-negative.")
        _string_tuple(self.missing_requirements, "missing_requirements")
        _string_tuple(self.relevance_reasons, "relevance_reasons")

    def to_dict(self) -> dict[str, Any]:
        state = dict(vars(self))
        state["missing_requirements"] = list(self.missing_requirements)
        state["relevance_reasons"] = list(self.relevance_reasons)
        return state

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> AnswerGrade:
        values = _validated_dataclass_state(cls, state, "Answer grade")
        for name in ("missing_requirements", "relevance_reasons"):
            if not isinstance(values[name], list):
                raise ValueError(f"Serialized {name} must be a list.")
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class RootCauseAttribution:
    primary_cause: str
    secondary_causes: tuple[str, ...]
    reasons: tuple[str, ...]
    confidence: str
    recommended_next_action: str

    def __post_init__(self) -> None:
        causes = {
            "retrieval",
            "section_classification",
            "evidence_selection",
            "sufficiency",
            "answer_construction",
            "generation",
            "citation_validation",
            "audience_rendering",
            "benchmark_ambiguity",
            "source_extraction_quality",
            "none",
        }
        if self.primary_cause not in causes:
            raise ValueError("Unknown primary root cause.")
        _string_tuple(self.secondary_causes, "secondary_causes")
        if any(
            item not in causes - {self.primary_cause} for item in self.secondary_causes
        ):
            raise ValueError("Unknown or duplicate secondary root cause.")
        _string_tuple(self.reasons, "reasons")
        if self.confidence not in {"high", "medium", "low"}:
            raise ValueError("confidence must be high, medium, or low.")
        _nonempty(self.recommended_next_action, "recommended_next_action")

    def to_dict(self) -> dict[str, Any]:
        state = dict(vars(self))
        state["secondary_causes"] = list(self.secondary_causes)
        state["reasons"] = list(self.reasons)
        return state

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> RootCauseAttribution:
        values = _validated_dataclass_state(cls, state, "Root-cause attribution")
        for name in ("secondary_causes", "reasons"):
            if not isinstance(values[name], list):
                raise ValueError(f"Serialized {name} must be a list.")
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class QuestionEvaluation:
    """Complete stage-wise grade for one approved benchmark question."""

    question_id: str
    paper_id: str
    question_type: str
    audience_level: str
    retrieval_results: tuple[dict[str, Any], ...]
    system_answer: dict[str, Any] | None
    retrieval: RetrievalGrade
    sufficiency: SufficiencyGrade | None
    answer: AnswerGrade | None
    concepts: ConceptCoverage | None
    citations: CitationGrade | None
    audience: AudienceGrade | None
    failure_categories: tuple[str, ...]
    root_cause: RootCauseAttribution
    automatic_pass: bool
    automated_confidence: str

    def __post_init__(self) -> None:
        _nonempty(self.question_id, "question_id")
        _nonempty(self.paper_id, "paper_id")
        if self.question_type not in QUESTION_TYPES:
            raise ValueError("Unknown question_type.")
        if self.audience_level not in AUDIENCE_LEVELS:
            raise ValueError("Unknown audience_level.")
        if not isinstance(self.retrieval_results, tuple) or not all(
            isinstance(item, dict) for item in self.retrieval_results
        ):
            raise TypeError("retrieval_results must contain dictionaries.")
        canonical_json(list(self.retrieval_results))
        if self.system_answer is not None:
            if not isinstance(self.system_answer, dict):
                raise TypeError("system_answer must be None or a dictionary.")
            canonical_json(self.system_answer)
        if not isinstance(self.retrieval, RetrievalGrade):
            raise TypeError("retrieval must be RetrievalGrade.")
        for name, expected in (
            ("sufficiency", SufficiencyGrade),
            ("answer", AnswerGrade),
            ("concepts", ConceptCoverage),
            ("citations", CitationGrade),
            ("audience", AudienceGrade),
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"{name} must be None or {expected.__name__}.")
        _string_tuple(self.failure_categories, "failure_categories")
        if not isinstance(self.root_cause, RootCauseAttribution):
            raise TypeError("root_cause must be RootCauseAttribution.")
        if not isinstance(self.automatic_pass, bool):
            raise TypeError("automatic_pass must be boolean.")
        if self.automated_confidence not in {"high", "medium", "low"}:
            raise ValueError("automated_confidence must be high, medium, or low.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "paper_id": self.paper_id,
            "question_type": self.question_type,
            "audience_level": self.audience_level,
            "retrieval_results": list(self.retrieval_results),
            "system_answer": self.system_answer,
            "retrieval": self.retrieval.to_dict(),
            "sufficiency": (
                None if self.sufficiency is None else self.sufficiency.to_dict()
            ),
            "answer": None if self.answer is None else self.answer.to_dict(),
            "concepts": None if self.concepts is None else self.concepts.to_dict(),
            "citations": (None if self.citations is None else self.citations.to_dict()),
            "audience": None if self.audience is None else self.audience.to_dict(),
            "failure_categories": list(self.failure_categories),
            "root_cause": self.root_cause.to_dict(),
            "automatic_pass": self.automatic_pass,
            "automated_confidence": self.automated_confidence,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> QuestionEvaluation:
        values = _validated_dataclass_state(cls, state, "Question evaluation")
        if not isinstance(values["retrieval_results"], list):
            raise ValueError("Serialized retrieval_results must be a list.")
        values["retrieval_results"] = tuple(values["retrieval_results"])
        values["retrieval"] = RetrievalGrade.from_dict(values["retrieval"])
        for name, expected in (
            ("sufficiency", SufficiencyGrade),
            ("answer", AnswerGrade),
            ("concepts", ConceptCoverage),
            ("citations", CitationGrade),
            ("audience", AudienceGrade),
        ):
            if values[name] is not None:
                values[name] = expected.from_dict(values[name])
        if not isinstance(values["failure_categories"], list):
            raise ValueError("Serialized failure_categories must be a list.")
        values["failure_categories"] = tuple(values["failure_categories"])
        values["root_cause"] = RootCauseAttribution.from_dict(values["root_cause"])
        return cls(**values)


@dataclass(frozen=True)
class EvaluationRun:
    """Versioned complete benchmark run with per-question and aggregate results."""

    run_id: str
    benchmark_sha256: str
    index_sha256: str
    configuration: EvaluationConfig
    question_results: tuple[QuestionEvaluation, ...]
    aggregate_metrics: dict[str, float]
    package_version: str = __version__
    format_version: int = EVALUATION_FORMAT_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.run_id, "run_id")
        _sha256(self.benchmark_sha256, "benchmark_sha256")
        _sha256(self.index_sha256, "index_sha256")
        if not isinstance(self.configuration, EvaluationConfig):
            raise TypeError("configuration must be EvaluationConfig.")
        if not isinstance(self.question_results, tuple) or not all(
            isinstance(item, QuestionEvaluation) for item in self.question_results
        ):
            raise TypeError("question_results must contain QuestionEvaluation objects.")
        identifiers = [item.question_id for item in self.question_results]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Evaluation question IDs must be unique.")
        if not isinstance(self.aggregate_metrics, dict) or any(
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for name, value in self.aggregate_metrics.items()
        ):
            raise ValueError("aggregate_metrics must map names to finite numbers.")
        _nonempty(self.package_version, "package_version")
        if self.format_version != EVALUATION_FORMAT_VERSION:
            raise ValueError("Evaluation format_version is unsupported.")

    @property
    def run_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json(self._identity_state()).encode("utf-8")
        ).hexdigest()

    def _identity_state(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "benchmark_sha256": self.benchmark_sha256,
            "index_sha256": self.index_sha256,
            "configuration": self.configuration.to_dict(),
            "question_results": [item.to_dict() for item in self.question_results],
            "aggregate_metrics": dict(sorted(self.aggregate_metrics.items())),
            "package_version": self.package_version,
            "format_version": self.format_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_state(), "run_sha256": self.run_sha256}

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> EvaluationRun:
        expected = set(cls.__dataclass_fields__) | {"run_sha256"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Evaluation run state is malformed.")
        values = dict(state)
        digest = values.pop("run_sha256")
        values["configuration"] = EvaluationConfig.from_dict(values["configuration"])
        results = values["question_results"]
        if not isinstance(results, list):
            raise ValueError("Serialized question_results must be a list.")
        values["question_results"] = tuple(
            QuestionEvaluation.from_dict(item) for item in results
        )
        run = cls(**values)
        if digest != run.run_sha256:
            raise ValueError("Evaluation run hash is inconsistent.")
        return run


@dataclass(frozen=True)
class HumanReviewRecord:
    """One reviewer judgment over an immutable automatic evaluation snapshot."""

    review_id: str
    run_id: str
    question_id: str
    question_evaluation_sha256: str
    system_answer: dict[str, Any] | None
    evidence: tuple[dict[str, Any], ...]
    automatic_grades: dict[str, Any]
    failure_categories: tuple[str, ...]
    reviewer_label: str | None = None
    reviewer_notes: str | None = None
    corrected_answer: str | None = None
    corrected_evidence: tuple[str, ...] = ()
    timestamp: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "review_id",
            "run_id",
            "question_id",
        ):
            _nonempty(getattr(self, name), name)
        _sha256(self.question_evaluation_sha256, "question_evaluation_sha256")
        if self.system_answer is not None:
            canonical_json(self.system_answer)
        if not isinstance(self.evidence, tuple) or not all(
            isinstance(item, dict) for item in self.evidence
        ):
            raise TypeError("evidence must contain dictionaries.")
        canonical_json(list(self.evidence))
        if not isinstance(self.automatic_grades, dict):
            raise TypeError("automatic_grades must be a dictionary.")
        canonical_json(self.automatic_grades)
        _string_tuple(self.failure_categories, "failure_categories")
        if (
            self.reviewer_label is not None
            and self.reviewer_label not in REVIEWER_LABELS
        ):
            raise ValueError("Unknown reviewer_label.")
        _optional_nonempty(self.reviewer_notes, "reviewer_notes")
        _optional_nonempty(self.corrected_answer, "corrected_answer")
        _string_tuple(self.corrected_evidence, "corrected_evidence")
        _optional_nonempty(self.timestamp, "timestamp")
        if self.corrected_answer is not None and self.reviewer_label not in {
            "correct",
            "partially_correct",
            "incorrect",
        }:
            raise ValueError("Corrected answers require a substantive reviewer label.")

    @property
    def reviewed(self) -> bool:
        return self.reviewer_label is not None

    def to_dict(self) -> dict[str, Any]:
        state = dict(vars(self))
        state["evidence"] = list(self.evidence)
        state["failure_categories"] = list(self.failure_categories)
        state["corrected_evidence"] = list(self.corrected_evidence)
        return state

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> HumanReviewRecord:
        values = _validated_dataclass_state(cls, state, "Human review record")
        for name in ("evidence", "failure_categories", "corrected_evidence"):
            if not isinstance(values[name], list):
                raise ValueError(f"Serialized {name} must be a list.")
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class CorrectionExample:
    """One human-approved failure correction for future instruction training."""

    question: str
    audience_level: str
    gold_evidence: tuple[str, ...]
    structured_answer_target: StructuredAnswerTarget
    corrected_answer: str
    incorrect_answer: str
    failure_categories: tuple[str, ...]
    citations: tuple[str, ...]
    paper_id: str
    source_sha256: str
    human_review_id: str

    def __post_init__(self) -> None:
        for name in (
            "question",
            "corrected_answer",
            "incorrect_answer",
            "paper_id",
            "human_review_id",
        ):
            _nonempty(getattr(self, name), name)
        if self.audience_level not in AUDIENCE_LEVELS:
            raise ValueError("Unknown correction audience level.")
        for name in (
            "gold_evidence",
            "failure_categories",
            "citations",
        ):
            _string_tuple(getattr(self, name), name)
        if not self.gold_evidence or not self.citations:
            raise ValueError("Corrections require gold evidence and citations.")
        if not isinstance(self.structured_answer_target, StructuredAnswerTarget):
            raise TypeError("structured_answer_target must be StructuredAnswerTarget.")
        _sha256(self.source_sha256, "source_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "audience_level": self.audience_level,
            "gold_evidence": list(self.gold_evidence),
            "structured_answer_target": self.structured_answer_target.to_dict(),
            "corrected_answer": self.corrected_answer,
            "incorrect_answer": self.incorrect_answer,
            "failure_categories": list(self.failure_categories),
            "citations": list(self.citations),
            "paper_id": self.paper_id,
            "source_sha256": self.source_sha256,
            "human_review_id": self.human_review_id,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> CorrectionExample:
        values = _validated_dataclass_state(cls, state, "Correction example")
        for name in ("gold_evidence", "failure_categories", "citations"):
            if not isinstance(values[name], list):
                raise ValueError(f"Serialized {name} must be a list.")
            values[name] = tuple(values[name])
        values["structured_answer_target"] = StructuredAnswerTarget.from_dict(
            values["structured_answer_target"]
        )
        return cls(**values)


def question_evaluation_sha256(value: QuestionEvaluation) -> str:
    """Hash one exact automatic evaluation for review-queue linkage."""
    if not isinstance(value, QuestionEvaluation):
        raise TypeError("value must be QuestionEvaluation.")
    return hashlib.sha256(canonical_json(value.to_dict()).encode("utf-8")).hexdigest()


def _validated_dataclass_state(
    cls: type,
    state: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    if not isinstance(state, Mapping) or set(state) != set(cls.__dataclass_fields__):
        raise ValueError(f"{label} state is malformed.")
    return dict(state)


def stable_run_id(
    benchmark: Benchmark,
    index: RetrievalIndex,
    configuration: EvaluationConfig,
) -> str:
    """Create a deterministic run ID from every comparison-critical identity."""
    return stable_identifier(
        "run",
        benchmark.benchmark_sha256,
        index.index_sha256,
        configuration.to_dict(),
        __version__,
    )
