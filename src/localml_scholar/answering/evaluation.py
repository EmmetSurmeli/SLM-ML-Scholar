"""Authored-fixture metrics for answerability, citations, and key facts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from localml_scholar.answering.models import GroundedAnswer
from localml_scholar.retrieval import tokenize_lexically


@dataclass(frozen=True)
class GroundedQuestion:
    """One project-authored question and transparent expected evidence/content."""

    question_id: str
    question: str
    relevant_chunk_ids: tuple[str, ...]
    acceptable_citation_groups: tuple[tuple[str, ...], ...]
    expected_answerable: bool
    expected_key_facts: tuple[str, ...]
    expected_citation_sources: tuple[str, ...]
    unacceptable_claims: tuple[str, ...]
    reference_extractive_answer: str | None = None

    def __post_init__(self) -> None:
        for name in ("question_id", "question"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        for name in (
            "relevant_chunk_ids",
            "expected_key_facts",
            "expected_citation_sources",
            "unacceptable_claims",
        ):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not all(
                isinstance(item, str) and item for item in value
            ):
                raise ValueError(f"{name} must contain non-empty strings.")
        if len(set(self.relevant_chunk_ids)) != len(self.relevant_chunk_ids):
            raise ValueError("relevant_chunk_ids must be unique.")
        if not isinstance(self.acceptable_citation_groups, tuple) or not all(
            isinstance(group, tuple)
            and group
            and all(isinstance(item, str) and item for item in group)
            for group in self.acceptable_citation_groups
        ):
            raise ValueError("acceptable_citation_groups must contain ID tuples.")
        if any(
            not set(group) <= set(self.relevant_chunk_ids)
            for group in self.acceptable_citation_groups
        ):
            raise ValueError("Citation groups must reference relevant chunk IDs.")
        if not isinstance(self.expected_answerable, bool):
            raise TypeError("expected_answerable must be boolean.")
        if self.expected_answerable and not self.relevant_chunk_ids:
            raise ValueError("Answerable questions require relevant chunks.")
        if not self.expected_answerable and self.relevant_chunk_ids:
            raise ValueError("Unanswerable questions cannot require chunks.")
        if self.reference_extractive_answer is not None and (
            not isinstance(self.reference_extractive_answer, str)
            or not self.reference_extractive_answer
        ):
            raise ValueError("reference_extractive_answer must be None or non-empty.")

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> GroundedQuestion:
        if not isinstance(state, Mapping) or set(state) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("Grounded QA fixture entry keys are malformed.")
        values = dict(state)
        for name in (
            "relevant_chunk_ids",
            "expected_key_facts",
            "expected_citation_sources",
            "unacceptable_claims",
        ):
            if not isinstance(values[name], list):
                raise ValueError(f"Fixture {name} must be a list.")
            values[name] = tuple(values[name])
        groups = values["acceptable_citation_groups"]
        if not isinstance(groups, list) or not all(
            isinstance(group, list) for group in groups
        ):
            raise ValueError("Fixture acceptable_citation_groups must be lists.")
        values["acceptable_citation_groups"] = tuple(tuple(group) for group in groups)
        return cls(**values)


def load_grounded_questions(path: str | Path) -> tuple[GroundedQuestion, ...]:
    """Load a deterministic project-authored question fixture."""
    source = Path(path)
    try:
        state = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Grounded QA fixture does not exist: {source}"
        ) from None
    except json.JSONDecodeError as error:
        raise ValueError("Grounded QA fixture is not valid JSON.") from error
    if not isinstance(state, dict) or set(state) != {
        "fixture_version",
        "questions",
    }:
        raise ValueError("Grounded QA fixture top-level keys are malformed.")
    if state["fixture_version"] != 1 or not isinstance(state["questions"], list):
        raise ValueError("Unsupported or malformed grounded QA fixture.")
    questions = tuple(GroundedQuestion.from_dict(item) for item in state["questions"])
    if not questions or len({item.question_id for item in questions}) != len(questions):
        raise ValueError("Grounded QA fixture IDs must be non-empty and unique.")
    return questions


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _fact_present(answer_text: str, fact: str) -> bool:
    normalized_answer = " ".join(answer_text.casefold().split())
    normalized_fact = " ".join(fact.casefold().split())
    if normalized_fact in normalized_answer:
        return True
    if re.search(r"[=<>]", fact):
        return False
    fact_terms = set(tokenize_lexically(fact))
    answer_terms = set(tokenize_lexically(answer_text))
    return bool(fact_terms) and fact_terms <= answer_terms


@dataclass(frozen=True)
class QuestionAnswerMetrics:
    """Per-question answerability, citation, support, and content metrics."""

    answerability_correct: float
    correct_answer_attempt: float
    correct_abstention: float
    false_answer: float
    false_abstention: float
    citation_validity: float
    citation_coverage: float
    citation_precision: float
    citation_recall: float
    source_location_correctness: float
    supported_claim_rate: float
    unsupported_claim_count: int
    numerical_mismatch_count: int
    negation_warning_count: int
    key_fact_recall: float
    prohibited_claim_count: int
    answer_character_count: int
    evidence_count: int

    def to_dict(self) -> dict[str, float | int]:
        return dict(vars(self))


@dataclass(frozen=True)
class AnswerEvaluation:
    """Deterministic per-question and aggregate authored-fixture metrics."""

    per_question: dict[str, QuestionAnswerMetrics]
    aggregate: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_question": {
                key: value.to_dict() for key, value in sorted(self.per_question.items())
            },
            "aggregate": dict(sorted(self.aggregate.items())),
        }


def evaluate_grounded_answers(
    questions: Sequence[GroundedQuestion],
    answers: Mapping[str, GroundedAnswer],
) -> AnswerEvaluation:
    """Evaluate exact question IDs with explicit zero-denominator behavior."""
    if isinstance(questions, (str, bytes)) or not isinstance(questions, Sequence):
        raise TypeError("questions must be a sequence.")
    if not isinstance(answers, Mapping):
        raise TypeError("answers must be a mapping.")
    expected_ids = {question.question_id for question in questions}
    if not questions or set(answers) != expected_ids:
        raise ValueError("answers must contain exactly every fixture question ID.")
    per_question: dict[str, QuestionAnswerMetrics] = {}
    for question in questions:
        answer = answers[question.question_id]
        if not isinstance(answer, GroundedAnswer):
            raise TypeError("answers must contain GroundedAnswer values.")
        attempted = not answer.abstained
        correct_attempt = float(question.expected_answerable and attempted)
        correct_abstention = float(
            not question.expected_answerable and answer.abstained
        )
        false_answer = float(not question.expected_answerable and attempted)
        false_abstention = float(question.expected_answerable and answer.abstained)
        cited_chunks = {
            item.chunk_id
            for item in answer.evidence
            if any(item.label in claim.citation_labels for claim in answer.claims)
        }
        relevant = set(question.relevant_chunk_ids)
        citation_precision = (
            1.0
            if not cited_chunks and not relevant
            else _safe_ratio(len(cited_chunks & relevant), len(cited_chunks))
        )
        if not question.acceptable_citation_groups:
            citation_recall = 1.0
        else:
            citation_recall = _safe_ratio(
                sum(
                    bool(cited_chunks & set(group))
                    for group in question.acceptable_citation_groups
                ),
                len(question.acceptable_citation_groups),
            )
        cited_sources = {
            item.source_name
            for item in answer.evidence
            if item.chunk_id in cited_chunks
        }
        expected_sources = set(question.expected_citation_sources)
        source_correctness = float(
            (not expected_sources and not cited_sources)
            or bool(cited_sources)
            and cited_sources <= expected_sources
        )
        substantive = [claim for claim in answer.claims if claim.substantive]
        supported_rate = (
            1.0
            if not substantive
            else sum(claim.supported for claim in substantive) / len(substantive)
        )
        key_fact_recall = (
            1.0
            if not question.expected_key_facts
            else sum(
                _fact_present(answer.answer_text, fact)
                for fact in question.expected_key_facts
            )
            / len(question.expected_key_facts)
        )
        prohibited = sum(
            _fact_present(answer.answer_text, claim)
            for claim in question.unacceptable_claims
        )
        per_question[question.question_id] = QuestionAnswerMetrics(
            answerability_correct=float(correct_attempt or correct_abstention),
            correct_answer_attempt=correct_attempt,
            correct_abstention=correct_abstention,
            false_answer=false_answer,
            false_abstention=false_abstention,
            citation_validity=float(answer.validation.citations_valid),
            citation_coverage=answer.validation.citation_coverage,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            source_location_correctness=source_correctness,
            supported_claim_rate=supported_rate,
            unsupported_claim_count=answer.validation.unsupported_claim_count,
            numerical_mismatch_count=answer.validation.numerical_mismatch_count,
            negation_warning_count=answer.validation.negation_warning_count,
            key_fact_recall=key_fact_recall,
            prohibited_claim_count=prohibited,
            answer_character_count=len(answer.answer_text),
            evidence_count=len(answer.evidence),
        )
    metric_names = tuple(QuestionAnswerMetrics.__dataclass_fields__)
    aggregate = {
        name: fmean(float(getattr(metrics, name)) for metrics in per_question.values())
        for name in metric_names
    }
    attempted_abstentions = sum(answer.abstained for answer in answers.values())
    actual_unanswerable = sum(not item.expected_answerable for item in questions)
    correct_abstentions = sum(
        not item.expected_answerable and answers[item.question_id].abstained
        for item in questions
    )
    aggregate["abstention_precision"] = _safe_ratio(
        correct_abstentions,
        attempted_abstentions,
    )
    aggregate["abstention_recall"] = _safe_ratio(
        correct_abstentions,
        actual_unanswerable,
    )
    if not all(math.isfinite(value) for value in aggregate.values()):
        raise FloatingPointError("Answer evaluation produced a non-finite metric.")
    return AnswerEvaluation(per_question=per_question, aggregate=aggregate)
