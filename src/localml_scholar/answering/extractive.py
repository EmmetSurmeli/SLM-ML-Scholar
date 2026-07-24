"""Deterministic source-copying answer baselines with mandatory citations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from localml_scholar.answering.citations import format_inline_citation
from localml_scholar.answering.evidence import meaningful_query_terms
from localml_scholar.answering.models import EvidenceItem
from localml_scholar.answering.segmentation import SentenceSpan, segment_source_text
from localml_scholar.retrieval import tokenize_lexically


@dataclass(frozen=True)
class ExtractiveAnswerConfig:
    """Sentence-copying strategy and bounded answer size."""

    strategy: str = "sentences"
    maximum_sentences: int = 3
    maximum_answer_characters: int = 1600
    include_code_blocks: bool = True
    query_coverage_stop: float = 0.8
    minimum_new_query_terms_after_first: int = 1
    minimum_relative_sentence_score: float = 0.5

    def __post_init__(self) -> None:
        if self.strategy not in {"top_passage", "sentences"}:
            raise ValueError("strategy must be 'top_passage' or 'sentences'.")
        for name in (
            "maximum_sentences",
            "maximum_answer_characters",
            "minimum_new_query_terms_after_first",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if not isinstance(self.include_code_blocks, bool):
            raise TypeError("include_code_blocks must be boolean.")
        if isinstance(self.query_coverage_stop, bool) or not isinstance(
            self.query_coverage_stop, (int, float)
        ):
            raise TypeError("query_coverage_stop must be a real number.")
        if not 0.0 < self.query_coverage_stop <= 1.0:
            raise ValueError("query_coverage_stop must lie in (0, 1].")
        if isinstance(self.minimum_relative_sentence_score, bool) or not isinstance(
            self.minimum_relative_sentence_score, (int, float)
        ):
            raise TypeError("minimum_relative_sentence_score must be a real number.")
        if not 0.0 <= self.minimum_relative_sentence_score <= 1.0:
            raise ValueError("minimum_relative_sentence_score must lie in [0, 1].")

    def to_dict(self) -> dict[str, object]:
        """Return the exact deterministic sentence-selection policy."""
        return dict(vars(self))


@dataclass(frozen=True)
class ExtractedSentence:
    """One exact evidence substring selected for an extractive answer."""

    text: str
    evidence_label: str
    evidence_id: str
    source_start: int
    source_end: int
    score: float
    kind: str


@dataclass(frozen=True)
class ExtractiveResult:
    """Rendered answer and exact source-sentence selection trace."""

    answer_text: str
    sentences: tuple[ExtractedSentence, ...]


def _sentence_score(
    question_terms: set[str],
    span: SentenceSpan,
    evidence: EvidenceItem,
) -> float:
    terms = set(tokenize_lexically(span.text))
    overlap = len(question_terms & terms)
    coverage = overlap / max(1, len(question_terms))
    compactness = overlap / max(1, len(terms))
    retrieval = evidence.retrieval_score / (1.0 + evidence.retrieval_score)
    return coverage * 0.55 + compactness * 0.25 + retrieval * 0.20


def _candidates(
    question: str,
    evidence: tuple[EvidenceItem, ...],
    config: ExtractiveAnswerConfig,
) -> list[tuple[float, EvidenceItem, SentenceSpan]]:
    question_terms = set(meaningful_query_terms(question))
    candidates: list[tuple[float, EvidenceItem, SentenceSpan]] = []
    for item in evidence:
        for span in segment_source_text(item.selected_text):
            if span.kind == "heading":
                continue
            if span.kind == "code" and not config.include_code_blocks:
                continue
            score = _sentence_score(question_terms, span, item)
            candidates.append((score, item, span))
    candidates.sort(
        key=lambda candidate: (
            -candidate[0],
            candidate[1].retrieval_rank,
            candidate[2].start_character,
            candidate[1].evidence_id,
        )
    )
    return candidates


def _select_sentences(
    question: str,
    evidence: tuple[EvidenceItem, ...],
    config: ExtractiveAnswerConfig,
) -> tuple[ExtractedSentence, ...]:
    candidates = _candidates(question, evidence, config)
    question_terms = set(meaningful_query_terms(question))
    if config.strategy == "top_passage":
        candidates = [
            candidate
            for candidate in candidates
            if candidate[1].evidence_id == evidence[0].evidence_id
        ]
        candidates.sort(key=lambda candidate: candidate[2].start_character)
    selected: list[ExtractedSentence] = []
    seen_text: set[str] = set()
    covered_question_terms: set[str] = set()
    rendered_characters = len("The indexed sources state:\n")
    best_score = candidates[0][0] if candidates else 0.0
    for score, item, span in candidates:
        normalized = " ".join(span.text.split()).casefold()
        if not normalized or normalized in seen_text:
            continue
        sentence_query_terms = question_terms & set(tokenize_lexically(span.text))
        new_query_terms = sentence_query_terms - covered_question_terms
        if config.strategy == "sentences":
            if score < best_score * config.minimum_relative_sentence_score:
                continue
            required_new_terms = (
                1 if not selected else config.minimum_new_query_terms_after_first
            )
            if len(new_query_terms) < required_new_terms:
                continue
        line_length = len(span.text) + len(item.label) + 6
        if rendered_characters + line_length > config.maximum_answer_characters:
            continue
        selected.append(
            ExtractedSentence(
                text=span.text,
                evidence_label=item.label,
                evidence_id=item.evidence_id,
                source_start=item.start_character + span.start_character,
                source_end=item.start_character + span.end_character,
                score=float(score),
                kind=span.kind,
            )
        )
        seen_text.add(normalized)
        covered_question_terms.update(sentence_query_terms)
        rendered_characters += line_length
        if len(selected) >= config.maximum_sentences:
            break
        if (
            config.strategy == "sentences"
            and len(covered_question_terms) / max(1, len(question_terms))
            >= config.query_coverage_stop
        ):
            break
    return tuple(selected)


class ExtractiveAnswerer:
    """Copy source sentences exactly and append their answer-local labels."""

    def __init__(self, config: ExtractiveAnswerConfig | None = None) -> None:
        self.config = config or ExtractiveAnswerConfig()
        if not isinstance(self.config, ExtractiveAnswerConfig):
            raise TypeError("config must be ExtractiveAnswerConfig.")

    def answer(
        self,
        question: str,
        evidence: tuple[EvidenceItem, ...],
    ) -> ExtractiveResult:
        """Return a deterministic cited answer without factual paraphrasing."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must contain non-whitespace text.")
        if not isinstance(evidence, tuple) or not evidence:
            raise ValueError("Extractive answering requires evidence.")
        if not all(isinstance(item, EvidenceItem) for item in evidence):
            raise TypeError("evidence must contain EvidenceItem objects.")
        selected = _select_sentences(question, evidence, self.config)
        if not selected:
            raise ValueError("No source sentence fits the extractive answer budget.")
        lines = ["The indexed sources state:"]
        for sentence in selected:
            citation = format_inline_citation((sentence.evidence_label,))
            lines.append(f"- {sentence.text} {citation}")
        answer_text = "\n".join(lines)
        if not all(math.isfinite(sentence.score) for sentence in selected):
            raise FloatingPointError("Extractive sentence score became non-finite.")
        return ExtractiveResult(answer_text=answer_text, sentences=selected)
