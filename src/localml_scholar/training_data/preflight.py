"""Cheap deterministic gates before autonomous paper curation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from localml_scholar.retrieval import Document, RetrievalIndex
from localml_scholar.retrieval.section_inference import (
    canonical_section_role,
    infer_scholarly_headings,
    rebuild_document_sections,
    section_topics_compatible,
)

_TOPIC_SIGNALS: dict[str, tuple[str, ...]] = {
    "ablation": (
        "ablation",
        "ablate",
        "ablated",
        "component analysis",
        "variant",
        "without",
    ),
    "architecture": (
        "architecture",
        "layer",
        "module",
        "network",
        "encoder",
        "decoder",
        "block",
    ),
    "complexity": (
        "complexity",
        "runtime",
        "memory",
        "flops",
        "asymptotic",
        "computational cost",
    ),
    "reproduction": (
        "optimizer",
        "learning rate",
        "batch size",
        "training",
        "epochs",
        "hardware",
    ),
    "experiment": ("experiment", "dataset", "evaluation", "baseline", "metric"),
    "limitation": ("limitation", "drawback", "failure", "future work"),
    "derivation": ("equation", "lemma", "theorem", "proof", "objective", "derive"),
    "method": ("method", "model", "algorithm", "approach", "propose"),
}
_ANSWERABILITY = {"answerable", "partial", "abstain", "external_required"}


@dataclass(frozen=True)
class IngestionHealthConfig:
    """Thresholds used to block unreliable autonomous question generation."""

    maximum_untitled_fraction: float = 0.25
    maximum_duplicate_fraction: float = 0.50
    minimum_extracted_characters: int = 20

    def __post_init__(self) -> None:
        for name in ("maximum_untitled_fraction", "maximum_duplicate_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if isinstance(self.minimum_extracted_characters, bool) or not isinstance(
            self.minimum_extracted_characters, int
        ):
            raise TypeError("minimum_extracted_characters must be an integer.")
        if self.minimum_extracted_characters <= 0:
            raise ValueError("minimum_extracted_characters must be positive.")


@dataclass(frozen=True)
class PaperIngestionHealth:
    """Inspectable per-paper extraction and section-structure health."""

    paper_id: str
    section_count: int
    titled_section_fraction: float
    duplicate_section_fraction: float
    untitled_section_fraction: float
    average_chunk_length: float
    empty_chunk_count: int
    heading_confidence: float
    extraction_warnings: tuple[str, ...]
    healthy_for_question_generation: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "section_count": self.section_count,
            "titled_section_fraction": self.titled_section_fraction,
            "duplicate_section_fraction": self.duplicate_section_fraction,
            "untitled_section_fraction": self.untitled_section_fraction,
            "average_chunk_length": self.average_chunk_length,
            "empty_chunk_count": self.empty_chunk_count,
            "heading_confidence": self.heading_confidence,
            "extraction_warnings": list(self.extraction_warnings),
            "healthy_for_question_generation": self.healthy_for_question_generation,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PaperIngestionHealth:
        """Load a strict cached health report."""
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Cached paper-ingestion health is malformed.")
        state = dict(value)
        warnings = state["extraction_warnings"]
        if not isinstance(warnings, list):
            raise ValueError("Cached extraction_warnings must be a list.")
        state["extraction_warnings"] = tuple(warnings)
        return cls(**state)


class DeterministicPreflightCache:
    """Hash-keyed cache that never reuses data after a source/index change."""

    def __init__(self, state: dict[str, Any] | None = None) -> None:
        self._state = {} if state is None else dict(state)
        if not all(
            isinstance(key, str) and isinstance(value, dict)
            for key, value in self._state.items()
        ):
            raise ValueError("Preflight cache must map hashes to objects.")

    def get(self, source_hash: str) -> dict[str, Any] | None:
        """Return a defensive copy of a cached source-hash entry."""
        value = self._state.get(source_hash)
        return None if value is None else dict(value)

    def put(self, source_hash: str, value: dict[str, Any]) -> None:
        """Store deterministic derived data under its exact source hash."""
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise ValueError("source_hash must be a SHA-256-shaped string.")
        if not isinstance(value, dict):
            raise TypeError("Cached value must be a dictionary.")
        self._state[source_hash] = dict(value)

    def to_dict(self) -> dict[str, Any]:
        """Return serializable cache state in deterministic key order."""
        return {key: dict(self._state[key]) for key in sorted(self._state)}


def topic_signals(text: str, headings: tuple[str, ...] = ()) -> dict[str, bool]:
    """Return deterministic topic availability from source text and headings."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not isinstance(headings, tuple) or not all(
        isinstance(item, str) for item in headings
    ):
        raise TypeError("headings must be a tuple of strings.")
    normalized = " ".join((*headings, text)).casefold()
    matches = {
        topic: {signal for signal in signals if signal in normalized}
        for topic, signals in _TOPIC_SIGNALS.items()
    }
    matches["ablation"].difference_update({"without", "variant"})
    if re.search(
        r"\b(?:variant\w*[^\n.!?]{0,160}without|without[^\n.!?]{0,160}variant\w*)\b",
        normalized,
    ):
        matches["ablation"].add("variant_without")
    if any(
        re.search(r"\b(?:ablation|variant|variation)s?\b", heading.casefold())
        for heading in headings
    ):
        matches["ablation"].add("section_heading")
    return {topic: bool(values) for topic, values in matches.items()}


def question_topic_eligible(
    question_type: str,
    question: str,
    *,
    paper_text: str,
    headings: tuple[str, ...] = (),
    expected_sections: tuple[str, ...] = (),
) -> bool:
    """Reject templates whose required topic is absent from the paper."""
    if not isinstance(expected_sections, tuple) or not all(
        isinstance(item, str) and item.strip() for item in expected_sections
    ):
        raise ValueError("expected_sections must be a tuple of non-empty strings.")
    signals = topic_signals(paper_text, headings)
    topic = {
        "hyperparameter": "reproduction",
        "result": "experiment",
        "equation": "derivation",
        "teaching": "architecture" if "attention" in question.casefold() else "method",
        "intuition": "method",
        "user_style": "method",
        "summary": "method",
        "extension": "method",
        "counterfactual": "method",
    }.get(question_type, question_type)
    if topic == "critical_reasoning":
        lowered = question.casefold()
        if "experiment" in lowered or "comparison" in lowered:
            topic = "experiment"
        else:
            return True
    if topic == "limitation" and "inferred" in question.casefold():
        return False
    normalized_headings = tuple(heading.casefold() for heading in headings)
    if topic == "ablation":
        # Mentions such as "without dropout" in ordinary prose are too weak to
        # justify an autonomous ablation question. Require a structural signal.
        return any(
            re.search(r"\b(?:ablation|ablate|ablated|variation|variant)s?\b", heading)
            for heading in normalized_headings
        )
    if topic == "architecture":
        # Generic occurrences of layer/model/network are ubiquitous. A titled
        # architecture or encoder/decoder section is a conservative indication
        # that the paper contains a retrievable architecture description.
        return any(
            re.search(
                r"\b(?:architecture|architectures|encoder|decoder|"
                r"network architectures)\b",
                heading,
            )
            for heading in normalized_headings
        )
    if topic == "complexity":
        heading_signal = any(
            re.search(
                r"\b(?:complexity|runtime|memory|flops|asymptotic|io analysis)\b",
                heading,
            )
            for heading in normalized_headings
        )
        text_signal = bool(
            re.search(
                r"\b(?:time|memory|computational|io)\s+(?:and\s+)?complexity\b|"
                r"\b(?:flops|asymptotic|big[- ]?o)\b",
                paper_text.casefold(),
            )
        )
        return heading_signal or text_signal
    if (
        expected_sections
        and headings
        and not section_topics_compatible(expected_sections, headings)
    ):
        return False
    return signals.get(topic, True)


def expected_answerability(question_type: str) -> str:
    """Classify deliberate abstention categories separately from normal metrics."""
    value = {
        "insufficient_evidence": "abstain",
        "external_context": "external_required",
        "historical_impact": "external_required",
        "false_premise": "partial",
    }.get(question_type, "answerable")
    if value not in _ANSWERABILITY:  # pragma: no cover - defensive invariant
        raise RuntimeError("Invalid answerability policy.")
    return value


def paper_ingestion_health(
    document: Document,
    *,
    config: IngestionHealthConfig | None = None,
) -> PaperIngestionHealth:
    """Evaluate actual or inferable section structure without external services."""
    if not isinstance(document, Document):
        raise TypeError("document must be a Document.")
    resolved = config or IngestionHealthConfig()
    if not isinstance(resolved, IngestionHealthConfig):
        raise TypeError("config must be IngestionHealthConfig.")
    actual = [section.heading for section in document.sections]
    inferred = infer_scholarly_headings(document.text)
    actual_titles = [item for item in actual if item]
    effective = actual_titles or [item.title for item in inferred]
    section_count = len(document.sections) if actual_titles else max(1, len(inferred))
    untitled = sum(not item for item in actual)
    titled_fraction = len(effective) / max(1, section_count)
    normalized = [item.casefold() for item in effective]
    duplicate = len(normalized) - len(set(normalized))
    duplicate_fraction = duplicate / max(1, len(normalized))
    warnings: list[str] = []
    if len(document.text.strip()) < resolved.minimum_extracted_characters:
        warnings.append("empty_or_near_empty_extraction")
    actual_untitled_fraction = untitled / max(1, len(document.sections))
    if actual_untitled_fraction > resolved.maximum_untitled_fraction and not inferred:
        warnings.append("section_structure_low_confidence")
    if not any(canonical_section_role(item) is not None for item in effective):
        warnings.append("no_recognizable_major_section")
    if duplicate_fraction > resolved.maximum_duplicate_fraction:
        warnings.append("duplicate_heading_rate_high")
    confidence_values = [item.confidence for item in inferred]
    heading_confidence = (
        1.0
        if any(actual)
        else sum(confidence_values) / len(confidence_values)
        if confidence_values
        else 0.0
    )
    lengths = [len(section.text) for section in document.sections]
    return PaperIngestionHealth(
        paper_id=document.document_id,
        section_count=section_count,
        titled_section_fraction=min(1.0, titled_fraction),
        duplicate_section_fraction=duplicate_fraction,
        untitled_section_fraction=(actual_untitled_fraction if not inferred else 0.0),
        average_chunk_length=sum(lengths) / max(1, len(lengths)),
        empty_chunk_count=sum(
            not section.text.strip() for section in document.sections
        ),
        heading_confidence=heading_confidence,
        extraction_warnings=tuple(warnings),
        healthy_for_question_generation=not warnings,
    )


def rebuild_index_section_structure(
    index: RetrievalIndex,
) -> tuple[RetrievalIndex, dict[str, Any]]:
    """Rebuild an immutable index after deterministic section repair."""
    if not isinstance(index, RetrievalIndex):
        raise TypeError("index must be a RetrievalIndex.")
    documents = tuple(rebuild_document_sections(item) for item in index.documents)
    changed = sum(
        before.to_dict() != after.to_dict()
        for before, after in zip(index.documents, documents, strict=True)
    )
    if not changed:
        return index, {"documents_changed": 0, "index_changed": False}
    rebuilt = RetrievalIndex.build(
        documents,
        index_config=index.index_config,
        chunking_config=index.chunking_config,
        lexical_config=index.lexical_config,
        bm25_config=index.bm25_config,
    )
    if index.semantic_index is not None:
        rebuilt = rebuilt.enrich_semantic(index.semantic_index.config)
    return rebuilt, {
        "documents_changed": changed,
        "index_changed": rebuilt.index_sha256 != index.index_sha256,
        "old_index_sha256": index.index_sha256,
        "new_index_sha256": rebuilt.index_sha256,
    }


def pipeline_self_test() -> dict[str, Any]:
    """Run a fast authored end-to-end preflight without Codex or network access."""
    from localml_scholar.answering import GroundedAnswerPipeline
    from localml_scholar.retrieval import ingest_markdown, normalize_query_terms
    from localml_scholar.retrieval.query import question_concepts
    from localml_scholar.training_data.questions import generate_paper_questions

    supported_text = """# Controlled Study

## Abstract
We propose a small encoder model for deterministic classification.

## Method
The architecture contains two encoder layers and a linear classifier.

## Ablation
The ablation removes the second layer and reduces accuracy from 90 to 80 percent.

## Experiments
Training uses Adam, a batch size of 16, and a learning rate of 0.001.

## Limitations
The evaluation uses only one synthetic dataset.
"""
    unsupported_text = """# Minimal Study

## Abstract
We describe a deterministic sorting method.

## Method
The method sorts integer inputs in ascending order.

## Results
The method returns the expected order on the authored fixture.
"""
    supported = ingest_markdown(supported_text, source="fixture-supported.md")
    unsupported = ingest_markdown(unsupported_text, source="fixture-unsupported.md")
    headings = infer_scholarly_headings(
        "Abstract\nText\n1 Introduction\nText\n3.2 Training\nText\nREFERENCES\n"
    )
    health = paper_ingestion_health(supported)
    generated = generate_paper_questions(
        supported.document_id,
        supported.title or "Controlled Study",
        count=40,
        section_titles=tuple(
            item.heading for item in supported.sections if item.heading
        ),
        paper_text=supported.text,
    )
    unsupported_generated = generate_paper_questions(
        unsupported.document_id,
        unsupported.title or "Minimal Study",
        count=40,
        section_titles=tuple(
            item.heading for item in unsupported.sections if item.heading
        ),
        paper_text=unsupported.text,
    )
    index = RetrievalIndex.build((supported, unsupported))
    answer = GroundedAnswerPipeline(index).answer(
        "What does the ablation establish?",
        filters=None,
    )
    abstention = GroundedAnswerPipeline(index).answer(
        "What FLOPs are reported for the nonexistent quantum decoder?",
    )
    checks = {
        "section_extraction": {item.title for item in headings}
        >= {"Abstract", "Introduction", "Training", "References"},
        "ingestion_health": health.healthy_for_question_generation,
        "topic_aware_generation": any(
            item.question_type == "ablation" for item in generated
        )
        and not any(item.question_type == "ablation" for item in unsupported_generated),
        "query_normalization": normalize_query_terms(
            "What does the ablation establish?"
        )
        == ("ablation", "establish"),
        "essential_concepts": question_concepts(
            "What does the ablation establish?"
        ).essential_terms
        == ("ablation",),
        "evidence_sufficiency": not answer.abstained and answer.sufficiency.sufficient,
        "abstention_behavior": abstention.abstained
        and abstention.metadata.get("grounded_abstention") is not None,
        "citation_construction": bool(answer.citations) and answer.validation.accepted,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "codex_calls": 0,
        "fixture_documents": 2,
    }
