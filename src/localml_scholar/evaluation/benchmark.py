"""Deterministic benchmark candidates and explicit human-approval transitions."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from localml_scholar.evaluation.schemas import (
    Benchmark,
    BenchmarkQuestion,
    ConceptGroup,
    GoldEvidence,
)
from localml_scholar.retrieval import RetrievalIndex
from localml_scholar.scholarly import ScholarlyAnalysisPipeline


def _chunk_for_range(index: RetrievalIndex, document_id: str, start: int, end: int):
    candidates = tuple(
        chunk
        for chunk in index.chunks
        if chunk.document_id == document_id
        and chunk.start_character <= start < end <= chunk.end_character
    )
    if candidates:
        return min(
            candidates,
            key=lambda item: (
                item.end_character - item.start_character,
                item.ordinal,
                item.chunk_id,
            ),
        )
    overlapping = tuple(
        chunk
        for chunk in index.chunks
        if chunk.document_id == document_id
        and max(chunk.start_character, start) < min(chunk.end_character, end)
    )
    return overlapping[0] if overlapping else None


def _gold_from_citation(index: RetrievalIndex, citation) -> tuple[GoldEvidence, ...]:
    chunk = _chunk_for_range(
        index,
        citation.document_id,
        citation.start_character,
        citation.end_character,
    )
    if chunk is None:
        return ()
    start = max(citation.start_character, chunk.start_character)
    end = min(citation.end_character, chunk.end_character)
    return (
        GoldEvidence(
            chunk_id=chunk.chunk_id,
            section_id=chunk.section_id,
            start_character=start,
            end_character=end,
            relevance_grade=3,
        ),
    )


def _concept(value: object) -> tuple[ConceptGroup, ...]:
    if isinstance(value, str) and value.strip():
        return (ConceptGroup(value.strip()),)
    if isinstance(value, dict):
        candidates = tuple(
            str(item).strip()
            for item in value.values()
            if isinstance(item, (str, int, float)) and str(item).strip()
        )
        return tuple(ConceptGroup(item) for item in dict.fromkeys(candidates[:2]))
    if isinstance(value, list):
        candidates = tuple(
            str(item).strip()
            for item in value
            if isinstance(item, (str, int, float)) and str(item).strip()
        )
        return tuple(ConceptGroup(item) for item in dict.fromkeys(candidates[:2]))
    return ()


def _candidate(
    *,
    paper_id: str,
    question: str,
    question_type: str,
    audience: str,
    expected_sections: tuple[str, ...],
    evidence=(),
    concepts=(),
    prohibited=(),
    confidence_reason: str,
    answerability: str = "paper_answerable",
    sufficiency: str = "sufficient",
) -> BenchmarkQuestion:
    return BenchmarkQuestion.create(
        paper_id=paper_id,
        question=question,
        question_type=question_type,
        audience_level=audience,
        answerability=answerability,
        paper_sufficiency=sufficiency,
        expected_sections=expected_sections,
        gold_evidence=tuple(evidence),
        required_concepts=tuple(concepts),
        prohibited_claims=tuple(prohibited),
        review_status="proposed",
        metadata={
            "candidate_generator": "scholarly_artifacts_v1",
            "confidence_reason": confidence_reason,
            "trusted_gold": False,
        },
    )


def generate_candidate_benchmark(
    index: RetrievalIndex,
    document_id: str,
    *,
    name: str | None = None,
    benchmark_version: str = "0.1-candidates",
) -> Benchmark:
    """Propose source-linked questions; never mark them trusted or approved."""
    if not isinstance(index, RetrievalIndex):
        raise TypeError("index must be a RetrievalIndex.")
    documents = {document.document_id: document for document in index.documents}
    if document_id not in documents:
        raise ValueError(f"Index does not contain document_id {document_id!r}.")
    analysis = ScholarlyAnalysisPipeline(index).analyze_paper(document_id)
    paper = analysis.paper
    audiences = ("beginner", "undergraduate", "researcher")
    proposals: list[BenchmarkQuestion] = []

    metadata_fields = (
        ("title", "What is the exact title of the paper?", "title"),
        ("authors", "Who wrote the paper?", "authors"),
        ("year", "When was the paper published?", "publication year"),
    )
    for ordinal, (field, question, concept_name) in enumerate(metadata_fields):
        evidence = getattr(paper, field)
        if evidence is None:
            continue
        proposals.append(
            _candidate(
                paper_id=document_id,
                question=question,
                question_type="metadata",
                audience=audiences[ordinal % len(audiences)],
                expected_sections=("title", "abstract"),
                evidence=_gold_from_citation(index, evidence.citation),
                concepts=(ConceptGroup(concept_name),),
                confidence_reason=f"explicit_{field}_artifact",
            )
        )

    artifact_groups = (
        (
            analysis.methodology,
            "main_method",
            "What is the paper's main method?",
            ("method", "methodology", "architecture"),
        ),
        (
            analysis.assumptions,
            "assumption",
            "What assumptions does the method state?",
            ("method", "theory", "discussion"),
        ),
        (
            analysis.datasets,
            "experiment",
            "Which datasets are used in the experiments?",
            ("experiment", "data"),
        ),
        (
            analysis.metrics,
            "experiment",
            "Which evaluation metrics are reported?",
            ("experiment", "result"),
        ),
        (
            analysis.hyperparameters,
            "hyperparameter",
            "Which explicit training hyperparameters are reported?",
            ("training", "experiment", "implementation"),
        ),
        (
            analysis.results,
            "result",
            "What quantitative results does the paper report?",
            ("result", "experiment"),
        ),
        (
            analysis.ablations,
            "ablation",
            "What ablation studies does the paper report?",
            ("ablation", "experiment"),
        ),
        (
            analysis.limitations,
            "limitation",
            "What limitations does the paper explicitly state?",
            ("limitation", "discussion", "conclusion"),
        ),
    )
    for records, question_type, question, sections in artifact_groups:
        for record in records[:3]:
            proposals.append(
                _candidate(
                    paper_id=document_id,
                    question=question,
                    question_type=question_type,
                    audience=audiences[len(proposals) % len(audiences)],
                    expected_sections=sections,
                    evidence=_gold_from_citation(index, record.citation),
                    concepts=_concept(record.value),
                    confidence_reason=f"explicit_{record.category}_artifact",
                )
            )

    for equation in analysis.equations[:5]:
        number = equation.equation_number
        question = (
            f"What does Equation {number} express?"
            if number is not None
            else "What does this detected equation express?"
        )
        proposals.append(
            _candidate(
                paper_id=document_id,
                question=question,
                question_type="equation",
                audience=audiences[len(proposals) % len(audiences)],
                expected_sections=("method", "theory", "appendix"),
                evidence=_gold_from_citation(index, equation.citation),
                concepts=(ConceptGroup(equation.normalized_text[:160]),),
                confidence_reason="detected_equation_with_exact_source_range",
            )
        )

    for entry in analysis.notation[:5]:
        evidence = (
            entry.selected_definition.citation
            if entry.selected_definition is not None
            else entry.occurrences[0]
        )
        proposals.append(
            _candidate(
                paper_id=document_id,
                question=f"What does the symbol {entry.symbol} mean in the paper?",
                question_type="notation",
                audience=audiences[len(proposals) % len(audiences)],
                expected_sections=("method", "theory", "appendix"),
                evidence=_gold_from_citation(index, evidence),
                concepts=(ConceptGroup(entry.symbol),),
                confidence_reason="notation_occurrence_or_definition_artifact",
            )
        )

    proposals.extend(
        [
            _candidate(
                paper_id=document_id,
                question="Does the paper prove its method is always superior?",
                question_type="false_premise",
                audience="researcher",
                expected_sections=("result", "limitation", "discussion"),
                prohibited=("The method is always superior",),
                confidence_reason="authored_false_premise_control",
                answerability="ambiguous",
                sufficiency="partially_sufficient",
            ),
            _candidate(
                paper_id=document_id,
                question="What historical developments followed from this paper?",
                question_type="historical_impact",
                audience="researcher",
                expected_sections=("introduction", "conclusion", "references"),
                prohibited=("The paper alone caused later progress",),
                confidence_reason="authored_external_context_control",
                answerability="external_sources_required",
                sufficiency="external_required",
            ),
        ]
    )
    unique = {
        item.question_id: item
        for item in proposals
        if item.gold_evidence or (item.answerability != "paper_answerable")
    }
    questions = tuple(unique[key] for key in sorted(unique))
    if not questions:
        raise ValueError("Scholarly artifacts produced no reviewable candidates.")
    document = documents[document_id]
    return Benchmark(
        name=name or f"{document.title or document.source_name} candidates",
        benchmark_version=benchmark_version,
        index_sha256=index.index_sha256,
        document_hashes={document_id: document.content_sha256},
        questions=questions,
    )


def apply_review_decisions(
    candidates: Benchmark,
    decisions: dict[str, dict[str, Any]],
) -> Benchmark:
    """Apply explicit reviewer decisions without auto-approving omitted questions."""
    if not isinstance(candidates, Benchmark):
        raise TypeError("candidates must be a Benchmark.")
    if not isinstance(decisions, dict):
        raise TypeError("decisions must be a dictionary.")
    known = {item.question_id for item in candidates.questions}
    if not set(decisions) <= known:
        raise ValueError("Review decisions contain an unknown question ID.")
    reviewed: list[BenchmarkQuestion] = []
    editable_fields = {
        "question",
        "question_type",
        "audience_level",
        "answerability",
        "paper_sufficiency",
        "expected_sections",
        "forbidden_sections",
        "gold_evidence",
        "acceptable_chunk_ids",
        "required_concepts",
        "optional_concepts",
        "prohibited_claims",
        "expected_numbers",
        "expected_identifiers",
        "acceptable_abstention_reasons",
        "completeness_requirements",
        "gold_core_answer",
        "gold_notes",
    }
    tuple_fields = editable_fields - {
        "question",
        "question_type",
        "audience_level",
        "answerability",
        "paper_sufficiency",
        "gold_core_answer",
        "gold_notes",
    }
    for question in candidates.questions:
        decision = decisions.get(question.question_id)
        if decision is None:
            reviewed.append(question)
            continue
        if not isinstance(decision, dict):
            raise ValueError("Each review decision must be an object.")
        status = decision.get("status")
        if status not in {"approved", "edited", "rejected"}:
            raise ValueError(
                "Review decision status must be approved, edited, or rejected."
            )
        edits = decision.get("edits", {})
        if not isinstance(edits, dict) or not set(edits) <= editable_fields:
            raise ValueError("Review decision contains unsupported edits.")
        values = dict(edits)
        for field in tuple_fields & set(values):
            if not isinstance(values[field], list):
                raise ValueError(f"Review edit {field} must be a list.")
            if field == "gold_evidence":
                values[field] = tuple(
                    GoldEvidence.from_dict(item) for item in values[field]
                )
            elif field in {"required_concepts", "optional_concepts"}:
                values[field] = tuple(
                    ConceptGroup.from_dict(item) for item in values[field]
                )
            else:
                values[field] = tuple(values[field])
        notes = decision.get("gold_notes")
        if notes is not None:
            values["gold_notes"] = notes
        if status in {"approved", "edited"} and not (
            values.get("gold_notes") or question.gold_notes
        ):
            raise ValueError("Approved review decisions require gold_notes.")
        reviewed.append(replace(question, review_status=status, **values))
    return replace(candidates, questions=tuple(reviewed))
