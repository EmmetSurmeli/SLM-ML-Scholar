"""Shared deterministic fixture runner for Milestone 9 experiments."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from experiments.grounded_qa_fixture import (
    build_grounded_fixture_index,
    load_grounded_fixture_questions,
)
from localml_scholar._version import __version__
from localml_scholar.answering import (
    EvidenceSelectionConfig,
    GroundedAnswerPipeline,
    GroundedGenerativeAnswerer,
)
from localml_scholar.answering.evaluation import evaluate_grounded_answers
from localml_scholar.retrieval import evaluate_rankings
from localml_scholar.serialization import atomic_write_text


def evaluate_fixture_method(
    *,
    method: str,
    output_directory: str | Path,
    generative_answerer: GroundedGenerativeAnswerer | None = None,
    retrieval_method: str = "bm25",
    retrieval_top_k: int = 8,
    evidence_top_k: int = 4,
) -> dict[str, Any]:
    """Run one controlled answer method on the authored fixture and persist it."""
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    index = build_grounded_fixture_index()
    questions = load_grounded_fixture_questions()
    index_path = index.save(destination / "fixture_index.json")
    config = EvidenceSelectionConfig(
        retrieval_method=retrieval_method,
        retrieval_top_k=retrieval_top_k,
        evidence_top_k=evidence_top_k,
    )
    pipeline = GroundedAnswerPipeline(
        index,
        evidence_config=config,
        generative_answerer=generative_answerer,
    )
    answers = {}
    latencies: list[float] = []
    rankings: dict[str, list[str]] = {}
    relevance: dict[str, list[str]] = {}
    for question in questions:
        search_results = index.search(
            question.question,
            method=retrieval_method,
            top_k=retrieval_top_k,
        )
        if question.relevant_chunk_ids:
            rankings[question.question_id] = [
                result.chunk_id for result in search_results
            ]
            relevance[question.question_id] = list(question.relevant_chunk_ids)
        start = time.perf_counter()
        answers[question.question_id] = pipeline.answer(
            question.question,
            method=method,
        )
        latencies.append(time.perf_counter() - start)
    answer_evaluation = evaluate_grounded_answers(questions, answers)
    retrieval_evaluation = evaluate_rankings(
        rankings,
        relevance,
        valid_chunk_ids={chunk.chunk_id for chunk in index.chunks},
    )
    generation_attempts = [
        answer for answer in answers.values() if answer.raw_generated_text is not None
    ]
    rejected_generations = [
        answer
        for answer in generation_attempts
        if answer.fallback_used or not answer.validation.accepted
    ]
    failures = [
        {
            "question_id": question.question_id,
            "expected_answerable": question.expected_answerable,
            "abstained": answers[question.question_id].abstained,
            "accepted": answers[question.question_id].validation.accepted,
            "rejection_reasons": list(
                answers[question.question_id].validation.rejection_reasons
            ),
        }
        for question in questions
        if (
            answers[question.question_id].abstained == question.expected_answerable
            or not answers[question.question_id].validation.accepted
            or answer_evaluation.per_question[question.question_id].key_fact_recall
            < 1.0
        )
    ]
    summary_path = destination / f"{method}_evaluation.json"
    summary: dict[str, Any] = {
        "milestone": 9,
        "package_version": __version__,
        "purpose": "authored-fixture implementation validation",
        "claim_boundary": (
            "Lexical and authored-fixture metrics do not prove semantic "
            "correctness or general paper-question-answering capability."
        ),
        "method": method,
        "retrieval_configuration": config.to_dict(),
        "index": {
            "path": str(index_path),
            "index_sha256": index.index_sha256,
            "corpus_sha256": index.corpus_sha256,
            "documents": len(index.documents),
            "chunks": len(index.chunks),
        },
        "retrieval_evaluation_answerable_questions": (retrieval_evaluation.to_dict()),
        "answer_evaluation": answer_evaluation.to_dict(),
        "mean_answer_latency_seconds": statistics.fmean(latencies),
        "answer_latencies_seconds": latencies,
        "fallback_rate": sum(answer.fallback_used for answer in answers.values())
        / len(answers),
        "accepted_rate": sum(answer.validation.accepted for answer in answers.values())
        / len(answers),
        "generation_attempt_count": len(generation_attempts),
        "generative_acceptance_rate": (
            0.0
            if not generation_attempts
            else (len(generation_attempts) - len(rejected_generations))
            / len(generation_attempts)
        ),
        "generative_rejection_rate": (
            0.0
            if not generation_attempts
            else len(rejected_generations) / len(generation_attempts)
        ),
        "failure_cases": failures,
        "answers": {
            question_id: answer.to_dict()
            for question_id, answer in sorted(answers.items())
        },
        "artifacts": {
            "index": str(index_path),
            "summary": str(summary_path),
        },
    }
    atomic_write_text(
        summary_path,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    return summary
