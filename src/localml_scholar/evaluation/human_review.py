"""Deterministic selective review queues and approved correction export."""

from __future__ import annotations

import random
from dataclasses import replace
from datetime import UTC, datetime

from localml_scholar.answering.citations import (
    CitationSyntaxError,
    citation_labels,
    strip_inline_citations,
)
from localml_scholar.evaluation.schemas import (
    Benchmark,
    CitedAnswerPoint,
    CorrectionExample,
    EvaluationRun,
    HumanReviewRecord,
    StructuredAnswerTarget,
    question_evaluation_sha256,
)
from localml_scholar.retrieval import RetrievalIndex
from localml_scholar.retrieval.documents import stable_identifier


def build_review_queue(
    run: EvaluationRun,
    benchmark: Benchmark,
    *,
    pass_sample_fraction: float = 0.1,
    random_seed: int = 0,
    disagreement_question_ids: tuple[str, ...] = (),
) -> tuple[HumanReviewRecord, ...]:
    """Include all risky cases plus a deterministic random sample of passes."""
    if not isinstance(run, EvaluationRun):
        raise TypeError("run must be EvaluationRun.")
    if not isinstance(benchmark, Benchmark):
        raise TypeError("benchmark must be Benchmark.")
    if (
        isinstance(pass_sample_fraction, bool)
        or not isinstance(pass_sample_fraction, (int, float))
        or not 0.0 <= pass_sample_fraction <= 1.0
    ):
        raise ValueError("pass_sample_fraction must lie in [0, 1].")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("random_seed must be an integer.")
    if not isinstance(disagreement_question_ids, tuple):
        raise TypeError("disagreement_question_ids must be a tuple.")
    question_by_id = {item.question_id: item for item in benchmark.questions}
    if any(item.question_id not in question_by_id for item in run.question_results):
        raise ValueError("Evaluation run contains a question absent from benchmark.")
    if not set(disagreement_question_ids) <= set(question_by_id):
        raise ValueError("Disagreement IDs must exist in the benchmark.")
    always_review: list = []
    passes: list = []
    for result in sorted(run.question_results, key=lambda item: item.question_id):
        question = question_by_id[result.question_id]
        risky = (
            not result.automatic_pass
            or result.automated_confidence == "low"
            or result.question_id in disagreement_question_ids
            or question.question_type
            in {
                "historical_impact",
                "synthesis",
                "interpretation",
            }
            or question.answerability == "ambiguous"
            or "answer_wrong_number" in result.failure_categories
            or (
                result.citations is not None
                and result.citations.existence_valid
                and result.citations.relevance_rate < 1.0
            )
        )
        (always_review if risky else passes).append(result)
    rng = random.Random(random_seed)
    sample_count = int(len(passes) * float(pass_sample_fraction))
    if pass_sample_fraction > 0.0 and passes and sample_count == 0:
        sample_count = 1
    sampled_ids = {
        item.question_id for item in rng.sample(passes, min(sample_count, len(passes)))
    }
    selected = always_review + [
        item for item in passes if item.question_id in sampled_ids
    ]
    records = []
    for result in sorted(selected, key=lambda item: item.question_id):
        answer = result.system_answer
        evidence = () if answer is None else tuple(answer.get("evidence", ()))
        records.append(
            HumanReviewRecord(
                review_id=stable_identifier(
                    "review",
                    run.run_id,
                    result.question_id,
                    question_evaluation_sha256(result),
                ),
                run_id=run.run_id,
                question_id=result.question_id,
                question_evaluation_sha256=question_evaluation_sha256(result),
                system_answer=answer,
                evidence=evidence,
                automatic_grades=result.to_dict(),
                failure_categories=result.failure_categories,
            )
        )
    return tuple(records)


def record_human_review(
    records: tuple[HumanReviewRecord, ...],
    review_id: str,
    *,
    reviewer_label: str,
    reviewer_notes: str,
    corrected_answer: str | None = None,
    corrected_evidence: tuple[str, ...] = (),
    timestamp: str | None = None,
) -> tuple[HumanReviewRecord, ...]:
    """Apply one review once; duplicate adjudication is rejected."""
    if not isinstance(records, tuple) or not all(
        isinstance(item, HumanReviewRecord) for item in records
    ):
        raise TypeError("records must contain HumanReviewRecord objects.")
    matches = [item for item in records if item.review_id == review_id]
    if len(matches) != 1:
        raise ValueError("review_id must identify exactly one review record.")
    if matches[0].reviewed:
        raise ValueError("This review record has already been adjudicated.")
    resolved_timestamp = timestamp or datetime.now(UTC).isoformat(timespec="seconds")
    updated = replace(
        matches[0],
        reviewer_label=reviewer_label,
        reviewer_notes=reviewer_notes,
        corrected_answer=corrected_answer,
        corrected_evidence=corrected_evidence,
        timestamp=resolved_timestamp,
    )
    return tuple(updated if item.review_id == review_id else item for item in records)


def export_approved_corrections(
    benchmark: Benchmark,
    index: RetrievalIndex,
    records: tuple[HumanReviewRecord, ...],
) -> tuple[CorrectionExample, ...]:
    """Export only human-reviewed, corrected, exact-evidence examples."""
    if not isinstance(benchmark, Benchmark):
        raise TypeError("benchmark must be Benchmark.")
    if not isinstance(index, RetrievalIndex):
        raise TypeError("index must be RetrievalIndex.")
    benchmark.validate_against_index(index)
    if not isinstance(records, tuple) or not all(
        isinstance(item, HumanReviewRecord) for item in records
    ):
        raise TypeError("records must contain HumanReviewRecord objects.")
    question_by_id = {item.question_id: item for item in benchmark.approved_questions}
    documents = {item.document_id: item for item in index.documents}
    corrections: list[CorrectionExample] = []
    for record in sorted(records, key=lambda item: item.review_id):
        question = question_by_id.get(record.question_id)
        if (
            question is None
            or record.reviewer_label
            not in {"correct", "partially_correct", "incorrect"}
            or record.corrected_answer is None
            or not record.corrected_evidence
            or record.system_answer is None
        ):
            continue
        valid_gold = set(question.relevant_chunk_ids)
        if not set(record.corrected_evidence) <= valid_gold:
            raise ValueError("Corrected evidence must use approved gold chunk IDs.")
        labels = tuple(
            f"C{position}"
            for position, _ in enumerate(record.corrected_evidence, start=1)
        )
        try:
            corrected_labels = citation_labels(record.corrected_answer)
        except CitationSyntaxError as error:
            raise ValueError(
                "Corrected answers must contain valid inline citations."
            ) from error
        if set(corrected_labels) != set(labels):
            raise ValueError(
                "Corrected answer citations must cover the corrected evidence."
            )
        core_text = (
            question.gold_core_answer
            or strip_inline_citations(record.corrected_answer).strip()
        )
        target = StructuredAnswerTarget(
            core_answer=CitedAnswerPoint(text=core_text, citations=labels)
        )
        answer_text = record.system_answer.get("answer_text")
        if not isinstance(answer_text, str) or not answer_text:
            raise ValueError("Reviewed system answer is malformed.")
        corrections.append(
            CorrectionExample(
                question=question.question,
                audience_level=question.audience_level,
                gold_evidence=record.corrected_evidence,
                structured_answer_target=target,
                corrected_answer=record.corrected_answer,
                incorrect_answer=answer_text,
                failure_categories=record.failure_categories,
                citations=labels,
                paper_id=question.paper_id,
                source_sha256=documents[question.paper_id].content_sha256,
                human_review_id=record.review_id,
            )
        )
    return tuple(corrections)


def aggregate_human_review_metrics(
    records: tuple[HumanReviewRecord, ...],
) -> dict[str, float]:
    """Summarize adjudicated review outcomes without imputing pending labels."""
    if not isinstance(records, tuple) or not all(
        isinstance(item, HumanReviewRecord) for item in records
    ):
        raise TypeError("records must contain HumanReviewRecord objects.")
    reviewed = tuple(item for item in records if item.reviewed)
    if not reviewed:
        return {
            "reviewed_count": 0.0,
            "human_reviewed_pass_rate": 0.0,
            "benchmark_disagreement_rate": 0.0,
        }
    return {
        "reviewed_count": float(len(reviewed)),
        "human_reviewed_pass_rate": (
            sum(item.reviewer_label == "correct" for item in reviewed) / len(reviewed)
        ),
        "benchmark_disagreement_rate": (
            sum(item.reviewer_label == "benchmark_problem" for item in reviewed)
            / len(reviewed)
        ),
    }
