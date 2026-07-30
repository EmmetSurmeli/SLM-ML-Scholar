"""Question-level evaluation-run regression comparison."""

from __future__ import annotations

from typing import Any

from localml_scholar.evaluation.schemas import EvaluationRun


def _quality_score(result) -> float:
    values = [
        result.retrieval.recall_at_5,
        result.retrieval.expected_section_hit,
    ]
    if result.sufficiency is not None:
        values.append(result.sufficiency.correct)
    if result.answer is not None:
        values.extend([result.answer.relevance, result.answer.completeness])
    if result.citations is not None:
        values.extend(
            [
                result.citations.relevance_rate,
                result.citations.support_rate,
                result.citations.coverage,
            ]
        )
    if result.audience is not None:
        values.append(result.audience.appropriateness)
    return sum(values) / len(values)


def compare_evaluation_runs(
    baseline: EvaluationRun,
    candidate: EvaluationRun,
) -> dict[str, Any]:
    """Report metric deltas and question-level changes without significance claims."""
    if not isinstance(baseline, EvaluationRun) or not isinstance(
        candidate, EvaluationRun
    ):
        raise TypeError("baseline and candidate must be EvaluationRun objects.")
    if baseline.benchmark_sha256 != candidate.benchmark_sha256:
        raise ValueError("Runs must use the same benchmark.")
    baseline_by_id = {item.question_id: item for item in baseline.question_results}
    candidate_by_id = {item.question_id: item for item in candidate.question_results}
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("Runs must evaluate the same question IDs.")
    metric_names = sorted(
        set(baseline.aggregate_metrics) | set(candidate.aggregate_metrics)
    )
    metric_deltas = {
        name: {
            "baseline": baseline.aggregate_metrics.get(name, 0.0),
            "candidate": candidate.aggregate_metrics.get(name, 0.0),
            "delta": candidate.aggregate_metrics.get(name, 0.0)
            - baseline.aggregate_metrics.get(name, 0.0),
        }
        for name in metric_names
    }
    improved: list[str] = []
    regressed: list[str] = []
    changes: list[dict[str, Any]] = []
    new_failures: dict[str, list[str]] = {}
    resolved_failures: dict[str, list[str]] = {}
    for question_id in sorted(baseline_by_id):
        old = baseline_by_id[question_id]
        new = candidate_by_id[question_id]
        delta = _quality_score(new) - _quality_score(old)
        if delta > 1e-12:
            improved.append(question_id)
        elif delta < -1e-12:
            regressed.append(question_id)
        added = sorted(set(new.failure_categories) - set(old.failure_categories))
        removed = sorted(set(old.failure_categories) - set(new.failure_categories))
        if added:
            new_failures[question_id] = added
        if removed:
            resolved_failures[question_id] = removed
        old_answer = (
            None if old.system_answer is None else old.system_answer.get("answer_text")
        )
        new_answer = (
            None if new.system_answer is None else new.system_answer.get("answer_text")
        )
        old_evidence = [item["chunk_id"] for item in old.retrieval_results]
        new_evidence = [item["chunk_id"] for item in new.retrieval_results]
        old_citations = (
            [] if old.system_answer is None else old.system_answer.get("citations", [])
        )
        new_citations = (
            [] if new.system_answer is None else new.system_answer.get("citations", [])
        )
        if (
            old_answer != new_answer
            or old_evidence != new_evidence
            or old_citations != new_citations
            or added
            or removed
        ):
            changes.append(
                {
                    "question_id": question_id,
                    "quality_delta": delta,
                    "answer_changed": old_answer != new_answer,
                    "evidence_changed": old_evidence != new_evidence,
                    "citation_changed": old_citations != new_citations,
                    "baseline_evidence": old_evidence,
                    "candidate_evidence": new_evidence,
                    "baseline_citations": old_citations,
                    "candidate_citations": new_citations,
                    "new_failures": added,
                    "resolved_failures": removed,
                }
            )
    return {
        "comparison_format_version": 1,
        "benchmark_sha256": baseline.benchmark_sha256,
        "baseline": {
            "run_id": baseline.run_id,
            "package_version": baseline.package_version,
            "configuration": baseline.configuration.to_dict(),
            "run_sha256": baseline.run_sha256,
        },
        "candidate": {
            "run_id": candidate.run_id,
            "package_version": candidate.package_version,
            "configuration": candidate.configuration.to_dict(),
            "run_sha256": candidate.run_sha256,
        },
        "metric_deltas": metric_deltas,
        "improved_questions": improved,
        "regressed_questions": regressed,
        "new_failures": new_failures,
        "resolved_failures": resolved_failures,
        "question_changes": changes,
        "statistical_significance": (
            "not_computed; no independence or sample-size assumptions were justified"
        ),
    }
