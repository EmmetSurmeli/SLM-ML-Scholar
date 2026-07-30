"""Human-readable Markdown reports for benchmarks, failures, reviews, and runs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from localml_scholar.evaluation.runner import grouped_metrics
from localml_scholar.evaluation.schemas import (
    Benchmark,
    CorrectionExample,
    EvaluationRun,
    HumanReviewRecord,
)


def _number(value: float) -> str:
    return f"{value:.4f}"


def render_benchmark_review(benchmark: Benchmark) -> str:
    """Render proposed and approved status without conflating candidates with gold."""
    if not isinstance(benchmark, Benchmark):
        raise TypeError("benchmark must be Benchmark.")
    counts = Counter(item.review_status for item in benchmark.questions)
    lines = [
        f"# Benchmark review: {benchmark.name}",
        "",
        f"- Version: `{benchmark.benchmark_version}`",
        f"- Benchmark hash: `{benchmark.benchmark_sha256}`",
        f"- Total questions: {len(benchmark.questions)}",
        f"- Approved/edited: {len(benchmark.approved_questions)}",
        f"- Proposed: {counts['proposed']}",
        f"- Rejected: {counts['rejected']}",
        "",
        "> Proposed questions are candidates, not trusted gold data.",
        "",
    ]
    for item in benchmark.questions:
        concepts = (
            ", ".join(value.concept for value in item.required_concepts) or "none"
        )
        lines.extend(
            [
                f"## {item.question_id}: {item.question}",
                "",
                f"- Status: **{item.review_status}**",
                f"- Type: `{item.question_type}`",
                f"- Audience: `{item.audience_level}`",
                f"- Answerability: `{item.answerability}`",
                f"- Paper sufficiency: `{item.paper_sufficiency}`",
                f"- Gold chunks: {', '.join(item.relevant_chunk_ids) or 'none'}",
                f"- Required concepts: {concepts}",
                f"- Gold notes: {item.gold_notes or 'awaiting human review'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _failure_example(result, question) -> list[str]:
    answer_text = (
        "Retrieval-only run."
        if result.system_answer is None
        else result.system_answer.get("answer_text", "Malformed answer artifact.")
    )
    evidence = [
        f"`{item['chunk_id']}` — "
        f"{item.get('citation', {}).get('display', 'location unavailable')}"
        for item in result.retrieval_results
    ]
    missing = [] if result.concepts is None else list(result.concepts.missing_required)
    return [
        f"### {question.question_id}: {question.question}",
        "",
        f"- Audience: `{question.audience_level}`",
        f"- Gold answerability: `{question.answerability}`",
        f"- Answer: {answer_text}",
        f"- Retrieved evidence: {', '.join(evidence) or 'none'}",
        f"- Gold evidence: {', '.join(question.relevant_chunk_ids) or 'none'}",
        f"- Missing concepts: {', '.join(missing) or 'none'}",
        f"- Failures: {', '.join(result.failure_categories) or 'none'}",
        f"- Likely root cause: `{result.root_cause.primary_cause}` "
        f"({result.root_cause.confidence} confidence)",
        f"- Suggested correction: {result.root_cause.recommended_next_action}",
        "",
    ]


def render_failure_report(run: EvaluationRun, benchmark: Benchmark) -> str:
    """Show every automatic failure with evidence and correction direction."""
    if not isinstance(run, EvaluationRun) or not isinstance(benchmark, Benchmark):
        raise TypeError("run and benchmark must use evaluation schemas.")
    question_by_id = {item.question_id: item for item in benchmark.questions}
    failing = [item for item in run.question_results if not item.automatic_pass]
    counts = Counter(
        category for item in failing for category in item.failure_categories
    )
    lines = [
        "# Failure report",
        "",
        f"- Run: `{run.run_id}`",
        f"- Failed examples: {len(failing)} / {len(run.question_results)}",
        "",
        "## Failure frequencies",
        "",
    ]
    lines.extend(
        f"- `{category}`: {count}" for category, count in sorted(counts.items())
    )
    lines.append("")
    for result in failing:
        lines.extend(_failure_example(result, question_by_id[result.question_id]))
    return "\n".join(lines).rstrip() + "\n"


def render_evaluation_summary(run: EvaluationRun, benchmark: Benchmark) -> str:
    """Render actual system, paper, question-type, and failure metrics."""
    if not isinstance(run, EvaluationRun) or not isinstance(benchmark, Benchmark):
        raise TypeError("run and benchmark must use evaluation schemas.")
    metrics = run.aggregate_metrics
    lines = [
        f"# Evaluation summary: {benchmark.name}",
        "",
        f"- Run ID: `{run.run_id}`",
        f"- Papers: {int(metrics.get('paper_count', 0.0))}",
        f"- Questions: {int(metrics.get('question_count', 0.0))}",
        f"- Method: `{run.configuration.mode}`",
        f"- Retriever: `{run.configuration.retrieval_method}`",
        "",
        "## Retrieval",
        "",
        f"- Recall@1: {_number(metrics.get('retrieval.recall_at_1', 0.0))}",
        f"- Recall@3: {_number(metrics.get('retrieval.recall_at_3', 0.0))}",
        f"- Recall@5: {_number(metrics.get('retrieval.recall_at_5', 0.0))}",
        f"- MRR: {_number(metrics.get('retrieval.mrr', 0.0))}",
        f"- Expected-section hit rate: "
        f"{_number(metrics.get('retrieval.expected_section_hit_rate', 0.0))}",
        f"- Forbidden-section rate: "
        f"{_number(metrics.get('retrieval.forbidden_section_rate', 0.0))}",
        f"- Boilerplate rate: "
        f"{_number(metrics.get('retrieval.boilerplate_rate', 0.0))}",
        "",
        "## Sufficiency and answering",
        "",
        f"- Sufficiency accuracy: {_number(metrics.get('sufficiency.accuracy', 0.0))}",
        f"- False-answer rate: "
        f"{_number(metrics.get('sufficiency.false_answer_rate', 0.0))}",
        f"- Answer relevance: {_number(metrics.get('answer.relevance', 0.0))}",
        f"- Required-concept recall: "
        f"{_number(metrics.get('answer.required_concept_recall', 0.0))}",
        f"- Completeness: {_number(metrics.get('answer.completeness', 0.0))}",
        "",
        "## Citations",
        "",
        f"- Validity: {_number(metrics.get('citation.validity', 0.0))}",
        f"- Support: {_number(metrics.get('citation.support', 0.0))}",
        f"- Relevance: {_number(metrics.get('citation.relevance', 0.0))}",
        f"- Coverage: {_number(metrics.get('citation.coverage', 0.0))}",
        "",
        "## Audience",
        "",
        f"- Beginner appropriateness: "
        f"{_number(metrics.get('audience.beginner_appropriateness', 0.0))}",
        f"- Undergraduate appropriateness: "
        f"{_number(metrics.get('audience.undergraduate_appropriateness', 0.0))}",
        f"- Researcher appropriateness: "
        f"{_number(metrics.get('audience.researcher_appropriateness', 0.0))}",
        f"- Cross-level factual consistency: "
        f"{_number(metrics.get('audience.factual_consistency', 0.0))}",
        "",
        "## Failures",
        "",
    ]
    failure_metrics = {
        key.removeprefix("failure."): value
        for key, value in metrics.items()
        if key.startswith("failure.")
    }
    lines.extend(
        f"- `{category}`: {int(count)}"
        for category, count in sorted(failure_metrics.items())
    )
    if not failure_metrics:
        lines.append("- No automatic failures.")
    lines.extend(["", "## Per paper", ""])
    for paper_id, values in grouped_metrics(
        run.question_results, key="paper_id"
    ).items():
        paper_results = [
            item for item in run.question_results if item.paper_id == paper_id
        ]
        wrong_sections = Counter(
            " / ".join(result.get("heading_path", ())) or "(document root)"
            for item in paper_results
            for result in item.retrieval_results
            if result["chunk_id"] in item.retrieval.wrong_section_chunk_ids
        )
        missing = Counter(
            concept
            for item in paper_results
            if item.concepts is not None
            for concept in item.concepts.missing_required
        )
        backlog = sum(
            not item.automatic_pass or item.automated_confidence == "low"
            for item in paper_results
        )
        common_wrong = (
            ", ".join(key for key, _ in wrong_sections.most_common(3)) or "none"
        )
        question_types = Counter(item.question_type for item in paper_results)
        audiences = Counter(item.audience_level for item in paper_results)
        failures = Counter(
            category for item in paper_results for category in item.failure_categories
        )
        ordered = sorted(
            paper_results,
            key=lambda item: (
                item.automatic_pass,
                item.retrieval.recall_at_5,
                0.0 if item.answer is None else item.answer.relevance,
                item.question_id,
            ),
        )
        type_summary = ", ".join(
            f"{key}={value}" for key, value in sorted(question_types.items())
        )
        audience_summary = ", ".join(
            f"{key}={value}" for key, value in sorted(audiences.items())
        )
        lines.extend(
            [
                f"### `{paper_id}`",
                "",
                f"- Questions: {len(paper_results)}",
                f"- Question types: {type_summary}",
                f"- Audiences: {audience_summary}",
                f"- Recall@5: {_number(values.get('retrieval.recall_at_5', 0.0))}",
                f"- Answer relevance: {_number(values.get('answer.relevance', 0.0))}",
                f"- Citation relevance: "
                f"{_number(values.get('citation.relevance', 0.0))}",
                f"- Common wrong chunks: {common_wrong}",
                f"- Common missing concepts: "
                f"{', '.join(key for key, _ in missing.most_common(3)) or 'none'}",
                f"- Common failures: "
                f"{', '.join(key for key, _ in failures.most_common(3)) or 'none'}",
                f"- Lowest-scoring example: `{ordered[0].question_id}`",
                f"- Highest-scoring example: `{ordered[-1].question_id}`",
                f"- Human-review backlog: {backlog}",
                "",
            ]
        )
    lines.extend(["## Per question type", ""])
    for question_type, values in grouped_metrics(
        run.question_results, key="question_type"
    ).items():
        lines.append(
            f"- `{question_type}`: "
            f"Recall@5 {_number(values.get('retrieval.recall_at_5', 0.0))}, "
            f"relevance {_number(values.get('answer.relevance', 0.0))}"
        )
    lines.extend(
        [
            "",
            "> These deterministic heuristics expose likely failures; they do not "
            "prove semantic correctness.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_review_queue(records: tuple[HumanReviewRecord, ...]) -> str:
    if not isinstance(records, tuple) or not all(
        isinstance(item, HumanReviewRecord) for item in records
    ):
        raise TypeError("records must contain HumanReviewRecord objects.")
    lines = ["# Human review queue", "", f"- Records: {len(records)}", ""]
    for item in records:
        answer_text = (
            "Retrieval-only"
            if item.system_answer is None
            else item.system_answer.get("answer_text", "Malformed")
        )
        lines.extend(
            [
                f"## {item.review_id}",
                "",
                f"- Question ID: `{item.question_id}`",
                f"- Failures: {', '.join(item.failure_categories) or 'sampled pass'}",
                f"- Reviewer label: {item.reviewer_label or 'pending'}",
                f"- Answer: {answer_text}",
                f"- Notes: {item.reviewer_notes or 'pending'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_comparison_report(report: dict[str, Any]) -> str:
    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary.")
    lines = [
        "# Regression comparison",
        "",
        f"- Baseline: `{report['baseline']['run_id']}`",
        f"- Candidate: `{report['candidate']['run_id']}`",
        f"- Improved questions: {len(report['improved_questions'])}",
        f"- Regressed questions: {len(report['regressed_questions'])}",
        f"- Significance: {report['statistical_significance']}",
        "",
        "## Metric deltas",
        "",
    ]
    for name, values in sorted(report["metric_deltas"].items()):
        lines.append(
            f"- `{name}`: {_number(values['baseline'])} → "
            f"{_number(values['candidate'])} "
            f"({values['delta']:+.4f})"
        )
    lines.extend(["", "## Regressed questions", ""])
    lines.extend(f"- `{question_id}`" for question_id in report["regressed_questions"])
    if not report["regressed_questions"]:
        lines.append("- None")
    return "\n".join(lines).rstrip() + "\n"


def render_correction_preview(
    examples: tuple[CorrectionExample, ...],
) -> str:
    if not isinstance(examples, tuple) or not all(
        isinstance(item, CorrectionExample) for item in examples
    ):
        raise TypeError("examples must contain CorrectionExample objects.")
    lines = [
        "# Human-approved correction dataset preview",
        "",
        f"- Examples: {len(examples)}",
        "",
    ]
    for item in examples:
        lines.extend(
            [
                f"## {item.question}",
                "",
                f"- Audience: `{item.audience_level}`",
                f"- Paper: `{item.paper_id}`",
                f"- Gold evidence: {', '.join(item.gold_evidence)}",
                f"- Failures: {', '.join(item.failure_categories)}",
                f"- Incorrect answer: {item.incorrect_answer}",
                f"- Corrected answer: {item.corrected_answer}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
