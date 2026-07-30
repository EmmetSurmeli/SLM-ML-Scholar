from __future__ import annotations

import json
from dataclasses import replace

from experiments.analyze_failure_taxonomy import main as analyze_failures
from experiments.evaluate_real_paper_benchmark import main as evaluate_real
from localml_scholar.answering import GroundedAnswerPipeline
from localml_scholar.evaluation import aggregate_human_review_metrics
from localml_scholar.evaluation.answer_grading import grade_sufficiency
from localml_scholar.evaluation.cli import main as evaluation_cli
from localml_scholar.evaluation.failures import (
    categorize_failures,
    is_automatic_pass,
    root_cause_attribution,
)
from localml_scholar.evaluation.human_review import (
    build_review_queue,
    record_human_review,
)
from localml_scholar.evaluation.schemas import (
    AnswerGrade,
    AudienceGrade,
    CitationGrade,
    ConceptCoverage,
)
from localml_scholar.evaluation.serialization import (
    load_evaluation_run,
    save_benchmark,
)
from tests.test_real_paper_evaluation_integration import _run
from tests.test_real_paper_evaluation_schemas import (
    evaluation_benchmark,
    evaluation_question,
)


def _answer_grade(**changes):
    values = {
        "relevance": 1.0,
        "completeness": 1.0,
        "answerability_correct": 1.0,
        "numerical_accuracy": 1.0,
        "prohibited_claim_count": 0,
        "false_premise_accepted": 0.0,
        "wrong_entity_type": 0.0,
        "missing_requirements": (),
        "relevance_reasons": (),
    }
    values.update(changes)
    return AnswerGrade(**values)


def _concept_grade(**changes):
    values = {
        "required_recall": 1.0,
        "optional_recall": 1.0,
        "present_required": (),
        "missing_required": (),
        "present_optional": (),
        "uncited_required": (),
        "unsupported_required": (),
    }
    values.update(changes)
    return ConceptCoverage(**values)


def _citation_grade(**changes):
    values = {
        "syntax_valid": 1.0,
        "existence_valid": 1.0,
        "source_location_correct": 1.0,
        "support_rate": 1.0,
        "relevance_rate": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "coverage": 1.0,
        "wrong_section_count": 0,
        "boilerplate_count": 0,
    }
    values.update(changes)
    return CitationGrade(**values)


def _audience_grade(level="undergraduate", appropriateness=1.0):
    return AudienceGrade(
        audience_level=level,
        appropriateness=appropriateness,
        jargon_density=0.0,
        average_sentence_words=10.0,
        equation_count=0,
        definition_present=1.0,
        mechanism_present=1.0,
        qualification_present=1.0,
        limitation_present=1.0,
        reasons=(),
        requires_human_review=False,
    )


def test_failure_taxonomy_is_multi_label_and_deterministic(grounded_index):
    question = evaluation_question(grounded_index)
    _, run = _run(grounded_index, mode="retrieval_only")
    retrieval = replace(
        run.question_results[0].retrieval,
        wrong_section_chunk_ids=("wrong",),
        missed_gold_chunk_ids=("missing",),
        boilerplate_chunk_ids=("boilerplate",),
        evidence_redundancy=0.5,
        irrelevant_positive_score_rate=1.0,
    )
    args = {
        "answer_object": None,
        "sufficiency": None,
        "answer": _answer_grade(
            relevance=0.0,
            completeness=0.0,
            numerical_accuracy=0.0,
            prohibited_claim_count=1,
            wrong_entity_type=1.0,
        ),
        "concepts": _concept_grade(missing_required=("mask",)),
        "citations": _citation_grade(
            relevance_rate=0.0,
            coverage=0.0,
            recall=0.0,
            wrong_section_count=1,
        ),
        "audience": _audience_grade(appropriateness=0.0),
    }
    first = categorize_failures(question, retrieval, **args)
    second = categorize_failures(question, retrieval, **args)
    assert first == second
    assert {
        "retrieval_wrong_section",
        "retrieval_missed_gold_evidence",
        "answer_not_relevant",
        "answer_incomplete",
        "prohibited_claim_present",
        "citation_irrelevant",
        "required_concept_missing",
        "audience_too_shallow",
    } <= set(first)


def test_root_cause_reports_ambiguity_without_certainty():
    cause = root_cause_attribution(
        ("retrieval_wrong_section", "answer_incomplete", "citation_irrelevant")
    )
    assert cause.primary_cause == "retrieval"
    assert cause.secondary_causes == (
        "answer_construction",
        "citation_validation",
    )
    assert cause.confidence == "medium"
    assert root_cause_attribution(()).primary_cause == "none"


def test_external_context_and_correct_abstention(grounded_index):
    answer = GroundedAnswerPipeline(grounded_index).answer(
        "What later external historical event changed quantum topology?",
        method="extractive",
    )
    question = evaluation_question(
        grounded_index,
        question="What later work resulted from this paper?",
        question_type="historical_impact",
        answerability="external_sources_required",
        paper_sufficiency="external_required",
        gold_evidence=(),
        required_concepts=(),
        review_status="approved",
    )
    sufficiency = grade_sufficiency(question, answer)
    _, run = _run(grounded_index, mode="retrieval_only")
    categories = categorize_failures(
        question,
        run.question_results[0].retrieval,
        answer_object=answer,
        sufficiency=sufficiency,
        answer=None,
        concepts=None,
        citations=None,
        audience=None,
    )
    assert "external_context_required" in categories
    if answer.abstained and sufficiency.correct:
        assert "correct_abstention" in categories
        assert is_automatic_pass(("correct_abstention",))


def test_human_review_metrics_do_not_impute_pending(grounded_index):
    benchmark, run = _run(grounded_index)
    queue = build_review_queue(
        run,
        benchmark,
        pass_sample_fraction=1.0,
    )
    assert aggregate_human_review_metrics(queue)["reviewed_count"] == 0.0
    reviewed = record_human_review(
        queue,
        queue[0].review_id,
        reviewer_label="benchmark_problem",
        reviewer_notes="Question is ambiguous.",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    metrics = aggregate_human_review_metrics(reviewed)
    assert metrics["reviewed_count"] == 1.0
    assert metrics["human_reviewed_pass_rate"] == 0.0
    assert metrics["benchmark_disagreement_rate"] == 1.0


def test_cli_run_report_queue_and_compare(grounded_index, tmp_path):
    index_path = grounded_index.save(tmp_path / "index.json")
    benchmark = evaluation_benchmark(grounded_index)
    benchmark_path = save_benchmark(benchmark, tmp_path / "benchmark.json")
    run_path = tmp_path / "run.json"
    assert (
        evaluation_cli(
            [
                "run",
                "--benchmark",
                str(benchmark_path),
                "--index",
                str(index_path),
                "--method",
                "retrieval_only",
                "--output",
                str(run_path),
            ]
        )
        == 0
    )
    report_path = tmp_path / "report.md"
    assert (
        evaluation_cli(
            [
                "report",
                "--run",
                str(run_path),
                "--benchmark",
                str(benchmark_path),
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    queue_path = tmp_path / "queue.json"
    assert (
        evaluation_cli(
            [
                "build-review-queue",
                "--run",
                str(run_path),
                "--benchmark",
                str(benchmark_path),
                "--pass-sample-fraction",
                "1",
                "--output",
                str(queue_path),
            ]
        )
        == 0
    )
    comparison_path = tmp_path / "comparison.json"
    assert (
        evaluation_cli(
            [
                "compare",
                "--baseline",
                str(run_path),
                "--candidate",
                str(run_path),
                "--output",
                str(comparison_path),
            ]
        )
        == 0
    )
    assert load_evaluation_run(run_path).configuration.mode == "retrieval_only"
    assert "# Evaluation summary" in report_path.read_text(encoding="utf-8")
    assert queue_path.exists() and comparison_path.exists()


def test_attention_starter_cli_is_unapproved(grounded_index, tmp_path):
    index_path = grounded_index.save(tmp_path / "index.json")
    document_id = next(
        item.document_id
        for item in grounded_index.documents
        if item.title == "Causal Attention"
    )
    output = tmp_path / "attention.json"
    assert (
        evaluation_cli(
            [
                "attention-starter",
                "--index",
                str(index_path),
                "--document-id",
                document_id,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert all(
        item["review_status"] == "proposed" for item in payload["payload"]["questions"]
    )


def test_cli_rejects_generative_run_without_checkpoint(
    grounded_index,
    tmp_path,
    capsys,
):
    index_path = grounded_index.save(tmp_path / "index.json")
    benchmark_path = save_benchmark(
        evaluation_benchmark(grounded_index),
        tmp_path / "benchmark.json",
    )
    assert (
        evaluation_cli(
            [
                "run",
                "--benchmark",
                str(benchmark_path),
                "--index",
                str(index_path),
                "--method",
                "generative",
                "--output",
                str(tmp_path / "run.json"),
            ]
        )
        == 2
    )
    assert "require --checkpoint" in capsys.readouterr().err.casefold()


def test_real_paper_experiment_refuses_to_fabricate(capsys):
    assert evaluate_real([]) == 2
    assert "will not fabricate" in capsys.readouterr().err


def test_failure_analysis_experiment_uses_saved_run(
    grounded_index,
    tmp_path,
):
    _, run = _run(grounded_index)
    from localml_scholar.evaluation.serialization import save_evaluation_run

    run_path = save_evaluation_run(run, tmp_path / "run.json")
    output = tmp_path / "analysis.json"
    assert analyze_failures(["--run", str(run_path), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["run_id"] == run.run_id
    assert "review_priority_question_ids" in payload
