from __future__ import annotations

from dataclasses import replace

import pytest

from localml_scholar.answering import (
    EvidenceSelectionConfig,
    GroundedAnswerPipeline,
)
from localml_scholar.evaluation import (
    EvaluationConfig,
    EvaluationRunner,
    aggregate_evaluation_metrics,
    attention_starter_question_count,
    build_review_queue,
    compare_evaluation_runs,
    export_approved_corrections,
    generate_attention_starter_benchmark,
    grouped_metrics,
    record_human_review,
)
from localml_scholar.evaluation.reports import (
    render_benchmark_review,
    render_comparison_report,
    render_correction_preview,
    render_evaluation_summary,
    render_failure_report,
    render_review_queue,
)
from localml_scholar.evaluation.serialization import (
    load_corrections,
    load_evaluation_run,
    load_review_records,
    save_corrections,
    save_evaluation_run,
    save_review_records,
)
from tests.test_real_paper_evaluation_schemas import (
    evaluation_benchmark,
    evaluation_question,
)


def _run(grounded_index, *, mode="extractive", top_k=5):
    benchmark = evaluation_benchmark(grounded_index)
    config = EvaluationConfig(
        mode=mode,
        top_k=top_k,
        retrieval_parameters={"top_k": top_k},
    )
    return benchmark, EvaluationRunner(
        benchmark,
        grounded_index,
        config,
    ).run()


def test_end_to_end_retrieval_only_and_extracting(grounded_index):
    benchmark, retrieval = _run(grounded_index, mode="retrieval_only")
    _, extractive = _run(grounded_index, mode="extractive")
    assert retrieval.question_results[0].system_answer is None
    assert extractive.question_results[0].system_answer is not None
    assert extractive.question_results[0].audience is not None
    assert extractive.benchmark_sha256 == benchmark.benchmark_sha256
    assert "retrieval.recall_at_5" in extractive.aggregate_metrics


def test_proposed_questions_cannot_be_run(grounded_index):
    proposed = evaluation_question(
        grounded_index,
        review_status="proposed",
        gold_notes=None,
    )
    benchmark = evaluation_benchmark(grounded_index, proposed)
    with pytest.raises(ValueError, match="human-approved"):
        EvaluationRunner(
            benchmark,
            grounded_index,
            EvaluationConfig(),
        )


def test_generative_mode_requires_explicit_pipeline(grounded_index):
    benchmark = evaluation_benchmark(grounded_index)
    with pytest.raises(ValueError, match="explicit local answering pipeline"):
        EvaluationRunner(
            benchmark,
            grounded_index,
            EvaluationConfig(
                mode="generative",
                model_checkpoint_sha256="1" * 64,
                tokenizer_sha256="2" * 64,
            ),
        )


def test_no_extractable_sentence_abstains_instead_of_crashing(
    grounded_index,
    monkeypatch,
):
    pipeline = GroundedAnswerPipeline(
        grounded_index,
        evidence_config=EvidenceSelectionConfig(
            retrieval_method="bm25",
            retrieval_top_k=5,
            evidence_top_k=4,
        ),
    )

    def no_sentence(*_args, **_kwargs):
        raise ValueError("No source sentence fits the extractive answer budget.")

    monkeypatch.setattr(pipeline.extractive_answerer, "answer", no_sentence)
    answer = pipeline.answer(
        "How does a causal mask prevent future-token leakage?",
        method="extractive",
    )
    assert answer.abstained
    assert (
        answer.metadata["answer_construction_abstention"]["reason"]
        == "no_extractable_source_sentence"
    )


def test_unexpected_extractive_errors_are_not_silently_caught(
    grounded_index,
    monkeypatch,
):
    pipeline = GroundedAnswerPipeline(grounded_index)

    def malformed(*_args, **_kwargs):
        raise ValueError("unexpected extraction defect")

    monkeypatch.setattr(pipeline.extractive_answerer, "answer", malformed)
    with pytest.raises(ValueError, match="unexpected extraction defect"):
        pipeline.answer(
            "How does a causal mask prevent future-token leakage?",
            method="extractive",
        )


def test_resume_reuses_exact_question_records(grounded_index):
    benchmark, run = _run(grounded_index)
    resumed = EvaluationRunner(
        benchmark,
        grounded_index,
        run.configuration,
    ).run(existing_run=run)
    assert resumed == run
    with pytest.raises(ValueError, match="incompatible"):
        EvaluationRunner(
            benchmark,
            grounded_index,
            replace(run.configuration, top_k=3),
        ).run(existing_run=run)


def test_evaluation_run_round_trip_accepts_recorded_older_version(
    grounded_index,
    tmp_path,
):
    _, run = _run(grounded_index)
    old = replace(run, package_version="1.1.0")
    path = save_evaluation_run(old, tmp_path / "run.json")
    assert load_evaluation_run(path) == old


def test_aggregate_and_grouped_metrics_are_exact(grounded_index):
    _, run = _run(grounded_index)
    result = run.question_results[0]
    metrics = aggregate_evaluation_metrics((result, result))
    assert metrics["question_count"] == 2.0
    assert metrics["retrieval.recall_at_5"] == result.retrieval.recall_at_5
    assert (
        grouped_metrics((result,), key="paper_id")[result.paper_id]["question_count"]
        == 1.0
    )
    assert aggregate_evaluation_metrics(()) == {
        "question_count": 0.0,
        "automatic_pass_rate": 0.0,
    }


def test_failure_review_queue_sampling_and_serialization(
    grounded_index,
    tmp_path,
):
    benchmark, run = _run(grounded_index)
    first = build_review_queue(
        run,
        benchmark,
        pass_sample_fraction=1.0,
        random_seed=7,
    )
    second = build_review_queue(
        run,
        benchmark,
        pass_sample_fraction=1.0,
        random_seed=7,
    )
    assert first == second
    assert first
    path = save_review_records(first, tmp_path / "queue.json")
    assert load_review_records(path) == first
    assert "# Human review queue" in render_review_queue(first)


def test_human_review_duplicate_protection_and_correction_export(
    grounded_index,
    tmp_path,
):
    benchmark, run = _run(grounded_index)
    queue = build_review_queue(
        run,
        benchmark,
        pass_sample_fraction=1.0,
    )
    record = queue[0]
    gold_chunk = benchmark.questions[0].relevant_chunk_ids[0]
    reviewed = record_human_review(
        queue,
        record.review_id,
        reviewer_label="partially_correct",
        reviewer_notes="Add the direct information-flow explanation.",
        corrected_answer="Later positions are blocked by the causal mask. [C1]",
        corrected_evidence=(gold_chunk,),
        timestamp="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="already"):
        record_human_review(
            reviewed,
            record.review_id,
            reviewer_label="correct",
            reviewer_notes="Duplicate.",
        )
    corrections = export_approved_corrections(
        benchmark,
        grounded_index,
        reviewed,
    )
    assert len(corrections) == 1
    assert corrections[0].source_sha256
    path = save_corrections(corrections, tmp_path / "corrections.json")
    assert load_corrections(path) == corrections
    assert "human-approved" in render_correction_preview(corrections).casefold()


def test_benchmark_problem_never_exports_training_example(grounded_index):
    benchmark, run = _run(grounded_index)
    queue = build_review_queue(
        run,
        benchmark,
        pass_sample_fraction=1.0,
    )
    reviewed = record_human_review(
        queue,
        queue[0].review_id,
        reviewer_label="benchmark_problem",
        reviewer_notes="Gold annotation needs revision.",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    assert (
        export_approved_corrections(
            benchmark,
            grounded_index,
            reviewed,
        )
        == ()
    )


def test_correction_export_requires_exact_inline_citations(grounded_index):
    benchmark, run = _run(grounded_index)
    queue = build_review_queue(
        run,
        benchmark,
        pass_sample_fraction=1.0,
    )
    reviewed = record_human_review(
        queue,
        queue[0].review_id,
        reviewer_label="incorrect",
        reviewer_notes="Correction omitted its citation.",
        corrected_answer="The mask blocks future tokens.",
        corrected_evidence=(benchmark.questions[0].relevant_chunk_ids[0],),
        timestamp="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="citations must cover"):
        export_approved_corrections(
            benchmark,
            grounded_index,
            reviewed,
        )


def test_reports_contain_real_values_and_failure_context(grounded_index):
    benchmark, run = _run(grounded_index)
    review = render_benchmark_review(benchmark)
    summary = render_evaluation_summary(run, benchmark)
    failures = render_failure_report(run, benchmark)
    assert benchmark.benchmark_sha256 in review
    assert f"- Questions: {len(run.question_results)}" in summary
    assert "These deterministic heuristics" in summary
    assert "# Failure report" in failures


def test_regression_comparison_exposes_question_changes(grounded_index):
    _, baseline = _run(grounded_index, mode="retrieval_only", top_k=1)
    _, candidate = _run(grounded_index, mode="retrieval_only", top_k=5)
    comparison = compare_evaluation_runs(baseline, candidate)
    assert comparison["baseline"]["configuration"]["top_k"] == 1
    assert comparison["candidate"]["configuration"]["top_k"] == 5
    assert "metric_deltas" in comparison
    assert comparison["statistical_significance"].startswith("not_computed")
    assert "# Regression comparison" in render_comparison_report(comparison)


def test_attention_starter_is_large_and_never_approved(grounded_index):
    document_id = next(
        item.document_id
        for item in grounded_index.documents
        if item.title == "Causal Attention"
    )
    benchmark = generate_attention_starter_benchmark(
        grounded_index,
        document_id,
    )
    assert attention_starter_question_count() >= 30
    assert len(benchmark.questions) == attention_starter_question_count()
    assert benchmark.approved_questions == ()
    assert all(item.metadata["trusted_gold"] is False for item in benchmark.questions)
    historical = next(
        item
        for item in benchmark.questions
        if item.question_type == "historical_impact"
    )
    assert historical.paper_sufficiency == "external_required"
    assert historical.answerability == "external_sources_required"
