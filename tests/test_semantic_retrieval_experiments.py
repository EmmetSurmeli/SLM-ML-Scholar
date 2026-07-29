from __future__ import annotations

import json
from pathlib import Path

from experiments.analyze_hybrid_sensitivity import run_sensitivity
from experiments.evaluate_grounded_retrievers import (
    run_grounded_retriever_evaluation,
)
from experiments.evaluate_retrieval_ablation import run_ablation
from experiments.inspect_semantic_retrieval import run_inspection
from experiments.semantic_retrieval_fixture import (
    build_semantic_fixture_index,
    load_semantic_judgments,
)


def test_semantic_fixture_judgments_use_exact_stable_chunk_ids() -> None:
    first = build_semantic_fixture_index()
    second = build_semantic_fixture_index()
    judgments = load_semantic_judgments()
    chunk_ids = {chunk.chunk_id for chunk in first.chunks}

    assert first.state_dict() == second.state_dict()
    assert {record["category"] for record in judgments} >= {
        "lexical_overlap",
        "synonym",
        "paraphrase",
        "notation",
        "numerical",
        "negation",
    }
    assert all(set(record["grades"]) <= chunk_ids for record in judgments)
    assert {record["split"] for record in judgments} == {
        "development",
        "evaluation",
    }


def test_ablation_writes_all_methods_metrics_and_failure_cases(
    tmp_path: Path,
) -> None:
    summary = run_ablation(tmp_path / "ablation")

    assert set(summary["methods"]) == {
        "tfidf",
        "bm25",
        "semantic",
        "weighted_bm25_semantic",
        "reciprocal_rank_fusion",
        "hybrid_reranked",
    }
    for method in summary["methods"].values():
        assert "average_precision" in method["metrics"]["aggregate"]
        assert "ndcg_at_5" in method["metrics"]["aggregate"]
        assert method["query_results"]
        assert isinstance(method["failure_query_ids_at_5"], list)
    saved = json.loads(
        (tmp_path / "ablation" / "retrieval_ablation.json").read_text(encoding="utf-8")
    )
    assert saved["semantic_sha256"] == summary["semantic_sha256"]


def test_inspection_and_sensitivity_are_reloadable_and_disclosed(
    tmp_path: Path,
) -> None:
    inspection = run_inspection(
        tmp_path / "inspection",
        query="why can suffix tokens not alter an earlier prediction",
    )
    sensitivity = run_sensitivity(tmp_path / "sensitivity")

    assert inspection["reload_state_equal"]
    assert inspection["reload_results_equal"]
    assert inspection["semantic_results"]
    assert inspection["reranked_results"]
    assert sensitivity["study_type"] == "descriptive_not_tuned"
    assert sensitivity["development_query_count"] > 0
    assert sensitivity["evaluation_query_count"] > 0
    assert len(sensitivity["settings"]) >= 10


def test_grounded_retriever_regression_preserves_citation_validation(
    tmp_path: Path,
) -> None:
    summary = run_grounded_retriever_evaluation(tmp_path / "grounded")

    assert set(summary["reports"]) == {
        "bm25",
        "semantic",
        "hybrid",
        "hybrid_reranked",
    }
    for report in summary["reports"].values():
        aggregate = report["answer_evaluation"]["aggregate"]
        assert aggregate["citation_validity"] == 1.0
        assert aggregate["citation_coverage"] == 1.0
        assert 0.0 <= aggregate["citation_recall"] <= 1.0
    assert summary["generative_evaluation"] == "not_run_without_an_explicit_checkpoint"
