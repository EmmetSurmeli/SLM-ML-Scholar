from pathlib import Path

from experiments.compare_retrievers import compare_retrievers
from experiments.inspect_document_ingestion import inspect_document_ingestion


def test_document_ingestion_inspection_round_trip(tmp_path: Path) -> None:
    summary = inspect_document_ingestion(
        output_directory=tmp_path / "inspection",
    )

    assert summary["reconstruction"]["all_chunk_slices_exact"]
    assert summary["reconstruction"]["coverage_validation_passed"]
    assert summary["index_round_trip"]["state_exact"]
    assert summary["index_round_trip"]["results_exact"]
    assert summary["ranked_passages"][0]["source_name"] == "attention.md"
    assert not summary["answer_generated"]
    assert Path(summary["artifacts"]["index"]).is_file()


def test_retriever_comparison_has_exact_fixture_metrics(tmp_path: Path) -> None:
    summary = compare_retrievers(
        output_directory=tmp_path / "comparison",
    )

    assert summary["query_count"] == 5
    assert summary["index"]["documents"] == 3
    assert summary["index"]["chunks"] == 9
    for run in summary["runs"]:
        assert run["reload_results_exact"]
        assert run["metrics"]["aggregate"]["hit_rate_at_3"] == 1.0
        assert run["metrics"]["aggregate"]["reciprocal_rank"] > 0.0
    assert not summary["answer_generated"]
    assert Path(summary["artifacts"]["summary"]).is_file()
