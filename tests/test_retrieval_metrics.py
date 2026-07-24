import pytest

from localml_scholar.retrieval import (
    evaluate_rankings,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_individual_retrieval_metrics() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"b", "d"}

    assert precision_at_k(retrieved, relevant, 1) == 0.0
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert hit_rate_at_k(retrieved, relevant, 1) == 0.0
    assert hit_rate_at_k(retrieved, relevant, 2) == 1.0


def test_metrics_handle_no_results_and_reject_no_relevance() -> None:
    assert precision_at_k([], {"a"}, 3) == 0.0
    assert recall_at_k([], {"a"}, 3) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0
    assert hit_rate_at_k([], {"a"}, 3) == 0.0
    with pytest.raises(ValueError, match="undefined"):
        recall_at_k([], set(), 3)


def test_aggregate_metrics_are_exact_means() -> None:
    evaluation = evaluate_rankings(
        {
            "q1": ["a", "x"],
            "q2": ["x", "b"],
        },
        {
            "q1": ["a"],
            "q2": ["b"],
        },
        valid_chunk_ids={"a", "b", "x"},
    )

    assert evaluation.per_query["q1"].reciprocal_rank == 1.0
    assert evaluation.per_query["q2"].reciprocal_rank == 0.5
    assert evaluation.aggregate.reciprocal_rank == 0.75
    assert evaluation.aggregate.hit_rate_at_3 == 1.0


@pytest.mark.parametrize(
    ("rankings", "relevance", "message"),
    [
        ({"q": ["a", "a"]}, {"q": ["a"]}, "duplicates"),
        ({"q": ["a"]}, {"q": ["a", "a"]}, "duplicates"),
        ({"q": ["a"]}, {"other": ["a"]}, "same queries"),
        ({"q": []}, {"q": []}, "relevant"),
    ],
)
def test_evaluation_rejects_malformed_inputs(rankings, relevance, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_rankings(rankings, relevance)


def test_evaluation_rejects_unknown_relevance_ids() -> None:
    with pytest.raises(ValueError, match="unknown"):
        evaluate_rankings(
            {"q": ["known"]},
            {"q": ["missing"]},
            valid_chunk_ids={"known"},
        )
