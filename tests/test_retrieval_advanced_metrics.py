import math

import pytest

from localml_scholar.retrieval import (
    average_precision,
    evaluate_rankings,
    mean_relevant_rank,
    ndcg_at_k,
)


def test_average_precision_and_missing_relevant_rank_are_exact() -> None:
    retrieved = ["a", "x", "b"]
    relevant = {"a", "b", "missing"}

    assert average_precision(retrieved, relevant) == pytest.approx(
        (1.0 + 2.0 / 3.0) / 3.0
    )
    assert mean_relevant_rank(retrieved, relevant) == pytest.approx((1 + 3 + 4) / 3)


def test_ndcg_uses_graded_gain_and_handles_no_relevant_result() -> None:
    grades = {"high": 2, "relevant": 1}
    expected_dcg = 1.0 + 3.0 / math.log2(3)
    expected_idcg = 3.0 + 1.0 / math.log2(3)

    assert ndcg_at_k(["relevant", "high"], grades, 2) == pytest.approx(
        expected_dcg / expected_idcg
    )
    assert ndcg_at_k(["none"], grades, 1) == 0.0


def test_evaluation_reports_graded_and_category_macro_metrics() -> None:
    evaluation = evaluate_rankings(
        {
            "direct": ["a", "x"],
            "synonym": ["x", "b"],
        },
        {
            "direct": ["a"],
            "synonym": ["b"],
        },
        grades={
            "direct": {"a": 2},
            "synonym": {"b": 1},
        },
        categories={
            "direct": "lexical_overlap",
            "synonym": "synonym",
        },
    )

    assert evaluation.aggregate.average_precision == 0.75
    assert evaluation.aggregate.recall_at_5 == 1.0
    assert evaluation.per_category["lexical_overlap"].reciprocal_rank == 1.0
    assert evaluation.per_category["synonym"].reciprocal_rank == 0.5
    assert evaluation.aggregation_policy == "macro_average_over_queries"


@pytest.mark.parametrize(
    ("grades", "message"),
    [
        ({"q": {"a": 0}}, "positive"),
        ({"other": {"a": 1}}, "same queries"),
    ],
)
def test_evaluation_rejects_malformed_grades(grades, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_rankings({"q": ["a"]}, {"q": ["a"]}, grades=grades)
