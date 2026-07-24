import math

import pytest

from localml_scholar.retrieval.bm25 import (
    BM25Config,
    bm25_inverse_document_frequency,
    bm25_term_contribution,
)
from localml_scholar.retrieval.tfidf import (
    cosine_score,
    smooth_inverse_document_frequency,
    sparse_tfidf_weights,
    sublinear_term_frequency,
)


def test_tfidf_formulas_match_hand_calculation() -> None:
    inverse_document_frequency = math.log(3 / 2) + 1

    assert sublinear_term_frequency(2) == pytest.approx(1 + math.log(2))
    assert smooth_inverse_document_frequency(2, 1) == pytest.approx(
        inverse_document_frequency
    )
    weights = sparse_tfidf_weights(
        {"common": 1, "rare": 2},
        {"common": 2, "rare": 1},
        2,
    )
    assert weights == pytest.approx(
        {
            "common": 1.0,
            "rare": (1 + math.log(2)) * inverse_document_frequency,
        }
    )


def test_tfidf_cosine_matches_hand_computed_sparse_vectors() -> None:
    query = {"a": 1.0, "b": 2.0}
    document = {"b": 3.0, "c": 4.0}

    score, numerator, query_norm, document_norm, contributions = cosine_score(
        query,
        document,
    )

    assert numerator == 6.0
    assert query_norm == pytest.approx(math.sqrt(5))
    assert document_norm == 5.0
    assert score == pytest.approx(6 / (5 * math.sqrt(5)))
    assert contributions == {"b": 6.0}


def test_tfidf_zero_overlap_and_zero_vector_are_safe() -> None:
    assert cosine_score({"a": 1.0}, {"b": 2.0})[0] == 0.0
    assert cosine_score({}, {"b": 2.0})[0] == 0.0


def test_bm25_formula_matches_hand_calculation() -> None:
    config = BM25Config(k1=1.2, b=0.75)
    expected_idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    length_normalization = 1.2 * (1 - 0.75 + 0.75 * 4 / 5)
    expected = expected_idf * 2 * 2.2 / (2 + length_normalization)

    contribution, inverse_document_frequency, normalization = bm25_term_contribution(
        term_frequency=2,
        document_frequency=1,
        document_length=4,
        average_document_length=5,
        number_of_chunks=3,
        config=config,
    )

    assert bm25_inverse_document_frequency(3, 1) == pytest.approx(expected_idf)
    assert inverse_document_frequency == pytest.approx(expected_idf)
    assert normalization == pytest.approx(length_normalization)
    assert contribution == pytest.approx(expected)


@pytest.mark.parametrize("b", [0.0, 1.0])
def test_bm25_length_normalization_boundaries(b: float) -> None:
    short = bm25_term_contribution(
        term_frequency=1,
        document_frequency=1,
        document_length=2,
        average_document_length=10,
        number_of_chunks=2,
        config=BM25Config(k1=1.2, b=b),
    )[0]
    long = bm25_term_contribution(
        term_frequency=1,
        document_frequency=1,
        document_length=20,
        average_document_length=10,
        number_of_chunks=2,
        config=BM25Config(k1=1.2, b=b),
    )[0]

    assert short == pytest.approx(long) if b == 0.0 else short > long


def test_bm25_zero_frequency_is_zero_and_rare_term_has_larger_idf() -> None:
    assert (
        bm25_term_contribution(
            term_frequency=0,
            document_frequency=1,
            document_length=2,
            average_document_length=2,
            number_of_chunks=3,
            config=BM25Config(),
        )[0]
        == 0.0
    )
    assert bm25_inverse_document_frequency(10, 1) > (
        bm25_inverse_document_frequency(10, 8)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k1": 0},
        {"k1": float("inf")},
        {"b": -0.1},
        {"b": 1.1},
        {"b": float("nan")},
    ],
)
def test_bm25_configuration_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        BM25Config(**kwargs)
