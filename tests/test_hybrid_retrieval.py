from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from localml_scholar.retrieval import (
    HybridRetrievalConfig,
    RerankingConfig,
    RetrievalIndex,
    SemanticRetrievalConfig,
    fuse_rankings,
    ingest_markdown,
    ingest_plain_text,
    maximum_positive_normalization,
    reranking_features,
    source_range_overlap,
    weighted_reranking_score,
)


def _result(chunk_id: str, rank: int, score: float):
    return SimpleNamespace(chunk_id=chunk_id, rank=rank, score=score)


def _index() -> RetrievalIndex:
    lexical = RetrievalIndex.build(
        [
            ingest_markdown(
                "# Decoder Mask\n\n"
                "Position 2 cannot attend to later_position 3, preventing leakage.",
                source="attention.md",
            ),
            ingest_plain_text(
                "The learning rate 0.01 controls optimization step size.",
                source="optimization.txt",
            ),
            ingest_plain_text(
                "Probability expectation summarizes random outcomes.",
                source="probability.txt",
            ),
        ]
    )
    return lexical.enrich_semantic(SemanticRetrievalConfig(dimensions=3))


def test_maximum_positive_normalization_and_invalid_scores() -> None:
    assert maximum_positive_normalization({"a": 2.0, "b": 1.0}) == {
        "a": 1.0,
        "b": 0.5,
    }
    assert maximum_positive_normalization({"a": 0.0}) == {"a": 0.0}
    with pytest.raises(ValueError, match="non-negative"):
        maximum_positive_normalization({"a": -1.0})


def test_weighted_fusion_alpha_endpoints_missing_components_and_ties() -> None:
    lexical = [_result("a", 1, 4.0), _result("b", 2, 2.0)]
    semantic = [_result("b", 1, 0.8), _result("c", 2, 0.4)]

    lexical_only = fuse_rankings(
        lexical,
        semantic,
        config=HybridRetrievalConfig(fusion="weighted", alpha=1.0),
    )
    semantic_only = fuse_rankings(
        lexical,
        semantic,
        config=HybridRetrievalConfig(fusion="weighted", alpha=0.0),
    )

    by_id = {item["chunk_id"]: item for item in lexical_only}
    assert by_id["a"]["score"] == 1.0
    assert by_id["b"]["score"] == 0.5
    assert by_id["c"]["score"] == 0.0
    by_id = {item["chunk_id"]: item for item in semantic_only}
    assert by_id["a"]["score"] == 0.0
    assert by_id["b"]["score"] == 1.0
    assert by_id["c"]["score"] == 0.5
    assert by_id["c"]["lexical_rank"] is None
    assert by_id["a"]["semantic_rank"] is None


def test_rrf_formula_and_component_metadata() -> None:
    config = HybridRetrievalConfig(
        fusion="rrf",
        rrf_rank_constant=10,
        lexical_weight=2.0,
        semantic_weight=1.0,
    )
    fused = fuse_rankings(
        [_result("a", 1, 3.0), _result("b", 2, 2.0)],
        [_result("b", 1, 0.9)],
        config=config,
    )
    by_id = {item["chunk_id"]: item for item in fused}

    assert by_id["a"]["score"] == pytest.approx(2.0 / 11.0)
    assert by_id["b"]["score"] == pytest.approx(2.0 / 12.0 + 1.0 / 11.0)
    assert by_id["b"]["retrieved_by"] == ["bm25", "semantic"]
    assert by_id["b"]["fusion_config"] == config.to_dict()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": -0.1},
        {"alpha": 1.1},
        {"fusion": "raw_addition"},
        {"lexical_method": "unknown"},
        {"rrf_rank_constant": 0},
        {"lexical_weight": 0.0, "semantic_weight": 0.0},
    ],
)
def test_hybrid_config_rejects_invalid_values(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        HybridRetrievalConfig(**kwargs)


def test_reranking_features_cover_phrase_heading_numbers_and_identifiers() -> None:
    index = _index()
    result = index.search("later_position 3", method="hybrid")[0]
    chunk = next(chunk for chunk in index.chunks if chunk.chunk_id == result.chunk_id)
    document = next(
        document
        for document in index.documents
        if document.document_id == chunk.document_id
    )

    features = reranking_features(
        query="later_position 3",
        chunk=chunk,
        document=document,
        fusion_details=result.scoring_details["fusion"],
        document_frequencies=index.document_frequencies,
        chunk_count=len(index.chunks),
    )

    assert features["exact_phrase"] == 1.0
    assert features["query_term_coverage"] == 1.0
    assert features["numerical_overlap"] == 1.0
    assert features["identifier_overlap"] == 1.0
    assert 0.0 <= features["length_penalty"] <= 1.0


def test_weighted_reranking_score_explanation_sums_exactly() -> None:
    features = {
        "lexical": 1.0,
        "semantic": 0.5,
        "exact_phrase": 1.0,
        "heading_match": 0.5,
        "query_term_coverage": 0.75,
        "rare_term_coverage": 0.6,
        "numerical_overlap": 1.0,
        "identifier_overlap": 1.0,
        "length_penalty": 0.2,
    }
    config = RerankingConfig(redundancy_threshold=0.5, redundancy_penalty=0.4)

    score, contributions = weighted_reranking_score(
        features,
        config,
        redundancy_overlap=0.75,
    )

    assert math.fsum(contributions.values()) == pytest.approx(score)
    assert contributions["redundancy_penalty"] == pytest.approx(-0.3)


def test_source_range_overlap_and_reranked_search_are_deterministic() -> None:
    index = _index()
    first_chunk = index.chunks[0]
    assert source_range_overlap(first_chunk, first_chunk) == 1.0
    assert source_range_overlap(first_chunk, index.chunks[-1]) == 0.0

    first = index.search(
        "later token leakage",
        method="hybrid_reranked",
        top_k=3,
    )
    second = index.search(
        "later token leakage",
        method="hybrid_reranked",
        top_k=3,
    )

    assert [result.to_dict() for result in first] == [
        result.to_dict() for result in second
    ]
    assert all(result.retrieval_method == "hybrid_reranked" for result in first)
    for result in first:
        reranker = result.scoring_details["reranker"]
        assert math.fsum(reranker["contributions"].values()) == pytest.approx(
            result.score
        )
        assert result.citation.chunk_id == result.chunk_id


def test_hybrid_requires_semantic_state_and_rejects_hidden_fallback() -> None:
    lexical = RetrievalIndex.build(
        [ingest_plain_text("lexical content", source="only.txt")]
    )
    with pytest.raises(ValueError, match="enrich"):
        lexical.search("content", method="hybrid")
    with pytest.raises(ValueError, match="require a hybrid method"):
        _index().search(
            "content",
            method="bm25",
            hybrid_config=HybridRetrievalConfig(),
        )
