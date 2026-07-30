from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from localml_scholar.retrieval import (
    ChunkingConfig,
    RetrievalIndex,
    SearchFilters,
    SemanticIndex,
    SemanticRetrievalConfig,
    build_tfidf_matrix,
    canonicalize_svd_signs,
    fit_lsa,
    ingest_plain_text,
)
from localml_scholar.retrieval.documents import canonical_json
from localml_scholar.retrieval.tfidf import (
    smooth_inverse_document_frequency,
    sublinear_term_frequency,
)


def _semantic_index(dimensions: int = 2) -> RetrievalIndex:
    documents = [
        ingest_plain_text(
            "A feline companion is a household cat that purrs.",
            source="cats.txt",
            metadata={"logical_collection": "animals"},
        ),
        ingest_plain_text(
            "A canine companion is a household dog that barks.",
            source="dogs.txt",
            metadata={"logical_collection": "animals"},
        ),
        ingest_plain_text(
            "Gradient descent optimization uses a learning rate.",
            source="optimization.txt",
            metadata={"logical_collection": "ml"},
        ),
    ]
    lexical = RetrievalIndex.build(
        documents,
        chunking_config=ChunkingConfig(
            target_characters=200,
            maximum_characters=240,
            overlap_characters=0,
            minimum_characters=1,
        ),
    )
    return lexical.enrich_semantic(SemanticRetrievalConfig(dimensions=dimensions))


def test_tfidf_matrix_has_exact_vocabulary_alignment_and_weights() -> None:
    frequencies = ({"alpha": 1}, {"alpha": 1, "beta": 2})
    document_frequencies = {"alpha": 2, "beta": 1}
    before = copy.deepcopy(frequencies)

    matrix = build_tfidf_matrix(
        frequencies,
        document_frequencies,
        ("alpha", "beta"),
    )

    beta = sublinear_term_frequency(2) * smooth_inverse_document_frequency(2, 1)
    np.testing.assert_allclose(matrix, [[1.0, 0.0], [1.0, beta]])
    assert matrix.dtype == np.float64
    assert frequencies == before
    with pytest.raises(ValueError, match="align"):
        build_tfidf_matrix(frequencies, {"alpha": 2}, ("alpha", "beta"))


def test_tfidf_matrix_allows_explicit_zero_rows_but_lsa_rejects_all_zero() -> None:
    matrix = build_tfidf_matrix(({}, {"term": 1}), {"term": 1}, ("term",))
    np.testing.assert_array_equal(matrix[0], [0.0])

    with pytest.raises(ValueError, match="all-zero"):
        fit_lsa(
            term_frequencies=({}, {}),
            document_frequencies={"term": 0},
            vocabulary=("term",),
            chunk_ids=("a", "b"),
        )


def test_svd_sign_canonicalization_preserves_reconstruction_and_ties() -> None:
    left = np.array([[1.0, 2.0], [3.0, 4.0]])
    right = np.array([[-0.8, 0.8], [0.2, -0.9]])
    before = left @ right

    canonical_left, canonical_right, pivots = canonicalize_svd_signs(left, right)

    np.testing.assert_allclose(canonical_left @ canonical_right, before)
    assert pivots == (0, 1)
    assert canonical_right[0, 0] >= 0.0
    assert canonical_right[1, 1] >= 0.0
    flipped_left, flipped_right, flipped_pivots = canonicalize_svd_signs(
        -canonical_left,
        -canonical_right,
    )
    np.testing.assert_array_equal(flipped_left, canonical_left)
    np.testing.assert_array_equal(flipped_right, canonical_right)
    assert flipped_pivots == pivots


def test_lsa_reconstruction_rank_validation_and_repeated_identity() -> None:
    frequencies = (
        {"alpha": 2, "bridge": 1},
        {"beta": 2, "bridge": 1},
        {"gamma": 2, "other": 1},
    )
    document_frequencies = {
        "alpha": 1,
        "beta": 1,
        "bridge": 2,
        "gamma": 1,
        "other": 1,
    }
    vocabulary = tuple(sorted(document_frequencies))
    config = SemanticRetrievalConfig(dimensions=2)

    first = fit_lsa(
        term_frequencies=frequencies,
        document_frequencies=document_frequencies,
        vocabulary=vocabulary,
        chunk_ids=("a", "b", "c"),
        config=config,
    )
    second = fit_lsa(
        term_frequencies=frequencies,
        document_frequencies=document_frequencies,
        vocabulary=vocabulary,
        chunk_ids=("a", "b", "c"),
        config=config,
    )

    assert first.to_dict() == second.to_dict()
    assert first.semantic_sha256 == second.semantic_sha256
    assert first.effective_rank == 3
    assert 0.0 < first.explained_squared_singular_fraction < 1.0
    assert first.reconstruction_error > 0.0
    assert not first.chunk_embeddings.flags.writeable
    assert not first.right_singular_vectors.flags.writeable
    with pytest.raises(ValueError, match="effective matrix rank"):
        fit_lsa(
            term_frequencies=frequencies,
            document_frequencies=document_frequencies,
            vocabulary=vocabulary,
            chunk_ids=("a", "b", "c"),
            config=SemanticRetrievalConfig(dimensions=4),
        )


def test_query_projection_oov_zero_and_known_terms_are_deterministic() -> None:
    index = _semantic_index()
    semantic = index.semantic_index
    first = semantic.project_query(
        ("cat", "unknown", "cat"), index.document_frequencies
    )
    second = semantic.project_query(
        ("cat", "unknown", "cat"), index.document_frequencies
    )

    np.testing.assert_array_equal(first.embedding, second.embedding)
    assert first.known_terms == ("cat",)
    assert first.out_of_vocabulary_terms == ("unknown",)
    assert first.raw_norm > 0.0
    assert np.linalg.norm(first.embedding) == pytest.approx(1.0)

    zero = semantic.project_query(("unknown",), index.document_frequencies)
    np.testing.assert_array_equal(zero.embedding, np.zeros(2))
    assert zero.raw_norm == 0.0
    with pytest.raises(ValueError, match="vocabulary"):
        semantic.project_query(("cat",), {"cat": 1})


def test_hand_computed_semantic_cosine_and_zero_vector_behavior() -> None:
    semantic = SemanticIndex(
        config=SemanticRetrievalConfig(dimensions=2),
        vocabulary=("alpha", "beta"),
        chunk_ids=("c1", "c2"),
        right_singular_vectors=np.eye(2, dtype=np.float64),
        chunk_embeddings=np.eye(2, dtype=np.float64),
        chunk_raw_norms=np.ones(2, dtype=np.float64),
        singular_values=np.ones(2, dtype=np.float64),
        effective_rank=2,
        reconstruction_error=0.0,
        explained_squared_singular_fraction=1.0,
        zero_row_indices=(),
        canonical_pivot_indices=(0, 1),
        matrix_shape=(2, 2),
    )
    query = semantic.project_query(("alpha",), {"alpha": 1, "beta": 1})
    scores, numerators = semantic.cosine_scores(query)

    np.testing.assert_allclose(scores, [1.0, 0.0])
    np.testing.assert_allclose(numerators, [1.0, 0.0])
    zero = semantic.project_query(("missing",), {"alpha": 1, "beta": 1})
    np.testing.assert_array_equal(semantic.cosine_scores(zero)[0], [0.0, 0.0])


def test_semantic_search_filters_ties_and_preserves_exact_citations() -> None:
    index = _semantic_index()
    first = index.search("household companion", method="semantic", top_k=10)
    second = index.search("household companion", method="semantic", top_k=10)

    assert [result.to_dict() for result in first] == [
        result.to_dict() for result in second
    ]
    assert first
    frequencies_by_chunk = dict(
        zip(
            (chunk.chunk_id for chunk in index.chunks),
            index.term_frequencies,
            strict=True,
        )
    )
    for result in first:
        chunk = next(
            chunk for chunk in index.chunks if chunk.chunk_id == result.chunk_id
        )
        assert result.text == chunk.text
        assert result.citation.chunk_id == chunk.chunk_id
        assert result.retrieval_method == "semantic"
        assert result.semantic_query_terms == ("companion", "household")
        assert set(result.matched_terms) == {
            term
            for term in result.semantic_query_terms
            if term in frequencies_by_chunk[result.chunk_id]
        }
        assert result.scoring_details["latent_dimension_labels"] is None
        assert result.scoring_details["similarity_is_not_proof_of_relevance"]
    filtered = index.search(
        "household companion",
        method="semantic",
        top_k=10,
        filters=SearchFilters(logical_collection="animals"),
    )
    assert filtered
    assert all(result.source_name in {"cats.txt", "dogs.txt"} for result in filtered)
    assert index.search("entirely_oov_term", method="semantic") == ()


def test_semantic_enrichment_round_trip_preserves_lexical_results(
    tmp_path: Path,
) -> None:
    lexical = _semantic_index().semantic_index
    base = RetrievalIndex.build(
        [
            ingest_plain_text("alpha beta bridge", source="a.txt"),
            ingest_plain_text("beta gamma bridge", source="b.txt"),
        ]
    )
    before = {
        method: [
            result.to_dict() for result in base.search("beta bridge", method=method)
        ]
        for method in ("tfidf", "bm25")
    }
    enriched = base.enrich_semantic(SemanticRetrievalConfig(dimensions=2))
    path = enriched.save(tmp_path / "enriched.json")
    loaded = RetrievalIndex.load(path)

    assert lexical is not None
    assert [chunk.chunk_id for chunk in base.chunks] == [
        chunk.chunk_id for chunk in enriched.chunks
    ]
    assert enriched.corpus_sha256 == base.corpus_sha256
    assert enriched.index_sha256 != base.index_sha256
    for method in ("tfidf", "bm25"):
        assert [
            result.to_dict() for result in enriched.search("beta bridge", method=method)
        ] == before[method]
    assert loaded.state_dict() == enriched.state_dict()
    assert [
        result.to_dict() for result in loaded.search("beta", method="semantic")
    ] == [result.to_dict() for result in enriched.search("beta", method="semantic")]
    assert path.read_bytes() == enriched.save(tmp_path / "copy.json").read_bytes()


def test_legacy_lexical_index_loads_and_requires_explicit_enrichment() -> None:
    current = RetrievalIndex.build(
        [ingest_plain_text("alpha beta", source="legacy.txt")]
    )
    state = current.state_dict()
    state.pop("semantic_index")
    state["index_format_version"] = 1
    state["index_type"] = "immutable_lexical_snapshot"
    state["package_version"] = "0.9.0"
    state_without_hash = dict(state)
    state_without_hash.pop("index_sha256")
    state["index_sha256"] = hashlib.sha256(
        canonical_json(state_without_hash).encode("utf-8")
    ).hexdigest()

    legacy = RetrievalIndex.from_state_dict(state)

    assert legacy.index_format_version == 1
    assert legacy.search("alpha", method="bm25")
    with pytest.raises(ValueError, match="enrich"):
        legacy.search("alpha", method="semantic")
    enriched = legacy.enrich_semantic(SemanticRetrievalConfig(dimensions=1))
    assert enriched.semantic_index is not None
    assert enriched.index_format_version == 2


def test_version_two_index_from_package_1_0_remains_loadable() -> None:
    current = _semantic_index()
    state = current.state_dict()
    state["package_version"] = "1.0.0"
    state_without_hash = dict(state)
    state_without_hash.pop("index_sha256")
    state["index_sha256"] = hashlib.sha256(
        canonical_json(state_without_hash).encode("utf-8")
    ).hexdigest()

    loaded = RetrievalIndex.from_state_dict(state)

    assert loaded.package_version == "1.0.0"
    assert loaded.index_format_version == 2
    assert [item.to_dict() for item in loaded.search("alpha", method="semantic")] == [
        item.to_dict() for item in current.search("alpha", method="semantic")
    ]


def test_malformed_semantic_state_is_rejected_transactionally() -> None:
    index = _semantic_index()
    before = index.state_dict()
    malformed = copy.deepcopy(before)
    malformed["semantic_index"]["chunk_embeddings"][0][0] = math.nan

    with pytest.raises(ValueError, match="finite"):
        RetrievalIndex.from_state_dict(malformed)
    assert index.state_dict() == before
