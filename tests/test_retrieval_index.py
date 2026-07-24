import copy
import json
from pathlib import Path

import pytest

from localml_scholar.retrieval import (
    BM25Config,
    ChunkingConfig,
    Citation,
    LexicalTokenizerConfig,
    RetrievalIndex,
    SearchFilters,
    SearchQuery,
    highlight_matches,
    ingest_markdown,
    ingest_pdf_text,
    ingest_plain_text,
)
from localml_scholar.retrieval.documents import PageText


def _index() -> RetrievalIndex:
    attention = ingest_markdown(
        "# Attention\n\n## Causal Mask\n\nA causal mask blocks future tokens.\n",
        source="attention.md",
        metadata={
            "authors": ["A. Author"],
            "publication_year": 2024,
            "logical_collection": "ml",
        },
    )
    optimization = ingest_markdown(
        "# Optimization\n\n## Gradient Descent\n\n"
        "Gradient descent uses a learning rate and gradient direction.\n",
        source="optimization.md",
        metadata={"logical_collection": "ml"},
    )
    probability = ingest_plain_text(
        "Probability describes random outcomes and expectation.",
        source="probability.txt",
        metadata={"logical_collection": "statistics"},
    )
    return RetrievalIndex.build(
        [probability, optimization, attention],
        chunking_config=ChunkingConfig(
            target_characters=100,
            maximum_characters=140,
            overlap_characters=10,
            minimum_characters=10,
        ),
    )


@pytest.mark.parametrize("method", ["tfidf", "bm25"])
def test_search_ranks_relevant_exact_passage_with_explanations(method: str) -> None:
    index = _index()

    results = index.search("How does a causal mask block future tokens?", method=method)

    assert results[0].source_name == "attention.md"
    assert results[0].heading_path == ("Attention", "Causal Mask")
    assert "future tokens" in results[0].text
    assert results[0].rank == 1
    assert results[0].score > 0.0
    assert set(results[0].matched_terms) >= {"causal", "mask", "future", "tokens"}
    assert sum(
        record["score_contribution"] for record in results[0].term_contributions
    ) == pytest.approx(results[0].score)
    if method == "tfidf":
        assert all(
            {
                "idf",
                "query_weight",
                "chunk_weight",
                "query_term_frequency",
            }
            <= record.keys()
            for record in results[0].term_contributions
        )
    assert results[0].citation.format().startswith("[Attention")


def test_bm25_repeated_query_terms_contribute_once() -> None:
    index = _index()
    one = index.search("gradient", method="bm25")
    repeated = index.search("gradient gradient gradient", method="bm25")

    assert [result.score for result in one] == [result.score for result in repeated]
    assert repeated[0].scoring_details["repeated_query_term_policy"] == "unique_terms"


def test_no_shared_terms_returns_no_results_and_ties_are_deterministic() -> None:
    index = _index()

    assert index.search("zzzz_unseen_term") == ()


@pytest.mark.parametrize("method", ["tfidf", "bm25"])
def test_equal_score_ties_use_document_id_then_chunk_order(method: str) -> None:
    documents = [
        ingest_plain_text("shared bravo", source="b.txt"),
        ingest_plain_text("shared alpha", source="a.txt"),
    ]
    index = RetrievalIndex.build(documents)

    first = index.search("shared", method=method, top_k=10)
    second = index.search("shared", method=method, top_k=10)

    assert first == second
    assert len(first) == 2
    assert first[0].score == pytest.approx(first[1].score)
    ordering_keys = [
        (
            result.document_id,
            index.chunks[
                next(
                    position
                    for position, chunk in enumerate(index.chunks)
                    if chunk.chunk_id == result.chunk_id
                )
            ].ordinal,
            result.chunk_id,
        )
        for result in first
    ]
    assert ordering_keys == sorted(ordering_keys)


def test_index_rejects_chunks_without_lexical_terms() -> None:
    document = ingest_plain_text("--- !!!", source="punctuation.txt")

    with pytest.raises(ValueError, match="no lexical terms"):
        RetrievalIndex.build([document])


@pytest.mark.parametrize(
    ("filters", "expected_sources"),
    [
        (SearchFilters(source_name="attention.md"), {"attention.md"}),
        (
            SearchFilters(media_type="text/markdown"),
            {"attention.md", "optimization.md"},
        ),
        (
            SearchFilters(logical_collection="ml"),
            {"attention.md", "optimization.md"},
        ),
        (SearchFilters(publication_year=2024), {"attention.md"}),
        (
            SearchFilters(heading_path_prefix=("Attention", "Causal Mask")),
            {"attention.md"},
        ),
    ],
)
def test_explicit_search_filters(
    filters: SearchFilters,
    expected_sources: set[str],
) -> None:
    results = _index().search(
        "causal future gradient",
        method="bm25",
        top_k=10,
        filters=filters,
    )

    assert results
    assert {result.source_name for result in results} <= expected_sources


def test_document_id_filter_and_no_filter_matches() -> None:
    index = _index()
    attention = next(
        document
        for document in index.documents
        if document.source_name == "attention.md"
    )
    results = index.search(
        "future",
        filters=SearchFilters(document_id=attention.document_id),
    )

    assert results and all(
        result.document_id == attention.document_id for result in results
    )
    assert (
        index.search(
            "future",
            filters=SearchFilters(document_id="doc_missing"),
        )
        == ()
    )


def test_search_query_validates_top_k_and_cannot_be_overridden() -> None:
    index = _index()
    query = SearchQuery.from_text("gradient", top_k=1)

    assert len(index.search(query)) == 1
    with pytest.raises(ValueError, match="override"):
        index.search(query, top_k=2)
    with pytest.raises(ValueError, match="non-whitespace"):
        SearchQuery.from_text(" ")
    with pytest.raises(ValueError, match="positive"):
        SearchQuery.from_text("term", top_k=0)


def test_index_round_trip_preserves_state_rankings_and_citations(
    tmp_path: Path,
) -> None:
    index = _index()
    path = index.save(tmp_path / "index.json")
    loaded = RetrievalIndex.load(path)

    assert loaded.state_dict() == index.state_dict()
    for method in ("tfidf", "bm25"):
        assert [
            result.to_dict() for result in loaded.search("future mask", method=method)
        ] == [result.to_dict() for result in index.search("future mask", method=method)]
    assert path.read_bytes() == index.save(tmp_path / "second.json").read_bytes()


def test_index_load_rejects_malformed_state_transactionally(tmp_path: Path) -> None:
    index = _index()
    before = index.state_dict()
    malformed = copy.deepcopy(before)
    malformed["chunks"][0]["text"] += "tampered"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(malformed), encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        RetrievalIndex.load(path)
    assert index.state_dict() == before


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: state.update(index_format_version=99),
        lambda state: state.update(package_version="0.0.0"),
        lambda state: state.pop("vocabulary"),
        lambda state: state.update(unexpected=True),
        lambda state: state["document_frequencies"].update(unknown=1),
        lambda state: state["term_frequencies"].__setitem__(0, []),
    ],
)
def test_index_rejects_unsupported_or_malformed_schema(mutation) -> None:
    state = copy.deepcopy(_index().state_dict())
    mutation(state)

    with pytest.raises((TypeError, ValueError)):
        RetrievalIndex.from_state_dict(state)


def test_change_detection_covers_content_sources_and_configurations() -> None:
    index = _index()
    unchanged = list(index.documents)
    changed = [
        document
        if document.source_name != "probability.txt"
        else ingest_plain_text("changed probability", source="probability.txt")
        for document in index.documents
    ]

    assert index.change_reasons(unchanged) == ("unchanged",)
    assert any(
        "source_content_changed" in reason for reason in index.change_reasons(changed)
    )
    assert index.change_reasons(
        unchanged,
        bm25_config=BM25Config(k1=2.0),
    ) == ("bm25_configuration_changed",)
    assert index.change_reasons(
        unchanged,
        chunking_config=ChunkingConfig(
            target_characters=80,
            maximum_characters=140,
            overlap_characters=10,
            minimum_characters=10,
        ),
    ) == ("chunking_configuration_changed",)
    assert index.change_reasons(
        unchanged,
        lexical_config=LexicalTokenizerConfig(casefold=False),
    ) == ("lexical_configuration_changed",)
    assert "source_removed:probability.txt" in index.change_reasons(
        [
            document
            for document in unchanged
            if document.source_name != "probability.txt"
        ]
    )
    assert "source_added:new.txt" in index.change_reasons(
        [*unchanged, ingest_plain_text("new source terms", source="new.txt")]
    )


def test_page_and_line_citations_never_fabricate_locations() -> None:
    page_document = ingest_pdf_text(
        [PageText(3, "Causal page text.")],
        source="paper.pdf",
        title="Paper",
    )
    page_result = RetrievalIndex.build([page_document]).search("causal")[0]
    line_result = _index().search("probability")[0]

    assert page_result.citation.format() == "[Paper, p. 3]"
    assert "line" in line_result.citation.format()
    assert "p." not in line_result.citation.format()
    restored = Citation.from_dict(page_result.citation.to_dict())
    assert restored == page_result.citation


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("page_start", 0, "positive"),
        ("page_end", 2, "fully present"),
        ("start_line", 0, "positive and ordered"),
        ("heading_path", [""], "non-empty"),
    ],
)
def test_citation_rejects_malformed_locations(
    field: str,
    value: object,
    message: str,
) -> None:
    state = _index().search("probability")[0].citation.to_dict()
    state[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        Citation.from_dict(state)


def test_display_highlighting_does_not_mutate_source_text() -> None:
    original = "Causal MASK blocks future tokens."

    highlighted = highlight_matches(original, ["causal", "mask"])

    assert highlighted == "[[Causal]] [[MASK]] blocks future tokens."
    assert original == "Causal MASK blocks future tokens."
    with pytest.raises(TypeError, match="sequence"):
        highlight_matches(original, "causal")
    with pytest.raises(ValueError, match="non-empty"):
        highlight_matches(original, [""])
