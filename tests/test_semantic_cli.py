from __future__ import annotations

import json
from pathlib import Path

from localml_scholar.answering.cli import main as answer_main
from localml_scholar.retrieval.search import main as search_main


def test_search_cli_enrich_semantic_hybrid_and_inspect(
    tmp_path: Path,
    capsys,
) -> None:
    first = tmp_path / "a.md"
    second = tmp_path / "b.md"
    first.write_text(
        "# Attention\n\nA decoder cannot attend to later positions.\n",
        encoding="utf-8",
    )
    second.write_text(
        "# Optimization\n\nA learning rate controls the update step.\n",
        encoding="utf-8",
    )
    lexical_path = tmp_path / "lexical.json"
    semantic_path = tmp_path / "semantic.json"
    assert (
        search_main(
            [
                "build",
                "--sources",
                str(first),
                str(second),
                "--output",
                str(lexical_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        search_main(
            [
                "enrich",
                "--index",
                str(lexical_path),
                "--output",
                str(semantic_path),
                "--dimensions",
                "2",
            ]
        )
        == 0
    )
    enrichment = json.loads(capsys.readouterr().out)
    assert enrichment["chunk_ids_preserved"]
    assert enrichment["semantic_sha256"]

    assert search_main(["inspect", "--index", str(semantic_path)]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["semantic"]["configuration"]["method"] == "lsa"

    assert (
        search_main(
            [
                "search",
                "--index",
                str(semantic_path),
                "--query",
                "later decoder position",
                "--method",
                "hybrid",
                "--fusion",
                "rrf",
                "--rerank",
                "--verbose",
                "--json",
            ]
        )
        == 0
    )
    search = json.loads(capsys.readouterr().out)
    assert search["method"] == "hybrid_reranked"
    assert search["results"][0]["scoring_details"]["reranker"]
    assert (
        search["results"][0]["citation"]["chunk_id"] == search["results"][0]["chunk_id"]
    )


def test_answer_cli_supports_semantic_retrieval(
    grounded_index,
    tmp_path: Path,
    capsys,
) -> None:
    dimensions = min(
        4,
        len(grounded_index.chunks),
        len(grounded_index.vocabulary),
    )
    enriched = grounded_index.enrich_semantic()
    if dimensions != 1:
        from localml_scholar.retrieval import SemanticRetrievalConfig

        enriched = grounded_index.enrich_semantic(
            SemanticRetrievalConfig(dimensions=dimensions)
        )
    path = enriched.save(tmp_path / "answer-index.json")

    assert (
        answer_main(
            [
                "--index",
                str(path),
                "--question",
                "How does a decoder prevent future token leakage?",
                "--retriever",
                "semantic",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    answer = payload["answer"]
    assert answer["metadata"]["retrieval_method"] == "semantic"
    assert answer["validation"]["citations_valid"]
    assert all(
        evidence["retrieval_method"] == "semantic" for evidence in answer["evidence"]
    )
