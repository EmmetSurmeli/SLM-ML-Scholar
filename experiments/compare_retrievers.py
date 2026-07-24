#!/usr/bin/env python3
"""Compare independent TF-IDF and BM25 on the project-authored fixture."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from experiments.retrieval_fixture import (  # noqa: E402
    build_fixture_index,
    load_fixture_relevance,
)
from localml_scholar._version import __version__  # noqa: E402
from localml_scholar.retrieval import evaluate_rankings  # noqa: E402
from localml_scholar.serialization import atomic_write_text  # noqa: E402


def compare_retrievers(
    *,
    output_directory: str | Path = "outputs/retriever_comparison",
) -> dict[str, Any]:
    """Run and serialize a controlled exact-ID lexical retrieval comparison."""
    destination = Path(output_directory)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.mkdir(parents=True, exist_ok=True)
    build_start = time.perf_counter()
    index = build_fixture_index()
    build_seconds = time.perf_counter() - build_start
    index_path = index.save(destination / "fixture_index.json")
    serialized_size = index_path.stat().st_size
    reloaded = type(index).load(index_path)
    relevance = load_fixture_relevance()
    valid_ids = {chunk.chunk_id for chunk in index.chunks}
    runs: list[dict[str, Any]] = []
    for method in ("tfidf", "bm25"):
        rankings: dict[str, list[str]] = {}
        examples: dict[str, list[dict[str, Any]]] = {}
        latencies: list[float] = []
        reload_consistent = True
        for query in sorted(relevance):
            start = time.perf_counter()
            results = index.search(query, method=method, top_k=3)
            latencies.append(time.perf_counter() - start)
            loaded_results = reloaded.search(query, method=method, top_k=3)
            reload_consistent &= [result.to_dict() for result in results] == [
                result.to_dict() for result in loaded_results
            ]
            rankings[query] = [result.chunk_id for result in results]
            examples[query] = [
                {
                    "rank": result.rank,
                    "score": result.score,
                    "chunk_id": result.chunk_id,
                    "citation": result.citation.to_dict(),
                    "text": result.text,
                    "matched_terms": list(result.matched_terms),
                    "term_contributions": list(result.term_contributions),
                }
                for result in results
            ]
        evaluation = evaluate_rankings(
            rankings,
            relevance,
            valid_chunk_ids=valid_ids,
        )
        runs.append(
            {
                "method": method,
                "metrics": evaluation.to_dict(),
                "mean_query_latency_seconds": statistics.fmean(latencies),
                "query_latencies_seconds": latencies,
                "rankings": rankings,
                "examples": examples,
                "reload_results_exact": reload_consistent,
            }
        )
    summary: dict[str, Any] = {
        "milestone": 8,
        "package_version": __version__,
        "purpose": "implementation validation on a tiny authored fixture",
        "claim_boundary": (
            "These results validate deterministic behavior only and do not "
            "establish general retrieval quality."
        ),
        "index": {
            "documents": len(index.documents),
            "chunks": len(index.chunks),
            "vocabulary_size": len(index.vocabulary),
            "average_chunk_length": index.average_chunk_length,
            "index_sha256": index.index_sha256,
            "corpus_sha256": index.corpus_sha256,
            "build_seconds": build_seconds,
            "serialized_bytes": serialized_size,
        },
        "query_count": len(relevance),
        "relevance": relevance,
        "runs": runs,
        "answer_generated": False,
    }
    summary_path = destination / "comparison_summary.json"
    summary["artifacts"] = {
        "index": str(index_path),
        "summary": str(summary_path),
    }
    atomic_write_text(
        summary_path,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare independent TF-IDF and BM25 retrieval."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/retriever_comparison"),
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    compare_retrievers(output_directory=args.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
