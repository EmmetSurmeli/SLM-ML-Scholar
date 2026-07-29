"""Evaluate lexical, semantic, hybrid, and reranked retrieval on one fixture."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.semantic_retrieval_fixture import (  # noqa: E402
    build_semantic_fixture_index,
    load_semantic_judgments,
)
from localml_scholar.retrieval import (  # noqa: E402
    HybridRetrievalConfig,
    RetrievalIndex,
    evaluate_rankings,
)
from localml_scholar.serialization import atomic_write_text  # noqa: E402


def _method_specs() -> tuple[tuple[str, str, HybridRetrievalConfig | None], ...]:
    return (
        ("tfidf", "tfidf", None),
        ("bm25", "bm25", None),
        ("semantic", "semantic", None),
        (
            "weighted_bm25_semantic",
            "hybrid",
            HybridRetrievalConfig(fusion="weighted", alpha=0.5),
        ),
        (
            "reciprocal_rank_fusion",
            "hybrid",
            HybridRetrievalConfig(fusion="rrf"),
        ),
        (
            "hybrid_reranked",
            "hybrid_reranked",
            HybridRetrievalConfig(fusion="rrf"),
        ),
    )


def evaluate_method(
    index: RetrievalIndex,
    judgments: tuple[dict[str, Any], ...],
    *,
    label: str,
    method: str,
    hybrid_config: HybridRetrievalConfig | None,
    top_k: int,
) -> dict[str, Any]:
    """Evaluate one explicit method and retain query-level score explanations."""
    rankings: dict[str, list[str]] = {}
    relevance: dict[str, list[str]] = {}
    grades: dict[str, dict[str, int]] = {}
    categories: dict[str, str] = {}
    queries: dict[str, Any] = {}
    started = time.perf_counter()
    for judgment in judgments:
        results = index.search(
            judgment["query"],
            method=method,
            top_k=top_k,
            hybrid_config=hybrid_config,
        )
        query_id = judgment["query_id"]
        rankings[query_id] = [result.chunk_id for result in results]
        relevance[query_id] = list(judgment["grades"])
        grades[query_id] = judgment["grades"]
        categories[query_id] = judgment["category"]
        queries[query_id] = {
            "query": judgment["query"],
            "category": judgment["category"],
            "split": judgment["split"],
            "ranking": [
                {
                    "rank": result.rank,
                    "chunk_id": result.chunk_id,
                    "score": result.score,
                    "source_name": result.source_name,
                    "heading_path": list(result.heading_path),
                    "citation": result.citation.to_dict(),
                    "matched_terms": list(result.matched_terms),
                    "scoring_details": result.scoring_details,
                    "relevance_grade": judgment["grades"].get(result.chunk_id, 0),
                }
                for result in results
            ],
        }
    elapsed = time.perf_counter() - started
    evaluation = evaluate_rankings(
        rankings,
        relevance,
        valid_chunk_ids={chunk.chunk_id for chunk in index.chunks},
        grades=grades,
        categories=categories,
    )
    failures = [
        query_id
        for query_id, ranking in rankings.items()
        if not set(ranking[:5]) & set(relevance[query_id])
    ]
    return {
        "label": label,
        "method": method,
        "hybrid_configuration": (
            None if hybrid_config is None else hybrid_config.to_dict()
        ),
        "top_k": top_k,
        "latency_seconds_total": elapsed,
        "latency_seconds_per_query": elapsed / len(judgments),
        "metrics": evaluation.to_dict(),
        "query_results": queries,
        "failure_query_ids_at_5": failures,
    }


def run_ablation(
    output_directory: Path,
    *,
    dimensions: int = 6,
    top_k: int = 10,
) -> dict[str, Any]:
    """Build one index and evaluate every method against identical judgments."""
    output_directory.mkdir(parents=True, exist_ok=True)
    index = build_semantic_fixture_index(dimensions=dimensions)
    judgments = load_semantic_judgments()
    index_path = index.save(output_directory / "semantic_fixture_index.json")
    methods = {
        label: evaluate_method(
            index,
            judgments,
            label=label,
            method=method,
            hybrid_config=hybrid_config,
            top_k=top_k,
        )
        for label, method, hybrid_config in _method_specs()
    }
    summary = {
        "experiment": "retrieval_ablation",
        "interpretation": (
            "Project-authored fixture comparison; it is not evidence of general "
            "retrieval superiority."
        ),
        "semantic_baseline": "tfidf_lsa",
        "semantic_configuration": index.semantic_index.config.to_dict(),
        "semantic_sha256": index.semantic_index.semantic_sha256,
        "index_sha256": index.index_sha256,
        "corpus_sha256": index.corpus_sha256,
        "serialized_index_path": str(index_path),
        "serialized_index_bytes": index_path.stat().st_size,
        "query_count": len(judgments),
        "methods": methods,
    }
    atomic_write_text(
        output_directory / "retrieval_ablation.json",
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return summary


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/retrieval_ablation"),
    )
    parser.add_argument("--dimensions", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    summary = run_ablation(
        args.output_directory,
        dimensions=args.dimensions,
        top_k=args.top_k,
    )
    aggregates = {
        label: result["metrics"]["aggregate"]
        for label, result in summary["methods"].items()
    }
    print(json.dumps(aggregates, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
