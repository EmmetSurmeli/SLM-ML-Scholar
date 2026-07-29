"""Inspect LSA projection, fusion arithmetic, reranking, and exact citations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.semantic_retrieval_fixture import (  # noqa: E402
    build_semantic_fixture_index,
)
from localml_scholar.retrieval import (  # noqa: E402
    RetrievalIndex,
    tokenize_lexically,
)
from localml_scholar.serialization import atomic_write_text  # noqa: E402


def run_inspection(
    output_directory: Path,
    *,
    query: str,
    dimensions: int = 6,
) -> dict:
    """Save a student-inspectable deterministic semantic retrieval trace."""
    output_directory.mkdir(parents=True, exist_ok=True)
    index = build_semantic_fixture_index(dimensions=dimensions)
    semantic = index.semantic_index
    terms = tokenize_lexically(query, index.lexical_config)
    projection = semantic.project_query(terms, index.document_frequencies)
    semantic_results = index.search(query, method="semantic", top_k=5)
    hybrid_results = index.search(query, method="hybrid", top_k=5)
    reranked_results = index.search(query, method="hybrid_reranked", top_k=5)
    index_path = index.save(output_directory / "inspection_index.json")
    loaded = RetrievalIndex.load(index_path)
    reloaded_results = loaded.search(query, method="hybrid_reranked", top_k=5)
    summary = {
        "experiment": "inspect_semantic_retrieval",
        "query": query,
        "lexical_terms": list(terms),
        "query_term_counts": dict(sorted(Counter(terms).items())),
        "query_tfidf_vector": projection.tfidf_weights,
        "latent_query_vector": projection.embedding.tolist(),
        "latent_query_raw_norm": projection.raw_norm,
        "out_of_vocabulary_terms": list(projection.out_of_vocabulary_terms),
        "latent_dimension": semantic.config.dimensions,
        "singular_values": semantic.singular_values.tolist(),
        "semantic_results": [result.to_dict() for result in semantic_results],
        "hybrid_results": [result.to_dict() for result in hybrid_results],
        "reranked_results": [result.to_dict() for result in reranked_results],
        "index_path": str(index_path),
        "index_sha256": index.index_sha256,
        "semantic_sha256": semantic.semantic_sha256,
        "reload_state_equal": loaded.state_dict() == index.state_dict(),
        "reload_results_equal": [result.to_dict() for result in reloaded_results]
        == [result.to_dict() for result in reranked_results],
        "warning": (
            "Latent coordinates are unlabeled; similarity is not proof of relevance."
        ),
    }
    atomic_write_text(
        output_directory / "semantic_retrieval_inspection.json",
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return summary


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/semantic_retrieval_inspection"),
    )
    parser.add_argument(
        "--query",
        default="why can appended suffix tokens not change an earlier prediction",
    )
    parser.add_argument("--dimensions", type=int, default=6)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    summary = run_inspection(
        args.output_directory,
        query=args.query,
        dimensions=args.dimensions,
    )
    print(
        json.dumps(
            {
                "query": summary["query"],
                "latent_dimension": summary["latent_dimension"],
                "reload_results_equal": summary["reload_results_equal"],
                "top_semantic_citations": [
                    result["citation"]["display"]
                    for result in summary["semantic_results"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
