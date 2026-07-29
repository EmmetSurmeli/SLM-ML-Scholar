"""Command-line index construction, inspection, and cited lexical search."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from localml_scholar.retrieval.chunking import ChunkingConfig
from localml_scholar.retrieval.hybrid import (
    HybridRetrievalConfig,
    RerankingConfig,
)
from localml_scholar.retrieval.index import (
    RetrievalIndex,
    SearchFilters,
    highlight_matches,
)
from localml_scholar.retrieval.ingestion import ingest_files
from localml_scholar.retrieval.semantic import SemanticRetrievalConfig


def _build(args: argparse.Namespace) -> dict:
    documents = ingest_files(args.sources)
    index = RetrievalIndex.build(
        documents,
        chunking_config=ChunkingConfig(
            target_characters=args.target_characters,
            maximum_characters=args.maximum_characters,
            overlap_characters=args.overlap_characters,
            minimum_characters=args.minimum_characters,
        ),
    )
    if args.semantic_dimensions is not None:
        index = index.enrich_semantic(
            SemanticRetrievalConfig(dimensions=args.semantic_dimensions)
        )
    path = index.save(args.output)
    return {
        "operation": "build",
        "index": str(path),
        "index_sha256": index.index_sha256,
        "corpus_sha256": index.corpus_sha256,
        "documents": len(index.documents),
        "sections": sum(len(document.sections) for document in index.documents),
        "chunks": len(index.chunks),
        "vocabulary_size": len(index.vocabulary),
        "average_chunk_length": index.average_chunk_length,
        "semantic_available": index.semantic_index is not None,
    }


def _inspect(args: argparse.Namespace) -> dict:
    index = RetrievalIndex.load(args.index)
    return {
        "operation": "inspect",
        "index_sha256": index.index_sha256,
        "corpus_sha256": index.corpus_sha256,
        "package_version": index.package_version,
        "documents": [document.to_dict() for document in index.documents],
        "chunks": [chunk.to_dict() for chunk in index.chunks],
        "vocabulary": list(index.vocabulary),
        "document_frequencies": index.document_frequencies,
        "average_chunk_length": index.average_chunk_length,
        "index_format_version": index.index_format_version,
        "semantic": (
            None
            if index.semantic_index is None
            else {
                "semantic_sha256": index.semantic_index.semantic_sha256,
                "configuration": index.semantic_index.config.to_dict(),
                "matrix_shape": list(index.semantic_index.matrix_shape),
                "effective_rank": index.semantic_index.effective_rank,
                "singular_values": index.semantic_index.singular_values.tolist(),
                "reconstruction_error": index.semantic_index.reconstruction_error,
                "explained_squared_singular_fraction": (
                    index.semantic_index.explained_squared_singular_fraction
                ),
                "zero_row_indices": list(index.semantic_index.zero_row_indices),
            }
        ),
    }


def _enrich(args: argparse.Namespace) -> dict:
    index = RetrievalIndex.load(args.index)
    enriched = index.enrich_semantic(
        SemanticRetrievalConfig(
            dimensions=args.dimensions,
            normalize_embeddings=not args.no_normalize_embeddings,
        )
    )
    destination = enriched.save(args.output)
    return {
        "operation": "enrich",
        "input_index": str(args.index),
        "output_index": str(destination),
        "original_index_sha256": index.index_sha256,
        "index_sha256": enriched.index_sha256,
        "corpus_sha256": enriched.corpus_sha256,
        "semantic_sha256": enriched.semantic_index.semantic_sha256,
        "semantic_configuration": enriched.semantic_index.config.to_dict(),
        "chunk_ids_preserved": [chunk.chunk_id for chunk in index.chunks]
        == [chunk.chunk_id for chunk in enriched.chunks],
    }


def _search(args: argparse.Namespace) -> dict:
    index = RetrievalIndex.load(args.index)
    filters = SearchFilters(
        document_id=args.document_id,
        source_name=args.source_name,
        media_type=args.media_type,
        heading_path_prefix=tuple(args.heading_prefix or ()),
        publication_year=args.publication_year,
        logical_collection=args.collection,
    )
    method = (
        "hybrid_reranked" if args.method == "hybrid" and args.rerank else args.method
    )
    if args.rerank and args.method != "hybrid":
        raise ValueError("--rerank requires --method hybrid.")
    hybrid_config = None
    reranking_config = None
    if method in {"hybrid", "hybrid_reranked"}:
        hybrid_config = HybridRetrievalConfig(
            lexical_method=args.lexical_method,
            fusion=args.fusion,
            alpha=args.alpha,
            rrf_rank_constant=args.rrf_rank_constant,
            lexical_candidate_count=args.lexical_candidate_count,
            semantic_candidate_count=args.semantic_candidate_count,
        )
    if method == "hybrid_reranked":
        reranking_config = RerankingConfig(
            candidate_count=args.reranking_candidate_count,
            redundancy_threshold=args.redundancy_threshold,
        )
    results = index.search(
        args.query,
        method=method,
        top_k=args.top_k,
        filters=filters,
        hybrid_config=hybrid_config,
        reranking_config=reranking_config,
    )
    serialized = []
    for result in results:
        state = result.to_dict()
        if args.verbose:
            state["highlighted_text"] = highlight_matches(
                result.text,
                result.matched_terms,
            )
        else:
            state.pop("term_contributions")
            state.pop("scoring_details")
        serialized.append(state)
    return {
        "operation": "search",
        "query": args.query,
        "method": method,
        "retrieval_configuration": {
            "hybrid": (None if hybrid_config is None else hybrid_config.to_dict()),
            "reranking": (
                None if reranking_config is None else reranking_config.to_dict()
            ),
        },
        "top_k": args.top_k,
        "result_count": len(serialized),
        "results": serialized,
        "answer_generated": False,
    }


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or search a deterministic local lexical index."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--sources", type=Path, nargs="+", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--target-characters", type=int, default=600)
    build.add_argument("--maximum-characters", type=int, default=900)
    build.add_argument("--overlap-characters", type=int, default=100)
    build.add_argument("--minimum-characters", type=int, default=80)
    build.add_argument("--semantic-dimensions", type=int)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--index", type=Path, required=True)
    enrich = subparsers.add_parser("enrich")
    enrich.add_argument("--index", type=Path, required=True)
    enrich.add_argument("--output", type=Path, required=True)
    enrich.add_argument("--dimensions", type=int, default=1)
    enrich.add_argument("--no-normalize-embeddings", action="store_true")
    search = subparsers.add_parser("search")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument(
        "--method",
        choices=("bm25", "tfidf", "semantic", "hybrid", "hybrid_reranked"),
        default="bm25",
    )
    search.add_argument("--lexical-method", choices=("bm25", "tfidf"), default="bm25")
    search.add_argument("--fusion", choices=("weighted", "rrf"), default="rrf")
    search.add_argument("--alpha", type=float, default=0.5)
    search.add_argument("--rrf-rank-constant", type=int, default=60)
    search.add_argument("--lexical-candidate-count", type=int, default=20)
    search.add_argument("--semantic-candidate-count", type=int, default=20)
    search.add_argument("--rerank", action="store_true")
    search.add_argument("--reranking-candidate-count", type=int, default=20)
    search.add_argument("--redundancy-threshold", type=float, default=0.8)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--document-id")
    search.add_argument("--source-name")
    search.add_argument("--media-type")
    search.add_argument("--heading-prefix", action="append")
    search.add_argument("--publication-year", type=int)
    search.add_argument("--collection")
    search.add_argument("--verbose", action="store_true")
    search.add_argument("--json", action="store_true")
    return parser.parse_args(arguments)


def _human_readable(payload: dict) -> str:
    if payload["operation"] != "search":
        return json.dumps(payload, indent=2, ensure_ascii=False)
    lines = [
        f"query: {payload['query']}",
        f"method: {payload['method']}",
        f"results: {payload['result_count']}",
    ]
    for result in payload["results"]:
        lines.extend(
            [
                "",
                f"#{result['rank']} score={result['score']:.12f}",
                f"source: {result['source_name']}",
                f"citation: {result['citation']['display']}",
            ]
        )
        if "term_contributions" in result:
            lines.extend(
                [
                    f"matched terms: {', '.join(result['matched_terms'])}",
                    "scoring details: "
                    + json.dumps(
                        result["scoring_details"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "term contributions: "
                    + json.dumps(
                        result["term_contributions"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    f"highlighted: {result['highlighted_text']}",
                ]
            )
        lines.append(result["text"])
    return "\n".join(lines)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    if args.command == "build":
        payload = _build(args)
    elif args.command == "inspect":
        payload = _inspect(args)
    elif args.command == "enrich":
        payload = _enrich(args)
    else:
        payload = _search(args)
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(_human_readable(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
