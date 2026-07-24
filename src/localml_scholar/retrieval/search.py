"""Command-line index construction, inspection, and cited lexical search."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from localml_scholar.retrieval.chunking import ChunkingConfig
from localml_scholar.retrieval.index import (
    RetrievalIndex,
    SearchFilters,
    highlight_matches,
)
from localml_scholar.retrieval.ingestion import ingest_files


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
    results = index.search(
        args.query,
        method=args.method,
        top_k=args.top_k,
        filters=filters,
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
        "method": args.method,
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
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--index", type=Path, required=True)
    search = subparsers.add_parser("search")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--method", choices=("bm25", "tfidf"), default="bm25")
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
    else:
        payload = _search(args)
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(_human_readable(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
