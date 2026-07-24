#!/usr/bin/env python3
"""Inspect exact document, section, chunk, index, and citation behavior."""

from __future__ import annotations

import argparse
import json
import sys
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
    FIXTURE_CHUNKING,
    load_fixture_documents,
)
from localml_scholar._version import __version__  # noqa: E402
from localml_scholar.retrieval import (  # noqa: E402
    RetrievalIndex,
    highlight_matches,
    lexical_terms,
    validate_chunk_coverage,
)
from localml_scholar.retrieval.bm25 import (  # noqa: E402
    bm25_inverse_document_frequency,
)
from localml_scholar.retrieval.tfidf import (  # noqa: E402
    smooth_inverse_document_frequency,
)
from localml_scholar.serialization import atomic_write_text  # noqa: E402


def inspect_document_ingestion(
    *,
    output_directory: str | Path = "outputs/document_ingestion_inspection",
) -> dict[str, Any]:
    """Save a complete transparent inspection of the attention fixture."""
    documents = load_fixture_documents()
    document = next(item for item in documents if item.source_name == "attention.md")
    index = RetrievalIndex.build([document], chunking_config=FIXTURE_CHUNKING)
    for item in index.documents:
        validate_chunk_coverage(
            item,
            [chunk for chunk in index.chunks if chunk.document_id == item.document_id],
            index.chunking_config,
        )
    query = "How does causal masking prevent future token leakage?"
    results = index.search(query, method="bm25", top_k=3)
    destination = Path(output_directory)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.mkdir(parents=True, exist_ok=True)
    index_path = index.save(destination / "index.json")
    reloaded = RetrievalIndex.load(index_path)
    reloaded_results = reloaded.search(query, method="bm25", top_k=3)
    term_statistics = {
        term: {
            "document_frequency": frequency,
            "tfidf_idf": smooth_inverse_document_frequency(
                len(index.chunks),
                frequency,
            ),
            "bm25_idf": bm25_inverse_document_frequency(
                len(index.chunks),
                frequency,
            ),
        }
        for term, frequency in index.document_frequencies.items()
    }
    summary: dict[str, Any] = {
        "milestone": 8,
        "package_version": __version__,
        "purpose": "deterministic document-ingestion and retrieval inspection",
        "source": document.to_dict(),
        "detected_headings": [
            {
                "heading": section.heading,
                "heading_path": list(section.heading_path),
                "level": section.level,
                "start_character": section.start_character,
                "end_character": section.end_character,
                "start_line": section.start_line,
                "end_line": section.end_line,
            }
            for section in document.sections
        ],
        "chunking_configuration": FIXTURE_CHUNKING.to_dict(),
        "chunks": [chunk.to_dict() for chunk in index.chunks],
        "chunk_terms": {
            chunk.chunk_id: [
                {
                    "term": term.term,
                    "start_character": term.start_character,
                    "end_character": term.end_character,
                }
                for term in lexical_terms(chunk.text)
            ]
            for chunk in index.chunks
        },
        "term_statistics": term_statistics,
        "index": {
            "index_sha256": index.index_sha256,
            "corpus_sha256": index.corpus_sha256,
            "vocabulary_size": len(index.vocabulary),
            "average_chunk_length": index.average_chunk_length,
        },
        "reconstruction": {
            "all_chunk_slices_exact": all(
                chunk.text == document.text[chunk.start_character : chunk.end_character]
                for chunk in index.chunks
            ),
            "coverage_validation_passed": True,
        },
        "query": query,
        "ranked_passages": [
            {
                **result.to_dict(),
                "highlighted_text": highlight_matches(
                    result.text,
                    result.matched_terms,
                ),
            }
            for result in results
        ],
        "index_round_trip": {
            "state_exact": reloaded.state_dict() == index.state_dict(),
            "results_exact": [result.to_dict() for result in reloaded_results]
            == [result.to_dict() for result in results],
        },
        "answer_generated": False,
        "pdf_policy": "externally supplied page text only; no PDF parser or OCR",
    }
    summary_path = destination / "run_summary.json"
    summary["artifacts"] = {
        "index": str(index_path),
        "summary": str(summary_path),
    }
    atomic_write_text(
        summary_path,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    print(f"document: {document.document_id} ({document.source_name})")
    print(f"sections: {len(document.sections)}; chunks: {len(index.chunks)}")
    for section in document.sections:
        print(
            f"section {section.ordinal}: {section.heading_path} "
            f"chars={section.start_character}:{section.end_character} "
            f"lines={section.start_line}:{section.end_line}"
        )
    for result in results:
        print(
            f"rank={result.rank} score={result.score:.12f} "
            f"citation={result.citation.format()}"
        )
        print(result.text)
    print(f"index round trip exact: {summary['index_round_trip']}")
    print(f"summary: {summary_path}")
    return summary


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect deterministic Markdown ingestion and lexical retrieval."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/document_ingestion_inspection"),
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    inspect_document_ingestion(output_directory=args.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
