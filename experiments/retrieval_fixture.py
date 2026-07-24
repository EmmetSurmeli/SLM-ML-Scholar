"""Shared project-authored deterministic retrieval fixture."""

from __future__ import annotations

import json
from pathlib import Path

from localml_scholar.retrieval import (
    ChunkingConfig,
    Document,
    RetrievalIndex,
    ingest_markdown,
    ingest_plain_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "retrieval"
FIXTURE_CHUNKING = ChunkingConfig(
    target_characters=220,
    maximum_characters=300,
    overlap_characters=30,
    minimum_characters=40,
)


def load_fixture_documents() -> tuple[Document, ...]:
    """Load fixture contents with portable logical source identifiers."""
    documents: list[Document] = []
    for name in ("attention.md", "optimization.md", "probability.txt"):
        text = (FIXTURE_ROOT / name).read_text(encoding="utf-8")
        logical_source = f"fixtures/retrieval/{name}"
        if name.endswith(".md"):
            document = ingest_markdown(text, source=logical_source)
        else:
            document = ingest_plain_text(text, source=logical_source)
        documents.append(document)
    return tuple(documents)


def build_fixture_index() -> RetrievalIndex:
    """Build the canonical fixture index used by tests and experiments."""
    return RetrievalIndex.build(
        load_fixture_documents(),
        chunking_config=FIXTURE_CHUNKING,
    )


def load_fixture_relevance() -> dict[str, list[str]]:
    """Load committed exact chunk-ID relevance judgments."""
    state = json.loads((FIXTURE_ROOT / "relevance.json").read_text(encoding="utf-8"))
    if not isinstance(state, dict) or not all(
        isinstance(query, str)
        and isinstance(chunk_ids, list)
        and all(isinstance(chunk_id, str) for chunk_id in chunk_ids)
        for query, chunk_ids in state.items()
    ):
        raise ValueError("Fixture relevance judgments are malformed.")
    return state
