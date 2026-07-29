"""Project-authored corpus and graded judgments for Milestone 10 evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from localml_scholar.retrieval import (
    ChunkingConfig,
    RetrievalIndex,
    SemanticRetrievalConfig,
    ingest_markdown,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "retrieval"
SEMANTIC_JUDGMENTS = FIXTURE_ROOT / "semantic_relevance.json"
SEMANTIC_SOURCES = (
    "semantic_attention.md",
    "semantic_misc.md",
    "semantic_normalization.md",
    "semantic_optimization.md",
)
SEMANTIC_CHUNKING = ChunkingConfig(
    target_characters=260,
    maximum_characters=360,
    overlap_characters=0,
    minimum_characters=40,
)


def build_semantic_fixture_index(
    *,
    dimensions: int = 6,
) -> RetrievalIndex:
    """Build and enrich the canonical authored semantic retrieval fixture."""
    documents = [
        ingest_markdown(
            (FIXTURE_ROOT / name).read_text(encoding="utf-8"),
            source=f"fixtures/retrieval/{name}",
            metadata={"logical_collection": "semantic_retrieval_fixture"},
        )
        for name in SEMANTIC_SOURCES
    ]
    lexical = RetrievalIndex.build(documents, chunking_config=SEMANTIC_CHUNKING)
    return lexical.enrich_semantic(SemanticRetrievalConfig(dimensions=dimensions))


def load_semantic_judgments(
    path: str | Path = SEMANTIC_JUDGMENTS,
) -> tuple[dict[str, Any], ...]:
    """Load validated query categories and exact graded chunk judgments."""
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(state, list) or not state:
        raise ValueError("Semantic relevance fixture must be a non-empty list.")
    required = {"query_id", "query", "category", "split", "grades"}
    records: list[dict[str, Any]] = []
    for item in state:
        if not isinstance(item, Mapping) or set(item) != required:
            raise ValueError("Semantic relevance record keys are malformed.")
        record = dict(item)
        if any(
            not isinstance(record[name], str) or not record[name]
            for name in ("query_id", "query", "category", "split")
        ):
            raise ValueError("Semantic query metadata must use non-empty strings.")
        if record["split"] not in {"development", "evaluation"}:
            raise ValueError("Semantic query split must be development or evaluation.")
        grades = record["grades"]
        if (
            not isinstance(grades, dict)
            or not grades
            or any(
                not isinstance(chunk_id, str)
                or not chunk_id
                or isinstance(grade, bool)
                or not isinstance(grade, int)
                or grade not in {1, 2}
                for chunk_id, grade in grades.items()
            )
        ):
            raise ValueError("Semantic grades must map chunk IDs to grades 1 or 2.")
        records.append(record)
    if len({record["query_id"] for record in records}) != len(records):
        raise ValueError("Semantic query IDs must be unique.")
    return tuple(records)
