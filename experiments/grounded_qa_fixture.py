"""Shared project-authored corpus and judgments for grounded-answer experiments."""

from __future__ import annotations

from pathlib import Path

from experiments.retrieval_fixture import (
    FIXTURE_CHUNKING,
    FIXTURE_ROOT,
    load_fixture_documents,
)
from localml_scholar.answering.evaluation import (
    GroundedQuestion,
    load_grounded_questions,
)
from localml_scholar.retrieval import RetrievalIndex, ingest_markdown

QUESTION_FIXTURE = FIXTURE_ROOT / "grounded_questions.json"


def load_grounded_documents():
    """Return the retrieval fixture plus authored grounding-control text."""
    grounding_path = FIXTURE_ROOT / "grounding.md"
    grounding = ingest_markdown(
        grounding_path.read_text(encoding="utf-8"),
        source="fixtures/grounding.md",
        metadata={"logical_collection": "grounded_qa"},
    )
    return (*load_fixture_documents(), grounding)


def build_grounded_fixture_index() -> RetrievalIndex:
    """Build the deterministic four-document grounded-QA snapshot."""
    return RetrievalIndex.build(
        load_grounded_documents(),
        chunking_config=FIXTURE_CHUNKING,
    )


def load_grounded_fixture_questions(
    path: str | Path = QUESTION_FIXTURE,
) -> tuple[GroundedQuestion, ...]:
    """Load exact authored grounded-answer judgments."""
    return load_grounded_questions(path)
