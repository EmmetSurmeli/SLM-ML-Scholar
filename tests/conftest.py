from __future__ import annotations

import pytest

from experiments.grounded_qa_fixture import build_grounded_fixture_index
from localml_scholar.retrieval import RetrievalIndex, ingest_file


@pytest.fixture
def grounded_index() -> RetrievalIndex:
    return build_grounded_fixture_index()


@pytest.fixture
def scholarly_index() -> RetrievalIndex:
    fixture_root = (
        __import__("pathlib").Path(__file__).parent / "fixtures" / "scholarly"
    )
    return RetrievalIndex.build(
        tuple(ingest_file(path) for path in sorted(fixture_root.glob("*.md")))
    )
