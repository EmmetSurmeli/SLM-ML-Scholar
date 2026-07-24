from __future__ import annotations

import pytest

from experiments.grounded_qa_fixture import build_grounded_fixture_index
from localml_scholar.retrieval import RetrievalIndex


@pytest.fixture
def grounded_index() -> RetrievalIndex:
    return build_grounded_fixture_index()
