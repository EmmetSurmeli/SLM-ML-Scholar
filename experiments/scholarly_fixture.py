"""Shared project-authored fixtures for Milestone 11 experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from localml_scholar.retrieval import RetrievalIndex, ingest_file  # noqa: E402
from localml_scholar.scholarly import ScholarlyAnalysisPipeline  # noqa: E402

FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "scholarly"


def build_scholarly_fixture_index() -> RetrievalIndex:
    """Build a deterministic index over three concise authored papers."""
    paths = sorted(FIXTURE_ROOT.glob("*.md"))
    return RetrievalIndex.build(tuple(ingest_file(path) for path in paths))


def load_scholarly_judgments() -> dict[str, Any]:
    """Load manually authored expected fields."""
    return json.loads((FIXTURE_ROOT / "judgments.json").read_text(encoding="utf-8"))


def build_pipeline() -> tuple[RetrievalIndex, ScholarlyAnalysisPipeline]:
    """Build the fixture index and transformer-independent pipeline."""
    index = build_scholarly_fixture_index()
    return index, ScholarlyAnalysisPipeline(index)


def document_ids(index: RetrievalIndex) -> dict[str, str]:
    """Map stable source names to document IDs."""
    return {document.source_name: document.document_id for document in index.documents}


def write_report(path: Path, payload: dict[str, Any]) -> Path:
    """Write one deterministic, human-inspectable experiment report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
