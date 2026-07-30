"""Evaluate structured comparison and invalid-comparison warnings."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.scholarly_fixture import (  # noqa: E402
    build_pipeline,
    document_ids,
    write_report,
)
from localml_scholar.scholarly import render_comparison_markdown  # noqa: E402


def run(output: Path) -> dict:
    index, pipeline = build_pipeline()
    ids = document_ids(index)
    comparison = pipeline.compare_papers(
        (
            ids["sparse_gate_network.md"],
            ids["dense_gate_companion.md"],
        )
    )
    result_dimension = next(
        item for item in comparison.dimensions if item.name == "key_results"
    )
    all_values = [
        item
        for dimension in comparison.dimensions
        for values in dimension.values_by_paper.values()
        for item in values
    ]
    report = {
        "experiment": "authored_cross_paper_comparison",
        "index_sha256": index.index_sha256,
        "shared_dimension_count": sum(
            item.relationship == "shared" for item in comparison.dimensions
        ),
        "different_dimension_count": sum(
            item.relationship == "different" for item in comparison.dimensions
        ),
        "incomparable_dimension_count": sum(
            not item.comparable for item in comparison.dimensions
        ),
        "result_comparable": result_dimension.comparable,
        "invalid_comparison_warnings": list(result_dimension.warnings),
        "citation_coverage": (
            sum(bool(item.citation) for item in all_values) / max(1, len(all_values))
        ),
        "false_superiority_claim_count": comparison.false_superiority_claim_count,
        "comparison": comparison.to_dict(),
        "example_comparison_table": render_comparison_markdown(comparison),
    }
    write_report(output, report)
    return report


if __name__ == "__main__":
    destination = PROJECT_ROOT / "outputs" / "paper_comparison" / "evaluation.json"
    result = run(destination)
    print(
        {
            "incomparable_dimensions": result["incomparable_dimension_count"],
            "false_superiority_claims": result["false_superiority_claim_count"],
            "citation_coverage": result["citation_coverage"],
        }
    )
