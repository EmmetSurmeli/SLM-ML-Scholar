"""Evaluate citation-preserving reproduction checklists."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.scholarly_fixture import (  # noqa: E402
    build_pipeline,
    document_ids,
    load_scholarly_judgments,
    write_report,
)
from localml_scholar.scholarly import (  # noqa: E402
    citation_coverage,
    extraction_metrics,
)


def run(output: Path) -> dict:
    index, pipeline = build_pipeline()
    ids = document_ids(index)
    judgments = load_scholarly_judgments()
    papers = {}
    for source_name, expected in sorted(judgments.items()):
        checklist = pipeline.build_reproduction_checklist(ids[source_name])
        reasons = [item.reason for item in checklist.risk_flags]
        risk_metrics = extraction_metrics(reasons, expected["risk_flags"])
        by_status = {
            status: [item for item in checklist.items if item.status == status]
            for status in ("found", "not_found", "ambiguous", "conflicting")
        }
        status_metrics = {
            status: extraction_metrics(
                [item.item for item in values],
                expected["checklist_statuses"][status],
            ).to_dict()
            for status, values in by_status.items()
        }
        cited_values = [
            value.to_dict()
            for status in ("found", "ambiguous", "conflicting")
            for item in by_status[status]
            for value in item.values
        ]
        papers[source_name] = {
            "found_field_count": len(by_status["found"]),
            "missing_field_count": len(by_status["not_found"]),
            "ambiguity_count": len(by_status["ambiguous"]),
            "conflict_count": len(by_status["conflicting"]),
            "found_field_metrics": status_metrics["found"],
            "missing_field_detection": status_metrics["not_found"],
            "ambiguity_detection": status_metrics["ambiguous"],
            "conflict_detection": status_metrics["conflicting"],
            "risk_flag_metrics": risk_metrics.to_dict(),
            "citation_coverage": citation_coverage(cited_values),
            "example_checklist": checklist.to_dict(),
            "failure_cases": {
                "risk_flags": sorted(set(reasons) ^ set(expected["risk_flags"])),
                "statuses": [
                    status
                    for status, metrics in status_metrics.items()
                    if metrics["exact_value_accuracy"] != 1.0
                ],
            },
        }
    report = {
        "experiment": "authored_reproduction_checklists",
        "index_sha256": index.index_sha256,
        "papers": papers,
        "interpretation": (
            "Risk flags are document-completeness observations, not proof that "
            "reproduction is impossible."
        ),
    }
    write_report(output, report)
    return report


if __name__ == "__main__":
    destination = (
        PROJECT_ROOT / "outputs" / "reproduction_checklists" / "evaluation.json"
    )
    result = run(destination)
    print(
        {
            name: {
                "found": values["found_field_count"],
                "risk_f1": values["risk_flag_metrics"]["f1"],
            }
            for name, values in result["papers"].items()
        }
    )
