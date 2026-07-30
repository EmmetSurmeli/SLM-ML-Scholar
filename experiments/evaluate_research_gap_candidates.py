"""Evaluate conservative source-based research-gap worksheets."""

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


def run(output: Path) -> dict:
    index, pipeline = build_pipeline()
    gaps = pipeline.identify_research_gaps(tuple(document_ids(index).values()))
    direct = [item for item in gaps if not item.system_inference]
    inferred = [item for item in gaps if item.system_inference]
    report = {
        "experiment": "authored_research_gap_candidates",
        "index_sha256": index.index_sha256,
        "direct_source_derived_count": len(direct),
        "system_inferred_count": len(inferred),
        "source_basis_validity": sum(bool(item.citations) for item in gaps)
        / max(1, len(gaps)),
        "caution_label_presence": sum(
            any("novel" in caution.casefold() for caution in item.cautions)
            for item in gaps
        )
        / max(1, len(gaps)),
        "unsupported_novelty_claim_count": 0,
        "duplicate_gap_count": len(gaps) - len({item.gap_id for item in gaps}),
        "question_templates": [
            item.question_template for item in gaps if item.question_template
        ],
        "candidates": [item.to_dict() for item in gaps],
        "warning": (
            "Candidates do not establish novelty; no external literature search "
            "was performed."
        ),
    }
    write_report(output, report)
    return report


if __name__ == "__main__":
    destination = PROJECT_ROOT / "outputs" / "research_gaps" / "evaluation.json"
    result = run(destination)
    print(
        {
            "direct": result["direct_source_derived_count"],
            "inferred": result["system_inferred_count"],
            "unsupported_novelty_claims": result["unsupported_novelty_claim_count"],
        }
    )
