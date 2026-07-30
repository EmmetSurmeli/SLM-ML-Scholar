"""Inspect every major scholarly artifact for one authored paper."""

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
from localml_scholar.scholarly import (  # noqa: E402
    load_analysis,
    render_checklist_markdown,
    render_notation_markdown,
    save_analysis,
)


def run(output_directory: Path) -> dict:
    index, pipeline = build_pipeline()
    document_id = document_ids(index)["sparse_gate_network.md"]
    analysis = pipeline.analyze_paper(document_id)
    summary = pipeline.summarize_paper(document_id)
    checklist = pipeline.build_reproduction_checklist(document_id)
    index_path = index.save(output_directory / "fixture_index.json")
    analysis_path = save_analysis(
        analysis,
        output_directory / "paper_analysis.json",
        index=index,
        config=pipeline.config,
    )
    reloaded = load_analysis(analysis_path, index=index)
    report = {
        "experiment": "scholarly_analysis_inspection",
        "index_path": str(index_path),
        "analysis_path": str(analysis_path),
        "index_sha256": index.index_sha256,
        "metadata": analysis.paper.to_dict(),
        "section_roles": [item.to_dict() for item in analysis.paper.sections],
        "equations": [item.to_dict() for item in analysis.equations],
        "equation_analyses": [item.to_dict() for item in analysis.equation_analyses],
        "notation_glossary": [item.to_dict() for item in analysis.notation],
        "unresolved_symbols": list(analysis.unresolved_symbols),
        "assumptions": [item.to_dict() for item in analysis.assumptions],
        "methodology": [item.to_dict() for item in analysis.methodology],
        "datasets": [item.to_dict() for item in analysis.datasets],
        "metrics": [item.to_dict() for item in analysis.metrics],
        "hyperparameters": [item.to_dict() for item in analysis.hyperparameters],
        "experiments": [item.to_dict() for item in analysis.experiments],
        "results": [item.to_dict() for item in analysis.results],
        "ablations": [item.to_dict() for item in analysis.ablations],
        "limitations": [item.to_dict() for item in analysis.limitations],
        "references": [item.to_dict() for item in analysis.paper.references],
        "summary": summary.to_dict(),
        "reproduction_checklist": checklist.to_dict(),
        "risk_flags": [item.to_dict() for item in checklist.risk_flags],
        "notation_markdown": render_notation_markdown(analysis),
        "checklist_markdown": render_checklist_markdown(checklist),
        "artifact_reload_consistent": reloaded.to_dict() == analysis.to_dict(),
        "transformer_constructed": False,
    }
    write_report(output_directory / "inspection.json", report)
    return report


if __name__ == "__main__":
    destination = PROJECT_ROOT / "outputs" / "scholarly_inspection"
    result = run(destination)
    print(
        {
            "equations": len(result["equations"]),
            "notation_entries": len(result["notation_glossary"]),
            "reload_consistent": result["artifact_reload_consistent"],
        }
    )
